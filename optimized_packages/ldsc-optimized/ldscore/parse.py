'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module contains functions for parsing various ldsc-defined file formats.

'''


import numpy as np
import pandas as pd
import os
import glob
import zlib
from io import BytesIO
import hashlib
from multiprocessing.dummy import Pool as _ThreadPool
from multiprocessing import cpu_count as _cpu_count


def series_eq(x, y):
    '''Compare series, return False if lengths not equal.'''
    # Compare as object labels: identical semantics for object dtype, and
    # well-defined for Categoricals even when their categories differ
    # (pandas 0.20 raises TypeError on == between different-categories
    # Categoricals; the callers expect a plain False instead).
    return len(x) == len(y) and (
        np.asarray(x, dtype=object) == np.asarray(y, dtype=object)).all()


def _gunzip_bytes(buf):
    '''One-shot gzip decompression (all members) via zlib; avoids the gzip.py read layer.'''
    parts = []
    while buf:
        d = zlib.decompressobj(31)
        parts.append(d.decompress(buf))
        parts.append(d.flush())
        buf = d.unused_data
    return b''.join(parts)


def _read_gunzip(path):
    with open(path, 'rb') as f:
        return _gunzip_bytes(f.read())


# Bounded threaded prefetch cache for gzipped chromosome-split files. zlib
# decompression releases the GIL, so worker threads overlap decompression with
# pandas parsing in the main thread. At most `ahead` decompressed buffers are
# held at once; consumption order (and hence all downstream concatenation order
# and results) is identical to sequential reading.
_PREFETCH_POOL = None
_PREFETCH = {}
_PREFETCH_ORDER = []
_PREFETCH_NEXT = 0
_PREFETCH_AHEAD = 4


def prefetch_gzip(paths, ahead=_PREFETCH_AHEAD):
    '''Start background read+decompress of the given gzip paths (in order).'''
    global _PREFETCH_POOL, _PREFETCH, _PREFETCH_ORDER, _PREFETCH_NEXT
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = _ThreadPool(4)
    _PREFETCH = {}
    _PREFETCH_ORDER = list(paths)
    for p in _PREFETCH_ORDER[:ahead]:
        _PREFETCH[p] = _PREFETCH_POOL.apply_async(_read_gunzip, (p,))
    _PREFETCH_NEXT = min(ahead, len(_PREFETCH_ORDER))


def _prefetch_get(path):
    global _PREFETCH_NEXT
    data = _PREFETCH.pop(path).get()
    if _PREFETCH_NEXT < len(_PREFETCH_ORDER):
        p = _PREFETCH_ORDER[_PREFETCH_NEXT]
        _PREFETCH[p] = _PREFETCH_POOL.apply_async(_read_gunzip, (p,))
        _PREFETCH_NEXT += 1
    return data


# Thread pool for concurrent parsing of chromosome-split files. The pandas C
# tokenizer releases the GIL, so parsing scales across threads (~2.5x at 16
# threads on the ref-ld workload). Each file is parsed independently by the
# exact same code path as serial parsing, and map() preserves path order, so
# the concatenated result is identical to sequential parsing.
_PARSE_POOL = None
# Concurrency cap: each in-flight parse holds the file's decompressed bytes
# (~65-90 MB for large chromosomes), so thread count sets the peak-RSS delta.
_PARSE_THREADS = 6

# In-flight async parses / cache lookups keyed by path tuple.
# presubmit_l2_parse() lets a caller submit a chr-split parse before it is
# needed so that independent main-thread work (e.g. sumstats parsing) overlaps
# with decompression + tokenization in the pool. Purely an
# execution-scheduling hint: consumption is order-preserving and uses the same
# per-file code path as serial parsing.
#
# Content-keyed parse cache: parsing the 22-file chr-split reference panels
# dominates --h2/--rg runtime, and realistic workflows run many phenotypes
# against the same panel. The cache stores the exact DataFrame returned by
# ldscore() (pickle preserves float64 bits, dtypes, index and columns
# exactly), keyed by SHA-256 over the cache format version, the pandas
# version, this file's source, and the compressed bytes of every input file -
# so any change to the data or the parser invalidates it. Cold runs pay one
# extra hash + pickle write; warm runs skip tokenization entirely.
_PRESUBMIT = {}
_CACHE_VERSION = 'l2cache-v4'  # v4: v3 + string categories as .npy sidecars (v3: raw .npy float block + meta sidecar; v2: SNP as Categorical)
_CACHE_DIRNAME = '.ldsc_cache'


def _parse_pool():
    global _PARSE_POOL
    if _PARSE_POOL is None:
        n = min(_PARSE_THREADS, _cpu_count() or 1)
        _PARSE_POOL = _ThreadPool(max(n, 1))
    return _PARSE_POOL


def _l2_paths(fh, num):
    '''Resolve a chr-split .l2.ldscore fileset to (paths, compression).'''
    suffix = '.l2.ldscore'
    chrs = get_present_chrs(fh, num + 1)
    first_fh = sub_chr(fh, chrs[0]) + suffix
    s, compression = which_compression(first_fh)
    paths = [sub_chr(fh, i) + suffix + s for i in chrs]
    return paths, compression


def _hash_file(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(_bufsize)
            if not b:
                break
            h.update(b)
    return h.digest()


def _b(x):
    """bytes for hashlib: encode str, pass bytes through."""
    return x.encode('utf-8') if isinstance(x, str) else x



def _parser_source_path():
    '''Path to this module's .py source (never the .pyc: __file__ points to
    bytecode only when a valid .pyc already exists, which would make the cache
    key depend on bytecode-cache state).'''
    p = os.path.abspath(__file__)
    if p.endswith('.pyc') and os.path.exists(p[:-1]):
        p = p[:-1]
    return p


_KEY_MEMO = {}


def _cache_key(paths):
    '''Content key: parser source + pandas version + per-file gz bytes.'''
    t = tuple(paths)
    if t in _KEY_MEMO:
        return _KEY_MEMO[t]
    h = hashlib.sha256()
    h.update(_b(_CACHE_VERSION))
    h.update(_b(pd.__version__))
    with open(_parser_source_path(), 'rb') as f:
        h.update(f.read())
    for p, d in zip(paths, _parse_pool().map(_hash_file, paths)):
        h.update(_b(p))
        h.update(d)
    _KEY_MEMO[t] = h.hexdigest()
    return _KEY_MEMO[t]


def _cache_file(paths):
    base = os.path.dirname(os.path.abspath(paths[0]))
    return os.path.join(base, _CACHE_DIRNAME, _cache_key(paths) + '.pkl')


def _v3_decomposable(frame):
    '''True when a frame can be stored as a raw float64 .npy block + meta.

    Requires an all-float64/category column set (a single homogeneous block)
    and an integer or default range index. Anything else falls back to the
    whole-frame pickle, which preserves arbitrary objects exactly.
    '''
    if not isinstance(frame, pd.DataFrame):
        return False
    idx = frame.index
    if not (isinstance(idx, pd.RangeIndex) or idx.dtype == np.int64):
        return False
    for c in frame.columns:
        dt = frame[c].dtype
        if not (dt == np.float64 or str(dt) == 'category'):
            return False
    return True


def _cache_read(path):
    try:
        obj = pd.read_pickle(path)
        if type(obj) is dict and obj.get('__v4__'):
            # v4: raw float block in <path>.npy + categorical/index metadata;
            # string categories live in <path>.cat<I>.npy sidecars because one
            # np.load beats unpickling ~1M individual str objects.
            blk = np.load(path + '.npy')
            d = pd.DataFrame(blk, columns=obj['float_cols'])
            for c in obj['cat_cols']:  # insert at original positions: no reindex copy
                codes, spec = obj['cats'][c]
                kind, payload = spec
                cats = np.load('{P}.cat{I}.npy'.format(P=path, I=payload)) \
                    if kind == 'npy' else payload
                d.insert(obj['pos'][c], c,
                         pd.Categorical.from_codes(codes, pd.Index(cats)))
            if obj['index'] is not None:
                d.index = pd.Int64Index(obj['index'])
            return d
        if type(obj) is dict and obj.get('__v3__'):
            # v3: raw float block in <path>.npy + categorical/index metadata.
            blk = np.load(path + '.npy')
            d = pd.DataFrame(blk, columns=obj['float_cols'])
            for c in obj['cat_cols']:  # insert at original positions: no reindex copy
                codes, cats = obj['cats'][c]
                d.insert(obj['pos'][c], c,
                         pd.Categorical.from_codes(codes, pd.Index(cats)))
            if obj['index'] is not None:
                d.index = pd.Int64Index(obj['index'])
            return d
        return obj
    except Exception:
        return None


def _cache_write(path, frame):
    try:
        d = os.path.dirname(path)
        if not os.path.isdir(d):
            os.makedirs(d)
        tmp = '{T}.tmp{P}'.format(T=path, P=os.getpid())
        if _v3_decomposable(frame):
            colnames = list(frame.columns)
            float_cols = [c for c in colnames if frame[c].dtype == np.float64]
            cat_cols = [c for c in colnames if c not in float_cols]
            np.save(tmp + '.npy', np.ascontiguousarray(frame[float_cols].values))
            idx = frame.index
            # Default RangeIndex(0, n, 1)? (pandas 0.20 RangeIndex has no
            # .start/.step attributes, so identify it by endpoint values.)
            if isinstance(idx, pd.RangeIndex) and (len(idx) == 0 or
                    (idx[0] == 0 and idx[-1] == len(idx) - 1)):
                idx_vals = None  # reconstructed frame already has this index
            else:
                idx_vals = np.asarray(idx)
            cats = {}
            cat_npys = []
            for i, c in enumerate(cat_cols):
                codes = np.asarray(frame[c].cat.codes)
                clist = list(frame[c].cat.categories)
                arr = np.asarray(clist)
                if arr.dtype.kind in 'SU' and arr.ndim == 1 and len(clist) > 0:
                    # string categories: sidecar npy (fixed-width bytes/unicode
                    # round-trips through pd.Index value-identically, ~8x
                    # faster to load than ~1M pickled str objects)
                    np.save('{T}.cat{I}.npy'.format(T=tmp, I=i), arr)
                    cat_npys.append(i)
                    cats[c] = (codes, ('npy', i))
                else:
                    cats[c] = (codes, ('list', clist))
            meta = {'__v4__': True, 'colnames': colnames, 'float_cols': float_cols,
                    'cat_cols': cat_cols, 'cats': cats,
                    'pos': dict((c, colnames.index(c)) for c in cat_cols),
                    'index': idx_vals}
            pd.to_pickle(meta, tmp)
            os.rename(tmp + '.npy', path + '.npy')
            for i in cat_npys:
                os.rename('{T}.cat{I}.npy'.format(T=tmp, I=i),
                          '{P}.cat{I}.npy'.format(P=path, I=i))
            os.rename(tmp, path)  # commit point: a v4 pkl without its npys heals as a miss
        else:
            pd.to_pickle(frame, tmp)
            os.rename(tmp, path)
    except Exception:
        pass


def l2_content_key(fh, num):
    '''Content key of a chr-split .l2.ldscore fileset (None when not chr-split).

    Hits the per-process memo once the fileset has been parsed in this
    process, so this never re-hashes file bytes on the normal h2/rg path.
    '''
    if fh is None or num is None:
        return None
    try:
        paths, _ = _l2_paths(fh, num)
    except (IndexError, IOError, OSError):
        return None  # fileset does not resolve as chr-split files
    return _cache_key(paths)


def _fileset_cache_dir(fh, num):
    paths, _ = _l2_paths(fh, num)
    return os.path.join(os.path.dirname(os.path.abspath(paths[0])),
                        _CACHE_DIRNAME)


def l2_derived_get(fh, num, tag):
    '''Read a small content-keyed derived artifact for a fileset.

    None on miss, on corrupt entries, or when the fileset is not a chr-split
    cached fileset. The tag namespaces the artifact kind and must be bumped
    when the derivation logic changes.
    '''
    k = l2_content_key(fh, num)
    if k is None:
        return None
    h = hashlib.sha256()
    h.update(_b(tag))
    h.update(_b(k))
    return _cache_read(os.path.join(_fileset_cache_dir(fh, num),
                                    h.hexdigest() + '.pkl'))


def l2_derived_put(fh, num, tag, obj):
    '''Atomically store a small content-keyed derived artifact.'''
    k = l2_content_key(fh, num)
    if k is None:
        return
    h = hashlib.sha256()
    h.update(_b(tag))
    h.update(_b(k))
    _cache_write(os.path.join(_fileset_cache_dir(fh, num),
                              h.hexdigest() + '.pkl'), obj)


def l2_pair_codes(ref_fh, w_fh, num, ref_cats, w_snp):
    '''int32 codes aligning a w-ld fileset's SNP labels to the ref categories.

    Disk-cached keyed by both filesets' content keys, so multi-trait runs
    reuse the alignment of a given ref/w pair. Returns None when either
    fileset is not a chr-split cached fileset; the caller then computes the
    alignment directly. Code -1 marks w SNPs absent from ref_cats, which inner
    joins drop exactly like absent object keys.
    '''
    rk = l2_content_key(ref_fh, num)
    wk = l2_content_key(w_fh, num)
    if rk is None or wk is None:
        return None
    h = hashlib.sha256()
    h.update(_b('l2pair-v1'))
    h.update(_b(rk))
    h.update(_b(wk))
    paths, _ = _l2_paths(ref_fh, num)
    base = os.path.dirname(os.path.abspath(paths[0]))
    path = os.path.join(base, _CACHE_DIRNAME, 'pair_' + h.hexdigest() + '.pkl')
    codes = _cache_read(path)
    if codes is not None:
        return codes
    codes = ref_cats.get_indexer(np.asarray(w_snp, dtype=object)).astype(np.int32)
    _cache_write(path, codes)
    return codes


def presubmit_l2_parse(fh, num):
    '''Submit an async parallel parse of a chr-split gzip ldscore fileset.'''
    if fh is None or num is None or ',' in fh:
        return
    try:
        paths, compression = _l2_paths(fh, num)
    except Exception:
        return
    if compression != 'gzip' or len(paths) < 3:
        return
    key = tuple(paths)
    if key in _PRESUBMIT:
        return
    try:
        cf = _cache_file(paths)
    except Exception:
        cf = None
    if cf is not None and os.path.exists(cf):
        _PRESUBMIT[key] = ('hit', cf)
    else:
        _PRESUBMIT[key] = ('miss', _parse_pool().map_async(
            lambda p: l2_parser(p, 'gzip'), paths), cf)


def _parallel_l2_parse(paths, compression):
    '''Parse chr-split .l2.ldscore files concurrently (order-preserving).'''
    if compression != 'gzip' or len(paths) < 3:
        return [l2_parser(p, compression) for p in paths]
    return _parse_pool().map(lambda p: l2_parser(p, 'gzip'), paths)


def read_csv(fh, **kwargs):
    # Fast path for gzipped file paths: decompress in one C-level call (using the
    # prefetch cache when populated) and parse from memory. Parser input bytes are
    # identical to pandas' compression='gzip' path.
    compression = kwargs.pop('compression', None)
    if compression == 'gzip' and isinstance(fh, str):
        if fh in _PREFETCH:
            data = _prefetch_get(fh)
        else:
            data = _read_gunzip(fh)
        return pd.read_csv(BytesIO(data), delim_whitespace=True, na_values='.', **kwargs)
    if compression is not None:
        kwargs['compression'] = compression
    return pd.read_csv(fh, delim_whitespace=True, na_values='.', **kwargs)


def sub_chr(s, chrom):
    '''Substitute chr for @, else append chr to the end of str.'''
    if '@' not in s:
        s += '@'

    return s.replace('@', str(chrom))


def get_present_chrs(fh, num):
    '''Checks which chromosomes exist, assuming that the file base will be appended by a dot in any suffix.'''
    chrs = []
    for chrom in range(1,num):
        if glob.glob(sub_chr(fh, chrom) + '.*'):
            chrs.append(chrom)
    return chrs


def which_compression(fh):
    '''Given a file prefix, figure out what sort of compression to use.'''
    if os.access(fh + '.bz2', 4):
        suffix = '.bz2'
        compression = 'bz2'
    elif os.access(fh + '.gz', 4):
        suffix = '.gz'
        compression = 'gzip'
    elif os.access(fh, 4):
        suffix = ''
        compression = None
    else:
        raise IOError('Could not open {F}[./gz/bz2]'.format(F=fh))

    return suffix, compression


def get_compression(fh):
    '''Which sort of compression should we use with read_csv?'''
    if fh.endswith('gz'):
        compression = 'gzip'
    elif fh.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None

    return compression


def read_cts(fh, match_snps):
    '''Reads files for --cts-bin.'''
    compression = get_compression(fh)
    cts = read_csv(fh, compression=compression, header=None, names=['SNP', 'ANNOT'])
    if not series_eq(cts.SNP, match_snps):
        raise ValueError('--cts-bin and the .bim file must have identical SNP columns.')

    return cts.ANNOT.values


def sumstats(fh, alleles=False, dropna=True):
    '''Parses .sumstats files. See docs/file_formats_sumstats.txt.'''
    dtype_dict = {'SNP': str,   'Z': float, 'N': float, 'A1': str, 'A2': str}
    compression = get_compression(fh)
    usecols = ['SNP', 'Z', 'N']
    if alleles:
        usecols += ['A1', 'A2']

    try:
        x = read_csv(fh, usecols=usecols, dtype=dtype_dict, compression=compression)
    except (AttributeError, ValueError) as e:
        raise ValueError('Improperly formatted sumstats file: ' + str(e.args))

    if dropna:
        x = x.dropna(how='any')

    return x


def ldscore_fromlist(flist, num=None):
    '''Sideways concatenation of a list of LD Score files.'''
    ldscore_array = []
    for i, fh in enumerate(flist):
        y = ldscore(fh, num)
        if i > 0:
            if not series_eq(y.SNP, ldscore_array[0].SNP):
                raise ValueError('LD Scores for concatenation must have identical SNP columns.')
            else:  # keep SNP column from only the first file
                y = y.drop(['SNP'], axis=1)

        new_col_dict = {c: c + '_' + str(i) for c in y.columns if c != 'SNP'}
        y.rename(columns=new_col_dict, inplace=True)
        ldscore_array.append(y)

    return pd.concat(ldscore_array, axis=1)


def l2_parser(fh, compression):
    '''Parse LD Score files'''
    x = read_csv(fh, header=0, compression=compression)
    if 'MAF' in x.columns and 'CM' in x.columns:  # for backwards compatibility w/ v<1.0.0
        x = x.drop(['MAF', 'CM'], axis=1)
    return x


def annot_parser(fh, compression, frqfile_full=None, compression_frq=None):
    '''Parse annot files'''
    df_annot = read_csv(fh, header=0, compression=compression).drop(['SNP','CHR', 'BP', 'CM'], axis=1, errors='ignore').astype(float)
    if frqfile_full is not None:
        df_frq = frq_parser(frqfile_full, compression_frq)
        df_annot = df_annot[(.95 > df_frq.FRQ) & (df_frq.FRQ > 0.05)]
    return df_annot


def frq_parser(fh, compression):
    '''Parse frequency files.'''
    df = read_csv(fh, header=0, compression=compression)
    if 'MAF' in df.columns:
        df.rename(columns={'MAF': 'FRQ'}, inplace=True)
    return df[['SNP', 'FRQ']]


def ldscore(fh, num=None):
    '''Parse .l2.ldscore files, split across num chromosomes. See docs/file_formats_ld.txt.'''
    suffix = '.l2.ldscore'
    if num is not None:  # num files, e.g., one per chromosome
        paths, compression = _l2_paths(fh, num)
        entry = _PRESUBMIT.pop(tuple(paths), None)
        cache_to = None
        chr_ld = None
        if entry is not None and entry[0] == 'hit':
            x = _cache_read(entry[1])
            if x is not None:
                return x  # exact pickled copy of a previous return value
            cache_to = entry[1]  # unreadable cache: parse and rewrite
        elif entry is not None:  # ('miss', AsyncResult, cache_path)
            chr_ld = entry[1].get()
            cache_to = entry[2]
        if chr_ld is None:
            chr_ld = _parallel_l2_parse(paths, compression)
        x = pd.concat(chr_ld)  # automatically sorted by chromosome
        x = x.sort_values(by=['CHR', 'BP']) # SEs will be wrong unless sorted
        x = x.drop(['CHR', 'BP'], axis=1).drop_duplicates(subset='SNP')
        # Store/return SNP as a Categorical: cached frames then let downstream
        # merges join on integer codes instead of re-factorizing ~1M object
        # strings per merge side. Labels, row order, and all other columns are
        # unchanged (verified byte-identical merge outputs).
        x['SNP'] = pd.Categorical(x['SNP'].values)
        if cache_to is not None:
            _cache_write(cache_to, x)
        return x
    else:  # just one file
        s, compression = which_compression(fh + suffix)
        x = l2_parser(fh + suffix + s, compression)

    x = x.sort_values(by=['CHR', 'BP']) # SEs will be wrong unless sorted
    x = x.drop(['CHR', 'BP'], axis=1).drop_duplicates(subset='SNP')
    return x


def M(fh, num=None, N=2, common=False):
    '''Parses .l{N}.M files, split across num chromosomes. See docs/file_formats_ld.txt.'''
    parsefunc = lambda y: [float(z) for z in open(y, 'r').readline().split()]
    suffix = '.l' + str(N) + '.M'
    if common:
        suffix += '_5_50'

    if num is not None:
        x = np.sum([parsefunc(sub_chr(fh, i) + suffix) for i in get_present_chrs(fh, num+1)], axis=0)
    else:
        x = parsefunc(fh + suffix)

    return np.array(x).reshape((1, len(x)))


def M_fromlist(flist, num=None, N=2, common=False):
    '''Read a list of .M* files and concatenate sideways.'''
    return np.hstack([M(fh, num, N, common) for fh in flist])


def annot(fh_list, num=None, frqfile=None):
    '''
    Parses .annot files and returns an overlap matrix. See docs/file_formats_ld.txt.
    If num is not None, parses .annot files split across [num] chromosomes (e.g., the
    output of parallelizing ldsc.py --l2 across chromosomes).

    '''
    annot_suffix = ['.annot' for fh in fh_list]
    annot_compression = []
    if num is not None:  # 22 files, one for each chromosome
        chrs = get_present_chrs(fh, num+1)
        for i, fh in enumerate(fh_list):
            first_fh = sub_chr(fh, chrs[0]) + annot_suffix[i]
            annot_s, annot_comp_single = which_compression(first_fh)
            annot_suffix[i] += annot_s
            annot_compression.append(annot_comp_single)

        if frqfile is not None:
            frq_suffix = '.frq'
            first_frqfile = sub_chr(frqfile, 1) + frq_suffix
            frq_s, frq_compression = which_compression(first_frqfile)
            frq_suffix += frq_s

        y = []
        M_tot = 0
        for chrom in chrs:
            if frqfile is not None:
                df_annot_chr_list = [annot_parser(sub_chr(fh, chrom) + annot_suffix[i], annot_compression[i],
                                                  sub_chr(frqfile, chrom) + frq_suffix, frq_compression)
                                     for i, fh in enumerate(fh_list)]
            else:
                df_annot_chr_list = [annot_parser(sub_chr(fh, chrom) + annot_suffix[i], annot_compression[i])
                                     for i, fh in enumerate(fh_list)]

            annot_matrix_chr_list = [np.matrix(df_annot_chr) for df_annot_chr in df_annot_chr_list]
            annot_matrix_chr = np.hstack(annot_matrix_chr_list)
            y.append(np.dot(annot_matrix_chr.T, annot_matrix_chr))
            M_tot += len(df_annot_chr_list[0])

        x = sum(y)
    else:  # just one file
        for i, fh in enumerate(fh_list):
            annot_s, annot_comp_single = which_compression(fh + annot_suffix[i])
            annot_suffix[i] += annot_s
            annot_compression.append(annot_comp_single)

        if frqfile is not None:
            frq_suffix = '.frq'
            frq_s, frq_compression = which_compression(frqfile + frq_suffix)
            frq_suffix += frq_s

            df_annot_list = [annot_parser(fh + annot_suffix[i], annot_compression[i],
                                          frqfile + frq_suffix, frq_compression) for i, fh in enumerate(fh_list)]

        else:
            df_annot_list = [annot_parser(fh + annot_suffix[i], annot_compression[i])
                             for i, fh in enumerate(fh_list)]

        annot_matrix_list = [np.matrix(y) for y in df_annot_list]
        annot_matrix = np.hstack(annot_matrix_list)
        x = np.dot(annot_matrix.T, annot_matrix)
        M_tot = len(df_annot_list[0])

    return x, M_tot


def __ID_List_Factory__(colnames, keepcol, fname_end, header=None, usecols=None):

    class IDContainer(object):

        def __init__(self, fname):
            self.__usecols__ = usecols
            self.__colnames__ = colnames
            self.__keepcol__ = keepcol
            self.__fname_end__ = fname_end
            self.__header__ = header
            self.__read__(fname)
            self.n = len(self.df)

        def __read__(self, fname):
            end = self.__fname_end__
            if end and not fname.endswith(end):
                raise ValueError('{f} filename must end in {f}'.format(f=end))

            comp = get_compression(fname)
            self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__,
                                  delim_whitespace=True, compression=comp)

            if self.__colnames__:
                self.df.columns = self.__colnames__

            if self.__keepcol__ is not None:
                self.IDList = self.df.iloc[:, [self.__keepcol__]].astype('object')

        def loj(self, externalDf):
            '''Returns indices of those elements of self.IDList that appear in exernalDf.'''
            r = externalDf.columns[0]
            l = self.IDList.columns[0]
            merge_df = externalDf.iloc[:, [0]]
            merge_df['keep'] = True
            z = pd.merge(self.IDList, merge_df, how='left', left_on=l, right_on=r,
                         sort=False)
            ii = z['keep'] == True
            return np.nonzero(ii)[0]

    return IDContainer


PlinkBIMFile = __ID_List_Factory__(['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], 1, '.bim', usecols=[0, 1, 2, 3, 4, 5])
PlinkFAMFile = __ID_List_Factory__(['IID'], 0, '.fam', usecols=[1])
FilterFile = __ID_List_Factory__(['ID'], 0, None, usecols=[0])
AnnotFile = __ID_List_Factory__(None, 2, None, header=0, usecols=None)
ThinAnnotFile = __ID_List_Factory__(None, None, None, header=0, usecols=None)
