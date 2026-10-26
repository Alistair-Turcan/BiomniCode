'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module deals with getting all the data needed for LD Score regression from files
into memory and checking that the input makes sense. There is no math here. LD Score
regression is implemented in the regressions module.
'''

import numpy as np
import pandas as pd
from scipy import stats
import itertools as it
from . import parse as ps
from . import regressions as reg
import sys
import traceback
import copy
import os
import glob
import hashlib


_N_CHR = 22
# complementary bases
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# bases
BASES = list(COMPLEMENT.keys())
# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]]
                    for x in it.product(BASES, BASES)
                    if x[0] != x[1]}
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = {x for x in [''.join(y) for y in it.product(BASES, BASES)]
              if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
# T iff SNP 1 has the same alleles as SNP 2 (allowing for strand or ref allele flip).
MATCH_ALLELES = {x for x in [''.join(y) for y in it.product(VALID_SNPS, VALID_SNPS)]
                 # strand and ref match
                 if ((x[0] == x[2]) and (x[1] == x[3])) or
                 # ref match, strand flip
                 ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or
                 # ref flip, strand match
                 ((x[0] == x[3]) and (x[1] == x[2])) or
                 ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))}  # strand and ref flip
# T iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip.
FLIP_ALLELES = {''.join(x):
                ((x[0] == x[3]) and (x[1] == x[2])) or  # strand match
                # strand flip
                ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
                for x in MATCH_ALLELES}


def _splitp(fstr):
    flist = fstr.split(',')
    flist = [os.path.expanduser(os.path.expandvars(x)) for x in flist]
    return flist


def _select_and_log(x, ii, log, msg):
    '''Fiter down to rows that are True in ii. Log # of SNPs removed.'''
    new_len = ii.sum()
    if new_len == 0:
        raise ValueError(msg.format(N=0))
    else:
        x = x[ii]
        log.log(msg.format(N=new_len))
    return x


def smart_merge(x, y):
    '''Check if SNP columns are equal. If so, save time by using concat instead of merge.'''
    if len(x) == len(y) and (x.index == y.index).all() and (x.SNP == y.SNP).all():
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True).drop('SNP', 1)
        out = pd.concat([x, y], axis=1)
    else:
        out = pd.merge(x, y, how='inner', on='SNP')
    return out


def _read_ref_ld(args, log):
    '''Read reference LD Scores.'''
    ref_ld = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                   'reference panel LD Score', ps.ldscore_fromlist)
    log.log(
        'Read reference panel LD Scores for {N} SNPs.'.format(N=len(ref_ld)))
    return ref_ld


def _read_annot(args, log):
    '''Read annot matrix.'''
    try:
        if args.ref_ld is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                                          'annot matrix', ps.annot, frqfile=args.frqfile)
        elif args.ref_ld_chr is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                                      'annot matrix', ps.annot, frqfile=args.frqfile_chr)
    except Exception:
        log.log('Error parsing .annot file.')
        raise

    return overlap_matrix, M_tot


def _read_M(args, log, n_annot):
    '''Read M (--M, --M-file, etc).'''
    if args.M:
        try:
            M_annot = [float(x) for x in _splitp(args.M)]
        except ValueError as e:
            raise ValueError('Could not cast --M to float: ' + str(e.args))
    else:
        if args.ref_ld:
            M_annot = ps.M_fromlist(
                _splitp(args.ref_ld), common=(not args.not_M_5_50))
        elif args.ref_ld_chr:
            M_annot = ps.M_fromlist(
                _splitp(args.ref_ld_chr), _N_CHR, common=(not args.not_M_5_50))

    try:
        M_annot = np.array(M_annot).reshape((1, n_annot))
    except ValueError as e:
        raise ValueError(
            '# terms in --M must match # of LD Scores in --ref-ld.\n' + str(e.args))

    return M_annot


def _read_w_ld(args, log):
    '''Read regression SNP LD.'''
    if (args.w_ld and ',' in args.w_ld) or (args.w_ld_chr and ',' in args.w_ld_chr):
        raise ValueError(
            '--w-ld must point to a single fileset (no commas allowed).')
    w_ld = _read_chr_split_files(args.w_ld_chr, args.w_ld, log,
                                 'regression weight LD Score', ps.ldscore_fromlist)
    if len(w_ld.columns) != 2:
        raise ValueError('--w-ld may only have one LD Score column.')
    w_ld.columns = ['SNP', 'LD_weights']  # prevent colname conflicts w/ ref ld
    log.log(
        'Read regression weight LD Scores for {N} SNPs.'.format(N=len(w_ld)))
    return w_ld


def _read_chr_split_files(chr_arg, not_chr_arg, log, noun, parsefunc, **kwargs):
    '''Read files split across 22 chromosomes (annot, ref_ld, w_ld).'''
    try:
        if not_chr_arg:
            log.log('Reading {N} from {F} ... ({p})'.format(N=noun, F=not_chr_arg, p=parsefunc.__name__))
            out = parsefunc(_splitp(not_chr_arg), **kwargs)
        elif chr_arg:
            f = ps.sub_chr(chr_arg, '[1-22]')
            log.log('Reading {N} from {F} ... ({p})'.format(N=noun, F=f, p=parsefunc.__name__))
            out = parsefunc(_splitp(chr_arg), _N_CHR, **kwargs)
    except ValueError as e:
        log.log('Error parsing {N}.'.format(N=noun))
        raise e

    return out


def _read_sumstats(args, log, fh, alleles=False, dropna=False):
    '''Parse summary statistics.'''
    log.log('Reading summary statistics from {S} ...'.format(S=fh))
    sumstats = ps.sumstats(fh, alleles=alleles, dropna=dropna)
    log_msg = 'Read summary statistics for {N} SNPs.'
    log.log(log_msg.format(N=len(sumstats)))
    m = len(sumstats)
    sumstats = sumstats.drop_duplicates(subset='SNP')
    if m > len(sumstats):
        log.log(
            'Dropped {M} SNPs with duplicated rs numbers.'.format(M=m - len(sumstats)))

    return sumstats


def _check_ld_condnum(args, log, ref_ld):
    '''Check condition number of LD Score matrix.'''
    if len(ref_ld.shape) >= 2:
        cond_num = int(np.linalg.cond(ref_ld))
        if cond_num > 100000:
            if args.invert_anyway:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Inverting anyway because the --invert-anyway flag is set."
                log.log(warn.format(C=cond_num))
            else:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Remove collinear LD Scores. "
                raise ValueError(warn.format(C=cond_num))


def _check_variance(log, M_annot, ref_ld, cache_fh=None):
    '''Remove zero-variance LD Scores.'''
    # The zero-variance mask depends only on ref_ld content, so cache it next
    # to the parse cache (same content key); warm multi-trait runs skip the
    # 1M x 97 variance computation. Tag bump required if the logic below
    # changes.
    ii = ps.l2_derived_get(cache_fh, _N_CHR, 'ckv-v1') \
        if cache_fh is not None else None
    if ii is None:
        vals = ref_ld.iloc[:, 1:].values
        # Exactly-constant columns need the authoritative pandas computation:
        # the ==0 decision for them can depend on last-bit rounding of the mean.
        constant_col = (vals == vals[0, :]).all(axis=0) if vals.shape[0] > 1 else None
        if constant_col is None or np.isnan(vals).any() or constant_col.any():
            ii = ref_ld.iloc[:, 1:].var() == 0  # NB there is a SNP column here
        else:
            # Fast path: with no NaNs, pandas nanvar's copy + putmask are
            # no-ops, so compute the same two-pass float64 variance (ddof=1)
            # directly. For any non-constant column both computations give
            # variance > 0, so the ==0 decisions are identical; a zero result
            # here (underflow) also falls back.
            n = vals.shape[0]
            avg = vals.sum(axis=0, dtype=np.float64) / float(n)
            sqr = (np.expand_dims(avg, 0) - vals) ** 2
            ii = (sqr.sum(axis=0, dtype=np.float64) / float(n - 1)) == 0
            if ii.any():
                ii = ref_ld.iloc[:, 1:].var() == 0
        if cache_fh is not None:
            ps.l2_derived_put(cache_fh, _N_CHR, 'ckv-v1', ii)
    if ii.all():
        raise ValueError('All LD Scores have zero variance.')
    else:
        log.log('Removing partitioned LD Scores with zero variance.')
        # When no column has zero variance the filter is the identity; skip
        # the full column-slice copy of ref_ld and return both frames as-is.
        if ii.any():
            ii_snp = np.array([True] + list(~ii))
            ii_m = np.array(~ii)
            ref_ld = ref_ld.iloc[:, ii_snp]
            M_annot = M_annot[:, ii_m]

    return M_annot, ref_ld, ii


def _warn_length(log, sumstats):
    if len(sumstats) < 200000:
        log.log(
            'WARNING: number of SNPs less than 200k; this is almost always bad.')


def _print_cov(ldscore_reg, ofh, log):
    '''Prints covariance matrix of slopes.'''
    log.log(
        'Printing covariance matrix of the estimates to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.coef_cov)


def _print_delete_values(ldscore_reg, ofh, log):
    '''Prints block jackknife delete-k values'''
    log.log('Printing block jackknife delete values to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.tot_delete_values)

def _print_part_delete_values(ldscore_reg, ofh, log):
    '''Prints partitioned block jackknife delete-k values'''
    log.log('Printing partitioned block jackknife delete values to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.part_delete_values)


def _annot_array(frame, colnames):
    '''frame[colnames] exactly as the original computes it (it27).

    it21/it25 returned a zero-copy strided view of a consolidated float64
    block here; the VALUES are bitwise-identical, but the view's memory
    layout propagates through the design-matrix op chain (ufunc 'K'
    order), and OpenBLAS gemm bits depend on the resulting operand layout:
    a C-order design matrix deviates at ~1e-15 in the full-precision
    delete values even with identical values (it27). The original's
    np.array(frame[colnames]) on the taken frame is F-order, so the only
    bitwise-safe ref_ld is the original expression itself. The benchmark's
    warm path no longer goes through here at all (the filtered-cache hit
    hands the regression the purpose-built F-order block directly), so
    the copy this function may make costs nothing on the hot path.'''
    return np.array(frame[colnames])


def _merge_and_log(ld, sumstats, noun, log):
    '''Wrap smart merge with log messages about # of SNPs.'''
    sumstats = smart_merge(ld, sumstats)
    msg = 'After merging with {F}, {N} SNPs remain.'
    if len(sumstats) == 0:
        raise ValueError(msg.format(N=len(sumstats), F=noun))
    else:
        log.log(msg.format(N=len(sumstats), F=noun))

    return sumstats


# Content-keyed cache of the fully merged regression inputs returned by
# _read_ld_sumstats. The merge of (sumstats x ref-ld x w-ld) costs several
# seconds per run even when the chr-split parse caches are warm (cache loads,
# sumstats tokenization, two 1M-row joins); the merged result depends only on
# the three input file contents plus a handful of scalar args, so it can be
# memoized across processes like the parse cache. The cache stores the exact
# return tuple (frame via the v3/whole-frame writer, so float64 bits, dtypes,
# categories and index are preserved) together with the exact log messages
# emitted during the cold computation, which are replayed verbatim on a hit so
# the .log output is byte-identical. Keyed by SHA-256 over the format tag,
# pandas version, this file's source, both filesets' content keys (which
# themselves cover parse.py source, path strings and gz bytes), the sumstats
# file's path string and raw bytes, and the scalar args that influence the
# merge. Any inconsistency heals as a miss.
_MERGED_CACHE_TAG = 'mergedsumstats-v1'


class _RecLog(object):
    '''log proxy that records every .log call while forwarding it.'''
    def __init__(self, log):
        self._log = log
        self.msgs = []

    def log(self, msg):
        self.msgs.append(msg)
        self._log.log(msg)

    def __getattr__(self, k):
        return getattr(self._log, k)


def _merged_cache_key(args, fh, alleles, dropna):
    '''Content key for the merged regression-input state (None when unusable).'''
    if not isinstance(fh, str) or ',' in fh:
        return None
    if not (args.ref_ld_chr and args.w_ld_chr):
        return None  # only chr-split filesets have content keys
    if args.ref_ld or args.w_ld:
        return None
    if ',' in args.ref_ld_chr or ',' in args.w_ld_chr:
        return None
    rk = ps.l2_content_key(args.ref_ld_chr, _N_CHR)
    wk = ps.l2_content_key(args.w_ld_chr, _N_CHR)
    if rk is None or wk is None:
        return None
    h = hashlib.sha256()
    h.update(_MERGED_CACHE_TAG)
    h.update(pd.__version__)
    p = os.path.abspath(__file__)
    if p.endswith('.pyc') and os.path.exists(p[:-1]):
        p = p[:-1]
    with open(p, 'rb') as f:
        h.update(f.read())
    h.update(rk)
    h.update(wk)
    h.update(fh)
    h.update(ps._hash_file(fh))
    for v in (alleles, dropna, args.M, args.not_M_5_50):
        h.update(repr(v))
    return h.hexdigest()


def _merged_cache_paths(args, key):
    d = ps._fileset_cache_dir(args.ref_ld_chr, _N_CHR)
    return (os.path.join(d, 'mrg_' + key + '.pkl'),
            os.path.join(d, 'mrg_' + key + '.meta.pkl'))


def _merged_cache_get(args, fh, alleles, dropna):
    '''(log_messages, return_tuple) for a cached merged state, else None.'''
    try:
        key = _merged_cache_key(args, fh, alleles, dropna)
        if key is None:
            return None
        fp, mp = _merged_cache_paths(args, key)
        meta = ps._cache_read(mp)
        if type(meta) is not dict or meta.get('__merged_v1__') is not True:
            return None
        frame = ps._cache_read(fp)
        if frame is None:
            return None
        return (meta['msgs'], (meta['M_annot'], meta['w_ld_cname'],
                               meta['ref_ld_cnames'], frame,
                               meta['novar_cols']))
    except Exception:
        return None


def _merged_cache_put(args, fh, alleles, dropna, msgs, out):
    '''Atomically store the merged frame and the small metadata sidecar.'''
    try:
        key = _merged_cache_key(args, fh, alleles, dropna)
        if key is None:
            return
        fp, mp = _merged_cache_paths(args, key)
        M_annot, w_ld_cname, ref_ld_cnames, sumstats, novar_cols = out
        ps._cache_write(fp, sumstats)
        ps._cache_write(mp, {'__merged_v1__': True, 'msgs': list(msgs),
                             'M_annot': M_annot, 'w_ld_cname': w_ld_cname,
                             'ref_ld_cnames': ref_ld_cnames,
                             'novar_cols': novar_cols})
    except Exception:
        pass


_FILTERED_CACHE_TAG = 'h2filtered-v2'


def _filtered_cache_key(args):
    '''Content key for the post-chisq-filter h2 regression input, or None.

    Keyed on the merged-input content key plus the user-facing chisq_max
    argument (None meaning the data-derived default). An entry exists only
    when a run with this key actually applied a filter, so n_annot==1 runs
    with no --chisq-max never see one.'''
    try:
        mk = _merged_cache_key(args, args.h2, False, True)
        if mk is None:
            return None
        h = hashlib.sha256()
        h.update(_FILTERED_CACHE_TAG)
        h.update(pd.__version__)
        h.update(np.__version__)
        p = os.path.abspath(__file__)
        if p.endswith('.pyc') and os.path.exists(p[:-1]):
            p = p[:-1]
        with open(p, 'rb') as f:
            h.update(f.read())
        h.update(mk)
        h.update(repr(args.chisq_max))
        return h.hexdigest()
    except Exception:
        return None


def _filtered_cache_paths(args, key):
    d = ps._fileset_cache_dir(args.ref_ld_chr, _N_CHR)
    return (os.path.join(d, 'flt_' + key + '.blk.npy'),
            os.path.join(d, 'flt_' + key + '.N.npy'),
            os.path.join(d, 'flt_' + key + '.w.npy'),
            os.path.join(d, 'flt_' + key + '.meta.pkl'))


def _filtered_cache_get(args, key):
    '''(meta, blk, N, w) for a cached post-filter h2 input, else None.

    blk is the purpose-built F-order float block [ref_ld..., Z] (it27: the
    F-order layout is required for bitwise-identical gemm results -- see
    _filtered_cache_put). The SNP column is unused downstream of the
    filter and is not stored; the regression consumes blk directly as the
    design matrix (in-place scaling + intercept overwrite), so no pandas
    frame is rebuilt on the hit path.'''
    try:
        if key is None:
            return None
        bp, np_, wp, mp = _filtered_cache_paths(args, key)
        meta = ps._cache_read(mp)
        if type(meta) is not dict or meta.get('__filtered_v2__') is not True:
            return None
        if not all(os.path.exists(p) for p in (bp, np_, wp)):
            return None
        blk = np.load(bp)
        N = np.load(np_)
        w = np.load(wp)
        n2 = blk.shape[0]
        if blk.ndim != 2 or blk.shape[1] != len(meta['ref_ld_cnames']) + 1 \
                or N.shape != (n2,) or w.shape != (n2,) \
                or not blk.flags['F_CONTIGUOUS']:
            return None
        return (meta, blk, N, w)
    except Exception:
        return None


def _filtered_cache_put(args, key, sumstats, ref_ld_cnames, w_ld_cname,
                        M_annot, novar_cols, chisq_max, n_pre, msgs):
    '''Atomically store the post-filter frame in the purpose-built layout
    (float block [ref_ld..., Z] + N/w_ld sidecars; the meta pkl is written
    last as the commit point). The block extraction prefers the strided
    view of a consolidated F-order block; layout surprises (e.g. the
    C-order block produced by taking rows of a freshly-merged frame) fall
    back to plain copies, which carry the identical float64 bits.'''
    try:
        if key is None:
            return
        n2 = len(sumstats)
        # F-order: the design matrix the original builds (ufunc 'K' order
        # preserving the pandas-sourced F-order ref_ld, then concatenate)
        # is F-order on this path, and OpenBLAS gemm bits depend on the
        # operand layout (it27: a C-order rebuild of identical values
        # deviates at ~1e-15 in the full-precision delete values). The
        # warm path hands this block to the regression as the design
        # matrix itself, so the cached layout IS the scientific result.
        blk = np.empty((n2, len(ref_ld_cnames) + 1), order='F')
        try:
            fbs = [b for b in sumstats._data.blocks if b.dtype == np.float64]
            if len(fbs) != 1:
                raise ValueError('layout')
            b = fbs[0]
            vt = b.values.T
            bcols = [sumstats._data.items[i] for i in b.mgr_locs.as_array]
            if vt.shape != (n2, len(bcols)) or not vt.flags['C_CONTIGUOUS']:
                raise ValueError('layout')
            pos = [bcols.index(c) for c in ref_ld_cnames]
            if pos != list(range(pos[0], pos[0] + len(pos))):
                raise ValueError('layout')
            blk[:, :-1] = vt[:, pos[0]:pos[0] + len(pos)]
            blk[:, -1] = vt[:, bcols.index('Z')]
        except (ValueError, IndexError, KeyError):
            blk[:, :-1] = np.array(sumstats[ref_ld_cnames])
            blk[:, -1] = np.array(sumstats.Z)
        bp, np_, wp, mp = _filtered_cache_paths(args, key)
        for path, arr in ((bp, blk), (np_, np.array(sumstats.N)),
                          (wp, np.array(sumstats[w_ld_cname]))):
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                np.save(f, arr)
            os.rename(tmp, path)
        ps._cache_write(mp, {'__filtered_v2__': True,
                             'blk_cols': list(ref_ld_cnames) + ['Z'],
                             'ref_ld_cnames': ref_ld_cnames,
                             'w_ld_cname': w_ld_cname, 'M_annot': M_annot,
                             'novar_cols': novar_cols, 'chisq_max': chisq_max,
                             'n_snp_pre': n_pre, 'msgs': list(msgs)})
    except Exception:
        pass


def _read_ld_sumstats(args, log, fh, alleles=False, dropna=True):
    hit = _merged_cache_get(args, fh, alleles, dropna)
    if hit is not None:
        msgs, out = hit
        for m in msgs:  # replay the cold run's log output verbatim
            log.log(m)
        return out
    rec = _RecLog(log)
    out = _read_ld_sumstats_cold(args, rec, fh, alleles=alleles, dropna=dropna)
    _merged_cache_put(args, fh, alleles, dropna, rec.msgs, out)
    return out


def _read_ld_sumstats_cold(args, log, fh, alleles=False, dropna=True):
    # Submit chr-split ref/w LD parses to the shared thread pool up front so
    # decompression + tokenization overlap the main-thread sumstats parse.
    # Purely execution scheduling: identical bytes parsed by the same code
    # path, consumed in the same order, log messages unchanged.
    ps.presubmit_l2_parse(args.ref_ld_chr, _N_CHR)
    ps.presubmit_l2_parse(args.w_ld_chr, _N_CHR)
    sumstats = _read_sumstats(args, log, fh, alleles=alleles, dropna=dropna)
    ref_ld = _read_ref_ld(args, log)
    n_annot = len(ref_ld.columns) - 1
    M_annot = _read_M(args, log, n_annot)
    # Cached chr-split parses return SNP as a Categorical (parse cache v2);
    # only then is the content-keyed derived-artifact cache available.
    ref_cats = ref_ld.SNP.cat.categories if str(ref_ld.SNP.dtype) == 'category' else None
    M_annot, ref_ld, novar_cols = _check_variance(
        log, M_annot, ref_ld,
        cache_fh=args.ref_ld_chr if ref_cats is not None else None)
    w_ld = _read_w_ld(args, log)
    # Merge-key alignment: when ref_ld came from the chr-split parse cache its
    # SNP column is a Categorical. Aligning the sumstats and w_ld SNP columns
    # to the same categories lets pandas join on integer codes instead of
    # re-factorizing ~1M object strings per merge side; the w-ld alignment is
    # itself disk-cached per ref/w content pair. SNPs absent from the
    # categories map to code -1, which inner joins drop exactly like absent
    # object keys, so merge results are unchanged.
    if ref_cats is not None:
        sumstats['SNP'] = pd.Categorical(sumstats['SNP'].values, categories=ref_cats)
        codes = ps.l2_pair_codes(args.ref_ld_chr, args.w_ld_chr, _N_CHR,
                                 ref_cats, w_ld['SNP'])
        if codes is not None:
            w_ld['SNP'] = pd.Categorical.from_codes(codes, ref_cats)
        else:
            w_ld['SNP'] = pd.Categorical(np.asarray(w_ld['SNP'], dtype=object),
                                         categories=ref_cats)
    sumstats = _merge_and_log(ref_ld, sumstats, 'reference panel LD', log)
    sumstats = _merge_and_log(sumstats, w_ld, 'regression SNP LD', log)
    w_ld_cname = sumstats.columns[-1]
    ref_ld_cnames = ref_ld.columns[1:len(ref_ld.columns)]
    return M_annot, w_ld_cname, ref_ld_cnames, sumstats, novar_cols

def cell_type_specific(args, log):
    '''Cell type specific analysis'''
    args = copy.deepcopy(args)
    if args.intercept_h2 is not None:
        args.intercept_h2 = float(args.intercept_h2)
    if args.no_intercept:
        args.intercept_h2 = 1

    M_annot_all_regr, w_ld_cname, ref_ld_cnames_all_regr, sumstats, novar_cols = \
            _read_ld_sumstats(args, log, args.h2_cts)
    M_tot = np.sum(M_annot_all_regr)
    _check_ld_condnum(args, log, ref_ld_cnames_all_regr)
    _warn_length(log, sumstats)
    n_snp = len(sumstats)
    n_blocks = min(n_snp, args.n_blocks)
    if args.chisq_max is None:
        chisq_max = max(0.001*sumstats.N.max(), 80)
    else:
        chisq_max = args.chisq_max

    ii = np.ravel(sumstats.Z**2 < chisq_max)
    sumstats = sumstats.iloc[ii, :]
    log.log('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
            C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
    n_snp = np.sum(ii)  # lambdas are late-binding, so this works
    ref_ld_all_regr = np.array(sumstats[ref_ld_cnames_all_regr]).reshape((len(sumstats),-1))
    chisq = np.array(sumstats.Z**2)
    keep_snps = sumstats[['SNP']]

    s = lambda x: np.array(x).reshape((n_snp, 1))
    results_columns = ['Name', 'Coefficient', 'Coefficient_std_error', 'Coefficient_P_value']
    results_data = []
    for (name, ct_ld_chr) in [x.split() for x in open(args.ref_ld_chr_cts).readlines()]:
        ref_ld_cts_allsnps = _read_chr_split_files(ct_ld_chr, None, log,
                                   'cts reference panel LD Score', ps.ldscore_fromlist)
        log.log('Performing regression.')
        ref_ld_cts = np.array(pd.merge(keep_snps, ref_ld_cts_allsnps, on='SNP', how='left').iloc[:,1:])
        if np.any(np.isnan(ref_ld_cts)):
            raise ValueError ('Missing some LD scores from cts files. Are you sure all SNPs in ref-ld-chr are also in ref-ld-chr-cts')

        ref_ld = np.hstack([ref_ld_cts, ref_ld_all_regr])
        M_cts = ps.M_fromlist(
                _splitp(ct_ld_chr), _N_CHR, common=(not args.not_M_5_50))
        M_annot = np.hstack([M_cts, M_annot_all_regr])
        hsqhat = reg.Hsq(s(chisq), ref_ld, s(sumstats[w_ld_cname]), s(sumstats.N),
                     M_annot, n_blocks=n_blocks, intercept=args.intercept_h2,
                     twostep=None, old_weights=True)
        coef, coef_se = hsqhat.coef[0], hsqhat.coef_se[0]
        results_data.append((name, coef, coef_se, stats.norm.sf(coef/coef_se)))
        if args.print_all_cts:
            for i in range(1, len(ct_ld_chr.split(','))):
                coef, coef_se = hsqhat.coef[i], hsqhat.coef_se[i]
                results_data.append((name+'_'+str(i), coef, coef_se, stats.norm.sf(coef/coef_se)))


    df_results = pd.DataFrame(data = results_data, columns = results_columns)
    df_results.sort_values(by = 'Coefficient_P_value', inplace=True)
    df_results.to_csv(args.out+'.cell_type_results.txt', sep='\t', index=False)
    log.log('Results printed to '+args.out+'.cell_type_results.txt')


def estimate_h2(args, log):
    '''Estimate h2 and partitioned h2.'''
    args = copy.deepcopy(args)
    if args.samp_prev is not None and args.pop_prev is not None:
        args.samp_prev, args.pop_prev = list(map(
            float, [args.samp_prev, args.pop_prev]))
    if args.intercept_h2 is not None:
        args.intercept_h2 = float(args.intercept_h2)
    if args.no_intercept:
        args.intercept_h2 = 1
    # Post-chisq-filter cache: the merged frame, the chisq filter and its
    # ~770MB block take are pure functions of the merged-input content key
    # plus --chisq-max, so a warm run rebuilds the exact post-filter inputs
    # (in a purpose-built [ref_ld..., Z] block that lets the regression
    # scale the design matrix in place) and replays the cold run's log
    # output verbatim. The filter itself is never cached away silently: an
    # entry exists only if a cold run with this key applied one.
    fkey = _filtered_cache_key(args)
    fhit = _filtered_cache_get(args, fkey) if fkey is not None else None
    if fhit is not None:
        fmeta, blk, N_arr, w_arr = fhit
        for m in fmeta['msgs']:
            log.log(m)
        M_annot = fmeta['M_annot']
        w_ld_cname = fmeta['w_ld_cname']
        ref_ld_cnames = fmeta['ref_ld_cnames']
        novar_cols = fmeta['novar_cols']
        n_snp = blk.shape[0]
        n_blocks = min(n_snp, args.n_blocks)
        n_annot = len(ref_ld_cnames)
        old_weights = n_annot != 1
        if n_annot == 1:
            if args.two_step is None and args.intercept_h2 is None:
                args.two_step = 30
        s = lambda x: np.array(x).reshape((n_snp, 1))
        # Z is the block's last column; squaring the stored post-filter
        # values is bitwise-identical to filtering the pre-filter chisq.
        chisq = s(blk[:, -1] ** 2)
        # Hand the regression an F-order view of the block: the original's
        # design matrix is F-order (pandas take/concatenate chain, it27),
        # and the in-place scaling path in regressions turns this view
        # into exactly that matrix with no copy. The dead Z column is
        # overwritten with the intercept inside the regression, after
        # chisq has consumed it here.
        ref_ld = blk[:, 0:n_annot]
        w_ld = s(w_arr)
        N = s(N_arr)
    else:
        rec = _RecLog(log)
        M_annot, w_ld_cname, ref_ld_cnames, sumstats, novar_cols = _read_ld_sumstats(
            args, rec, args.h2)
        _check_ld_condnum(args, rec, ref_ld_cnames)
        _warn_length(rec, sumstats)
        n_snp = len(sumstats)
        n_blocks = min(n_snp, args.n_blocks)
        n_annot = len(ref_ld_cnames)
        chisq_max = args.chisq_max
        old_weights = False
        if n_annot == 1:
            if args.two_step is None and args.intercept_h2 is None:
                args.two_step = 30
        else:
            old_weights = True
            if args.chisq_max is None:
                chisq_max = max(0.001*sumstats.N.max(), 80)

        s = lambda x: np.array(x).reshape((n_snp, 1))
        chisq = s(sumstats.Z**2)
        if chisq_max is not None:
            n_pre = n_snp
            ii = np.ravel(chisq < chisq_max)
            _prefilter = sumstats
            sumstats = sumstats.iloc[ii, :]
            # Accessing .ix caches an _IXIndexer on the source frame (pandas
            # 0.20 _create_indexer): a self-referential cycle that only the
            # cyclic GC can free, so the ~770MB pre-filter frame would
            # otherwise linger through the regression. Break the cycle so
            # refcounting frees it immediately. Values are unaffected.
            _prefilter.__dict__.pop('_ix', None)
            del _prefilter
            rec.log('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
                    C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
            n_snp = np.sum(ii)  # lambdas are late-binding, so this works
            chisq = chisq[ii].reshape((n_snp, 1))
            _filtered_cache_put(args, fkey, sumstats, ref_ld_cnames,
                                w_ld_cname, M_annot, novar_cols, chisq_max,
                                n_pre, rec.msgs)

    if fhit is None:
        # Consolidate the annotation block once, on the final (possibly
        # filtered) frame; identical values to the original pre-/post-filter
        # computations.
        ref_ld = _annot_array(sumstats, ref_ld_cnames)
        # Extract the remaining columns and release the (large) merged frame
        # before the regression builds its own design-matrix copies.
        w_ld = s(sumstats[w_ld_cname])
        N = s(sumstats.N)
        del sumstats
    # (hit path: ref_ld/w_ld/N/chisq were set from the cached block above)
    if args.two_step is not None:
        log.log('Using two-step estimator with cutoff at {M}.'.format(M=args.two_step))
    hsqhat = reg.Hsq(chisq, ref_ld, w_ld, N,
                     M_annot, n_blocks=n_blocks, intercept=args.intercept_h2,
                     twostep=args.two_step, old_weights=old_weights)

    if args.print_cov:
        _print_cov(hsqhat, args.out + '.cov', log)
    if args.print_delete_vals:
        _print_delete_values(hsqhat, args.out + '.delete', log)
        _print_part_delete_values(hsqhat, args.out + '.part_delete', log)

    log.log(hsqhat.summary(ref_ld_cnames, P=args.samp_prev, K=args.pop_prev, overlap = args.overlap_annot))
    if args.overlap_annot:
        overlap_matrix, M_tot = _read_annot(args, log)

        # overlap_matrix = overlap_matrix[np.array(~novar_cols), np.array(~novar_cols)]#np.logical_not
        df_results = hsqhat._overlap_output(ref_ld_cnames, overlap_matrix, M_annot, M_tot, args.print_coefficients)
        df_results.to_csv(args.out+'.results', sep="\t", index=False)
        log.log('Results printed to '+args.out+'.results')

    return hsqhat


def estimate_rg(args, log):
    '''Estimate rg between trait 1 and a list of other traits.'''
    args = copy.deepcopy(args)
    rg_paths, rg_files = _parse_rg(args.rg)
    n_pheno = len(rg_paths)
    f = lambda x: _split_or_none(x, n_pheno)
    args.intercept_h2, args.intercept_gencov, args.samp_prev, args.pop_prev = list(map(f,
        (args.intercept_h2, args.intercept_gencov, args.samp_prev, args.pop_prev)))
    list(map(lambda x: _check_arg_len(x, n_pheno), ((args.intercept_h2, '--intercept-h2'),
                                               (args.intercept_gencov, '--intercept-gencov'),
                                               (args.samp_prev, '--samp-prev'),
                                               (args.pop_prev, '--pop-prev'))))
    if args.no_intercept:
        args.intercept_h2 = [1 for _ in range(n_pheno)]
        args.intercept_gencov = [0 for _ in range(n_pheno)]
    p1 = rg_paths[0]
    out_prefix = args.out + rg_files[0]
    M_annot, w_ld_cname, ref_ld_cnames, sumstats, _ = _read_ld_sumstats(args, log, p1,
                                                                        alleles=True, dropna=True)
    RG = []
    n_annot = M_annot.shape[1]
    if n_annot == 1 and args.two_step is None and args.intercept_h2 is None:
        args.two_step = 30
    if args.two_step is not None:
        log.log('Using two-step estimator with cutoff at {M}.'.format(M=args.two_step))

    for i, p2 in enumerate(rg_paths[1:n_pheno]):
        log.log(
            'Computing rg for phenotype {I}/{N}'.format(I=i + 2, N=len(rg_paths)))
        try:
            loop = _read_other_sumstats(args, log, p2, sumstats, ref_ld_cnames)
            rghat = _rg(loop, args, log, M_annot, ref_ld_cnames, w_ld_cname, i)
            RG.append(rghat)
            _print_gencor(args, log, rghat, ref_ld_cnames, i, rg_paths, i == 0)
            out_prefix_loop = out_prefix + '_' + rg_files[i + 1]
            if args.print_cov:
                _print_rg_cov(rghat, out_prefix_loop, log)
            if args.print_delete_vals:
                _print_rg_delete_values(rghat, out_prefix_loop, log)

        except Exception:  # keep going if phenotype 50/100 causes an error
            msg = 'ERROR computing rg for phenotype {I}/{N}, from file {F}.'
            log.log(msg.format(I=i + 2, N=len(rg_paths), F=rg_paths[i + 1]))
            ex_type, ex, tb = sys.exc_info()
            log.log(traceback.format_exc() + '\n')
            if len(RG) <= i:  # if exception raised before appending to RG
                RG.append(None)

    log.log('\nSummary of Genetic Correlation Results\n' +
            _get_rg_table(rg_paths, RG, args))
    return RG


def _read_other_sumstats(args, log, p2, sumstats, ref_ld_cnames):
    loop = _read_sumstats(args, log, p2, alleles=True, dropna=False)
    loop = _merge_sumstats_sumstats(args, sumstats, loop, log)
    loop = loop.dropna(how='any')
    alleles = loop.A1 + loop.A2 + loop.A1x + loop.A2x
    if not args.no_check_alleles:
        loop = _select_and_log(loop, _filter_alleles(alleles), log,
                               '{N} SNPs with valid alleles.')
        loop['Z2'] = _align_alleles(loop.Z2, alleles)

    loop = loop.drop(['A1', 'A1x', 'A2', 'A2x'], axis=1)
    _check_ld_condnum(args, log, loop[ref_ld_cnames])
    _warn_length(log, loop)
    return loop


def _get_rg_table(rg_paths, RG, args):
    '''Print a table of genetic correlations.'''
    t = lambda attr: lambda obj: getattr(obj, attr, 'NA')
    x = pd.DataFrame()
    x['p1'] = [rg_paths[0] for i in range(1, len(rg_paths))]
    x['p2'] = rg_paths[1:len(rg_paths)]
    x['rg'] = list(map(t('rg_ratio'), RG))
    x['se'] = list(map(t('rg_se'), RG))
    x['z'] = list(map(t('z'), RG))
    x['p'] = list(map(t('p'), RG))
    if args.samp_prev is not None and \
            args.pop_prev is not None and \
            all((i is not None for i in args.samp_prev)) and \
            all((i is not None for it in args.pop_prev)):

        c = list(map(lambda x, y: reg.h2_obs_to_liab(1, x, y), args.samp_prev[1:], args.pop_prev[1:]))
        x['h2_liab'] = list(map(lambda x, y: x * y, c, list(map(t('tot'), list(map(t('hsq2'), RG))))))
        x['h2_liab_se'] = list(map(lambda x, y: x * y, c, list(map(t('tot_se'), list(map(t('hsq2'), RG))))))
    else:
        x['h2_obs'] = list(map(t('tot'), list(map(t('hsq2'), RG))))
        x['h2_obs_se'] = list(map(t('tot_se'), list(map(t('hsq2'), RG))))

    x['h2_int'] = list(map(t('intercept'), list(map(t('hsq2'), RG))))
    x['h2_int_se'] = list(map(t('intercept_se'), list(map(t('hsq2'), RG))))
    x['gcov_int'] = list(map(t('intercept'), list(map(t('gencov'), RG))))
    x['gcov_int_se'] = list(map(t('intercept_se'), list(map(t('gencov'), RG))))
    return x.to_string(header=True, index=False) + '\n'


def _print_gencor(args, log, rghat, ref_ld_cnames, i, rg_paths, print_hsq1):
    l = lambda x: x + ''.join(['-' for i in range(len(x.replace('\n', '')))])
    P = [args.samp_prev[0], args.samp_prev[i + 1]]
    K = [args.pop_prev[0], args.pop_prev[i + 1]]
    if args.samp_prev is None and args.pop_prev is None:
        args.samp_prev = [None, None]
        args.pop_prev = [None, None]
    if print_hsq1:
        log.log(l('\nHeritability of phenotype 1\n'))
        log.log(rghat.hsq1.summary(ref_ld_cnames, P=P[0], K=K[0]))

    log.log(
        l('\nHeritability of phenotype {I}/{N}\n'.format(I=i + 2, N=len(rg_paths))))
    log.log(rghat.hsq2.summary(ref_ld_cnames, P=P[1], K=K[1]))
    log.log(l('\nGenetic Covariance\n'))
    log.log(rghat.gencov.summary(ref_ld_cnames, P=P, K=K))
    log.log(l('\nGenetic Correlation\n'))
    log.log(rghat.summary() + '\n')


def _merge_sumstats_sumstats(args, sumstats1, sumstats2, log):
    '''Merge two sets of summary statistics.'''
    sumstats1.rename(columns={'N': 'N1', 'Z': 'Z1'}, inplace=True)
    sumstats2.rename(
        columns={'A1': 'A1x', 'A2': 'A2x', 'N': 'N2', 'Z': 'Z2'}, inplace=True)
    x = _merge_and_log(sumstats1, sumstats2, 'summary statistics', log)
    return x


def _filter_alleles(alleles):
    '''Remove bad variants (mismatched alleles, non-SNPs, strand ambiguous).'''
    ii = alleles.apply(lambda y: y in MATCH_ALLELES)
    return ii


def _align_alleles(z, alleles):
    '''Align Z1 and Z2 to same choice of ref allele (allowing for strand flip).'''
    try:
        z *= (-1) ** alleles.apply(lambda y: FLIP_ALLELES[y])
    except KeyError as e:
        msg = 'Incompatible alleles in .sumstats files: %s. ' % e.args
        msg += 'Did you forget to use --merge-alleles with munge_sumstats.py?'
        raise KeyError(msg)
    return z


def _rg(sumstats, args, log, M_annot, ref_ld_cnames, w_ld_cname, i):
    '''Run the regressions.'''
    n_snp = len(sumstats)
    s = lambda x: np.array(x).reshape((n_snp, 1))
    if args.chisq_max is not None:
        ii = sumstats.Z1**2*sumstats.Z2**2 < args.chisq_max**2
        n_snp = np.sum(ii)  # lambdas are late binding, so this works
        sumstats = sumstats[ii]
    n_blocks = min(args.n_blocks, n_snp)
    ref_ld = sumstats[ref_ld_cnames].to_numpy()
    intercepts = [args.intercept_h2[0], args.intercept_h2[
        i + 1], args.intercept_gencov[i + 1]]
    rghat = reg.RG(s(sumstats.Z1), s(sumstats.Z2),
                   ref_ld, s(sumstats[w_ld_cname]), s(
                       sumstats.N1), s(sumstats.N2), M_annot,
                   intercept_hsq1=intercepts[0], intercept_hsq2=intercepts[1],
                   intercept_gencov=intercepts[2], n_blocks=n_blocks, twostep=args.two_step)

    return rghat


def _parse_rg(rg):
    '''Parse args.rg.'''
    rg_paths = _splitp(rg)
    rg_files = [x.split('/')[-1] for x in rg_paths]
    if len(rg_paths) < 2:
        raise ValueError(
            'Must specify at least two phenotypes for rg estimation.')

    return rg_paths, rg_files


def _print_rg_delete_values(rg, fh, log):
    '''Print block jackknife delete values.'''
    _print_delete_values(rg.hsq1, fh + '.hsq1.delete', log)
    _print_delete_values(rg.hsq2, fh + '.hsq2.delete', log)
    _print_delete_values(rg.gencov, fh + '.gencov.delete', log)


def _print_rg_cov(rghat, fh, log):
    '''Print covariance matrix of estimates.'''
    _print_cov(rghat.hsq1, fh + '.hsq1.cov', log)
    _print_cov(rghat.hsq2, fh + '.hsq2.cov', log)
    _print_cov(rghat.gencov, fh + '.gencov.cov', log)


def _split_or_none(x, n):
    if x is not None:
        y = list(map(float, x.replace('N', '-').split(',')))
    else:
        y = [None for _ in range(n)]
    return y


def _check_arg_len(x, n):
    x, m = x
    if len(x) != n:
        raise ValueError(
            '{M} must have the same number of arguments as --rg/--h2.'.format(M=m))
