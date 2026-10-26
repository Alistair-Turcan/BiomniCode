
import numpy as np
import bitarray as ba

# Packed 2-bit PLINK bed code (LSB-first within each byte) -> genotype value used
# by ldsc (0/1/2 = genotype, 9 = missing). Verified bit-identical to
# bitarray.decode(_bedcode) on all 256 byte values.
_BED_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)
_BED_LUT = np.array([0.0, 9.0, 1.0, 2.0])

# Per-byte aggregate LUTs for __filter_snps_maf__ (same LSB-first code order as
# the (m, nru) expansion): for byte v, the counts over its four 2-bit codes of
# first-bit-set, second-bit-set, and both-bits-set (code == 3). Padding codes
# are 0 and contribute 0 to all three counts.
_BED_BYTE_CODES = np.bitwise_and(np.arange(256, dtype=np.uint8)[:, None] >> _BED_SHIFTS, 3)
_BED_CNT1 = np.count_nonzero(_BED_BYTE_CODES & 1, axis=1).astype(np.uint8)
_BED_CNT2 = np.count_nonzero(_BED_BYTE_CODES >> 1, axis=1).astype(np.uint8)
_BED_CNT3 = np.count_nonzero(_BED_BYTE_CODES == 3, axis=1).astype(np.uint8)


def getBlockLefts(coords, max_dist):
    '''
    Converts coordinates + max block length to the a list of coordinates of the leftmost
    SNPs to be included in blocks.

    Parameters
    ----------
    coords : array
        Array of coordinates. Must be sorted.
    max_dist : float
        Maximum distance between SNPs included in the same window.

    Returns
    -------
    block_left : 1D np.ndarray with same length as block_left
        block_left[j] :=  min{k | dist(j, k) < max_dist}.

    '''
    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in range(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1

        block_left[i] = j

    return block_left


def block_left_to_right(block_left):
    '''
    Converts block lefts to block rights.

    Parameters
    ----------
    block_left : array
        Array of block lefts.

    Returns
    -------
    block_right : 1D np.ndarray with same length as block_left
        block_right[j] := max {k | block_left[k] <= j}

    '''
    M = len(block_left)
    j = 0
    block_right = np.zeros(M)
    for i in range(M):
        while j < M and block_left[j] <= i:
            j += 1

        block_right[i] = j

    return block_right


class __GenotypeArrayInMemory__(object):
    '''
    Parent class for various classes containing interfaces for files with genotype
    matrices, e.g., plink .bed files, etc
    '''
    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
        self.m = len(snp_list.IDList)
        self.n = n
        self.keep_snps = keep_snps
        self.keep_indivs = keep_indivs
        self.df = np.array(snp_list.df[['CHR', 'SNP', 'BP', 'CM']])
        self.colnames = ['CHR', 'SNP', 'BP', 'CM']
        self.mafMin = mafMin if mafMin is not None else 0
        self._currentSNP = 0
        (self.nru, self.geno) = self.__read__(fname, self.m, n)
        # filter individuals
        if keep_indivs is not None:
            keep_indivs = np.array(keep_indivs, dtype='int')
            if np.any(keep_indivs > self.n):
                raise ValueError('keep_indivs indices out of bounds')

            (self.geno, self.m, self.n) = self.__filter_indivs__(self.geno, keep_indivs, self.m,
                self.n)

            if self.n > 0:
                print('After filtering, {n} individuals remain'.format(n=self.n))
            else:
                raise ValueError('After filtering, no individuals remain')

        # filter SNPs
        if keep_snps is not None:
            keep_snps = np.array(keep_snps, dtype='int')
            if np.any(keep_snps > self.m):  # if keep_snps is None, this returns False
                raise ValueError('keep_snps indices out of bounds')

        (self.geno, self.m, self.n, self.kept_snps, self.freq) = self.__filter_snps_maf__(
            self.geno, self.m, self.n, self.mafMin, keep_snps)

        if self.m > 0:
            print('After filtering, {m} SNPs remain'.format(m=self.m))
        else:
            raise ValueError('After filtering, no SNPs remain')

        self.df = self.df[self.kept_snps, :]
        self.maf = np.minimum(self.freq, np.ones(self.m)-self.freq)
        self.sqrtpq = np.sqrt(self.freq*(np.ones(self.m)-self.freq))
        self.df = np.c_[self.df, self.maf]
        self.colnames.append('MAF')

    def __read__(self, fname, m, n):
        raise NotImplementedError

    def __filter_indivs__(geno, keep_indivs, m, n):
        raise NotImplementedError

    def __filter_maf_(geno, m, n, maf):
        raise NotImplementedError

    def ldScoreVarBlocks(self, block_left, c, annot=None):
        '''Computes an unbiased estimate of L2(j) for j=1,..,M.'''
        func = lambda x: self.__l2_unbiased__(x, self.n)
        snp_getter = self.nextSNPs
        return self.__corSumVarBlocks__(block_left, c, func, snp_getter, annot)

    def ldScoreBlockJackknife(self, block_left, c, annot=None, jN=10):
        func = lambda x: np.square(x)
        snp_getter = self.nextSNPs
        return self.__corSumBlockJackknife__(block_left, c, func, snp_getter, annot, jN)

    def __l2_unbiased__(self, x, n):
        denom = n-2 if n > 2 else n  # allow n<2 for testing purposes
        # in-place: x <- x^2 - (1 - x^2)/denom, elementwise ops in the same
        # order as the original expression, so results are bitwise-identical
        np.square(x, out=x)
        x -= (1-x) / denom
        return x

    # general methods for calculating sums of Pearson correlation coefficients
    def __corSumVarBlocks__(self, block_left, c, func, snp_getter, annot=None):
        '''
        Parameters
        ----------
        block_left : np.ndarray with shape (M, )
            block_left[i] = index of leftmost SNP included in LD Score of SNP i.
            if c > 1, then only entries that are multiples of c are examined, and it is
            assumed that block_left[a*c+i] = block_left[a*c], except at
            the beginning of the chromosome where the 0th SNP is included in the window.

        c : int
            Chunk size.
        func : function
            Function to be applied to the genotype correlation matrix. Before dotting with
            annot. Examples: for biased L2, np.square. For biased L4,
            lambda x: np.square(np.square(x)). For L1, lambda x: x.
        snp_getter : function(int)
            The method to be used to get the next SNPs (normalized genotypes? Normalized
            genotypes with the minor allele as reference allele? etc)
        annot: numpy array with shape (m,n_a)
            SNP annotations.

        Returns
        -------
        cor_sum : np.ndarray with shape (M, num_annots)
            Estimates.

        '''
        m, n = self.m, self.n
        block_sizes = np.array(np.arange(m) - block_left)
        block_sizes = np.ceil(block_sizes / c)*c
        check_annot = annot is not None
        if annot is None:
            annot = np.ones((m, 1))
        else:
            annot_m = annot.shape[0]
            if annot_m != self.m:
                raise ValueError('Incorrect number of SNPs in annot')

        n_a = annot.shape[1]  # number of annotations
        cor_sum = np.zeros((m, n_a))
        # b = index of first SNP for which SNP 0 is not included in LD Score
        b = np.nonzero(block_left > 0)
        if np.any(b):
            b = b[0][0]
        else:
            b = m
        b = int(np.ceil(b/c)*c)  # round up to a multiple of c
        if b > m:
            c = 1
            b = m
        l_A = 0  # l_A := index of leftmost SNP in matrix A
        A = snp_getter(b)
        rfuncAB = np.zeros((b, c))
        rfuncBB = np.zeros((c, c))
        # chunk inside of block
        for l_B in range(0, b, c):  # l_B := index of leftmost SNP in matrix B
            B = A[:, l_B:l_B+c]
            np.dot(A.T, B / n, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            cor_sum[l_A:l_A+b, :] += np.dot(rfuncAB, annot[l_B:l_B+c, :])
        # chunk to right of block
        b0 = b
        md = int(c*np.floor(m/c))
        end = md + 1 if md != m else md
        # Ring buffer for the sliding window A (F-order so window views stay
        # contiguous). Replaces the per-iteration np.hstack (fresh alloc + copy
        # of the whole window) with pointer arithmetic plus an append of the new
        # chunk. Capacity is 4x the maximum window so compactions are rare and
        # amortized away (it03 showed cap=max_b+c degenerates to a full-window
        # copy every iteration). Window contents (column set and order) are
        # identical to the hstack formulation, so results are unchanged.
        max_b = int(block_sizes[b0:].max()) if b0 < m else 1
        cap = 4*max_b + c
        A_buf = np.empty((n, cap), order='F')
        # Scratch for the A.T x B/n dot: one max-size buffer, sliced per
        # iteration (np.dot(out=) overwrites every element, so the original's
        # per-b-change np.zeros reallocation is pure churn). Rows [:b] of a
        # (max_b, c) buffer are C-contiguous, as out= requires. The final
        # partial chunk (l_B == md) changes c, so it gets a fresh buffer.
        rfuncAB_full = np.empty((max_b, c))
        a_start = 0
        for l_B in range(b0, end, c):
            # check if the annot matrix is all zeros for this block + chunk
            # this happens w/ sparse categories (i.e., pathways)
            # update the block
            old_b = b
            b = int(block_sizes[l_B])
            if l_B > b0 and b > 0:
                # block_size can't increase more than c
                # block_size can't be less than c unless it is zero
                # both of these things make sense
                d = old_b - b + c  # left-shift distance; 0 <= d <= old_b
                if a_start + old_b + c > cap:
                    # compact: move kept columns [a_start+d, a_start+old_b) to [0, b-c)
                    nkeep = b - c
                    if nkeep > 0:
                        A_buf[:, 0:nkeep] = A_buf[:, a_start+d:a_start+old_b]
                    a_start = 0
                else:
                    a_start += d
                A_buf[:, a_start+b-c:a_start+b] = B
                l_A += d
            elif l_B == b0 and b > 0:
                A_buf[:, 0:b] = A[:, b0-b:b0]
                a_start = 0
                l_A = b0-b
            elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
                a_start = 0
                l_A = l_B
            A = A_buf[:, a_start:a_start+b]
            if l_B == md:
                c = m - md
                rfuncAB = np.empty((b, c))
                rfuncBB = np.empty((c, c))
            elif b != old_b:
                rfuncAB = rfuncAB_full[:b]

            B = snp_getter(c)
            # With annot=None the ones-matrix makes p1/p2 always False, so the
            # all-zero check only runs when real annotations were supplied
            # (bitwise-identical: with ones the continue could never fire).
            if check_annot:
                p1 = np.all(annot[l_A:l_A+b, :] == 0)
                p2 = np.all(annot[l_B:l_B+c, :] == 0)
                if p1 and p2:
                    continue

            Bn = B / n  # one deterministic elementwise divide feeds both dots
            np.dot(A.T, Bn, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            cor_sum[l_A:l_A+b, :] += np.dot(rfuncAB, annot[l_B:l_B+c, :])
            cor_sum[l_B:l_B+c, :] += np.dot(annot[l_A:l_A+b, :].T, rfuncAB).T
            np.dot(B.T, Bn, out=rfuncBB)
            rfuncBB = func(rfuncBB)
            cor_sum[l_B:l_B+c, :] += np.dot(rfuncBB, annot[l_B:l_B+c, :])

        return cor_sum


class PlinkBEDFile(__GenotypeArrayInMemory__):
    '''
    Interface for Plink .bed format
    '''
    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
        self._bedcode = {
            2: ba.bitarray('11'),
            9: ba.bitarray('10'),
            1: ba.bitarray('01'),
            0: ba.bitarray('00')
            }

        self._geno_bytes = None  # lazy cache of self.geno as bytes (set in nextSNPs)
        __GenotypeArrayInMemory__.__init__(self, fname, n, snp_list, keep_snps=keep_snps,
            keep_indivs=keep_indivs, mafMin=mafMin)

    def __read__(self, fname, m, n):
        if not fname.endswith('.bed'):
            raise ValueError('.bed filename must end in .bed')

        fh = open(fname, 'rb')
        magicNumber = ba.bitarray(endian="little")
        magicNumber.fromfile(fh, 2)
        bedMode = ba.bitarray(endian="little")
        bedMode.fromfile(fh, 1)
        e = (4 - n % 4) if n % 4 != 0 else 0
        nru = n + e
        self.nru = nru
        # check magic number
        if magicNumber != ba.bitarray('0011011011011000'):
            raise IOError("Magic number from Plink .bed file not recognized")

        if bedMode != ba.bitarray('10000000'):
            raise IOError("Plink .bed file must be in default SNP-major mode")

        # check file length
        self.geno = ba.bitarray(endian="little")
        self.geno.fromfile(fh)
        self.__test_length__(self.geno, self.m, self.nru)
        return (self.nru, self.geno)

    def __test_length__(self, geno, m, nru):
        exp_len = 2*m*nru
        real_len = len(geno)
        if real_len != exp_len:
            s = "Plink .bed file has {n1} bits, expected {n2}"
            raise IOError(s.format(n1=real_len, n2=exp_len))

    def __filter_indivs__(self, geno, keep_indivs, m, n):
        n_new = len(keep_indivs)
        e = (4 - n_new % 4) if n_new % 4 != 0 else 0
        nru_new = n_new + e
        nru = self.nru
        z = ba.bitarray(m*2*nru_new, endian="little")
        z.setall(0)
        for e, i in enumerate(keep_indivs):
            z[2*e::2*nru_new] = geno[2*i::2*nru]
            z[2*e+1::2*nru_new] = geno[2*i+1::2*nru]

        self.nru = nru_new
        return (z, m, n_new)

    def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
        '''
        Credit to Chris Chang and the Plink2 developers for this algorithm
        Modified from plink_filter.c
        https://github.com/chrchang/plink-ng/blob/master/plink_filter.c

        Genotypes are read forwards (since we are cheating and using endian="little")

        A := (genotype) & 1010...
        B := (genotype) & 0101...
        C := (A >> 1) & B

        Then

        a := A.count() = missing ct + hom major ct
        b := B.count() = het ct + hom major ct
        c := C.count() = hom major ct

        Which implies that

        missing ct = a - c
        # of indivs with nonmissing genotype = n - a + c
        major allele ct = b + c
        major allele frequency = (b+c)/(2*(n-a+c))
        het ct + missing ct = a + b - 2*c

        Why does bitarray not have >> ????

        '''
        nru = self.nru
        bpb = nru // 4
        if keep_snps is None:
            idx = np.arange(m)
        else:
            idx = np.asarray(keep_snps, dtype='int')
        # Byte-LUT equivalent of the packed 2-bit expansion: per-byte code
        # counts summed across each row with an int64 accumulator (uint8 would
        # overflow: 4*bpb can exceed 255). Padding codes are zero and
        # contribute 0 to all three counts, exactly as in the expanded
        # (m, nru) formulation, so the integer counts are unchanged.
        Gb = np.frombuffer(geno.tobytes(), dtype=np.uint8).reshape(m, bpb)
        Gs = Gb[idx]
        a = np.add.reduce(_BED_CNT1[Gs], axis=1, dtype=np.int64)   # missing + hom major
        b2 = np.add.reduce(_BED_CNT2[Gs], axis=1, dtype=np.int64)  # het + hom major
        c = np.add.reduce(_BED_CNT3[Gs], axis=1, dtype=np.int64)   # hom major
        major_ct = b2 + c  # number of copies of the major allele
        n_nomiss = n - a + c  # number of individuals with nonmissing genotypes
        f = np.zeros(len(idx))
        np.divide(major_ct, 2.0*n_nomiss, out=f, where=(n_nomiss > 0))
        het_miss_ct = a + b2 - 2*c  # remove SNPs that are only either het or missing
        keep = (np.minimum(f, 1-f) > mafMin) & (het_miss_ct < n)
        kept = idx[keep]
        y_bytes = Gb[kept].tobytes()
        y = ba.bitarray(endian='little')
        y.frombytes(y_bytes)
        # Bytes of the returned (final) genotype matrix; nextSNPs reuses them
        # instead of re-serializing self.geno.
        self._geno_bytes = y_bytes
        return (y, len(kept), n, list(kept), [float(x) for x in f[keep]])

    def nextSNPs(self, b, minorRef=None):
        '''
        Unpacks the binary array of genotypes and returns an n x b matrix of floats of
        normalized genotypes for the next b SNPs, where n := number of samples.

        Parameters
        ----------
        b : int
            Number of SNPs to return.
        minorRef: bool, default None
            Should we flip reference alleles so that the minor allele is the reference?
            (This is useful for computing l1 w.r.t. minor allele).

        Returns
        -------
        X : np.array with dtype float64 with shape (n, b), where n := number of samples
            Matrix of genotypes normalized to mean zero and variance one. If minorRef is
            not None, then the minor allele will be the positive allele (i.e., two copies
            of the minor allele --> a positive number).

        '''

        try:
            b = int(b)
            if b <= 0:
                raise ValueError("b must be > 0")
        except TypeError:
            raise TypeError("b must be an integer")

        if self._currentSNP + b > self.m:
            s = '{b} SNPs requested, {k} SNPs remain'
            raise ValueError(s.format(b=b, k=(self.m-self._currentSNP)))

        c = self._currentSNP
        n = self.n
        nru = self.nru
        bpb = nru // 4
        if self._geno_bytes is None:
            self._geno_bytes = self.geno.tobytes()
        a = np.frombuffer(self._geno_bytes, dtype=np.uint8, count=b*bpb, offset=c*bpb)
        codes = np.bitwise_and(a.reshape(b, bpb, 1) >> _BED_SHIFTS, 3).reshape(b, nru)
        G = _BED_LUT[codes][:, 0:n]  # (b, n) float64 genotypes; rows are SNPs
        # Vectorized per-SNP normalization. Reductions run along the contiguous
        # axis of each SNP row, matching the pairwise-summation order of the
        # original per-SNP np.mean/np.std calls, so results are bitwise-identical
        # for SNPs without missing genotypes. SNPs with missing genotypes (none
        # in 1000G Phase 3) fall back to the exact original per-SNP computation.
        # Missing genotypes are exactly code 1 (the only code _BED_LUT maps to
        # 9.0), and padding codes are 0, so scanning the uint8 codes gives the
        # same rows as scanning the float64 block.
        miss_rows = np.nonzero((codes == 1).any(axis=1))[0]
        avg = G.sum(axis=1) / n
        D = G - avg[:, None]
        D *= D
        denom = np.sqrt(D.sum(axis=1) / n)
        denom[denom == 0] = 1
        if minorRef is not None:
            flip = np.array([self.freq[c + j] > 0.5 for j in range(b)])
            denom[flip] *= -1
        Y = np.empty((n, b))
        Yt = Y.T
        np.subtract(G, avg[:, None], out=Yt)
        Yt /= denom[:, None]
        for j in miss_rows:
            newsnp = G[j].copy()
            ii = newsnp != 9
            avg_j = np.mean(newsnp[ii])
            newsnp[np.logical_not(ii)] = avg_j
            denom_j = np.std(newsnp)
            if denom_j == 0:
                denom_j = 1

            if minorRef is not None and self.freq[c + j] > 0.5:
                denom_j = denom_j*-1

            Y[:, j] = (newsnp - avg_j) / denom_j

        self._currentSNP += b
        return Y
