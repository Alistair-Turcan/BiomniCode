from typing import Optional, Tuple, Union

import re, os, sys, time, tempfile, shutil
import numpy as np

from . import log, util, wald


def _sm():
    """Lazy statsmodels.api import (~0.9s of module import time; only the
    OLS beta blocks in the HE wrappers need it)."""
    import statsmodels.api as sm
    return sm



def _ZZT(random_covars: dict) -> dict:
    """
    Compute Z @ Z.T

    Parameters:
        random_covars:  design matrices for Extra random effects

    Returns:
        # Z @ Z.T of design matrices for Extra random effects
    """
    random_covars_ZZT = {}
    for key, Z in random_covars.items():
        random_covars_ZZT[key] = Z @ Z.T
    return random_covars_ZZT


def cal_Vy(hom_g2: float, hom_e2: float, V: np.ndarray, W: np.ndarray, r2:dict,
           K: np.ndarray, ctnu: np.ndarray, random_covars_ZZT:dict, K2: Optional[np.ndarray] = None, 
           hom2_g2: Optional[float] = None, V2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute covariance matrix of vectorized Cell Type-specific Pseudobulk

    Parameters:
        hom_g2:   variance of genetic effect shared across cell types
        hom_e2:   variance of env effect shared across cell types
        V:  covariance matrix of cell type-specific genetic effect
        W:  covariance matrix of cell type-specific environment effect
        r2: variance of Extra random effect
        K:  kinship matrix
        ctnu:   cell type-specific noise variance
        random_covars_ZZT:  Z @ Z.T of design matrices for Extra random effects
        K2: trans kinship matrix
        hom2_g2:  variance of trans genetic effect shared across cell types
        V2: covariance matrix of cell type-specific trans genetic effect

    Returns:
        covariance matrix of vectorized Cell Type-specific Pseudobulk V(y)
    """

    N, C = ctnu.shape
    A = hom_g2 * np.ones((C,C), dtype='int8') + V
    B = hom_e2 * np.ones((C,C), dtype='int8') + W
    NC = N * C
    # Bitwise-identical fast assembly of the same sum of Kronecker products.
    # Every cell receives exactly the same products (K[i,j]*A[c,d] etc.) added
    # in the same order as the np.kron(...) formulation below; only the
    # assembly layout differs (blocked strided writes instead of 4-D broadcast
    # temporaries, zero-products of kron(eye(N), B) skipped since x+0.0 == x,
    # and the ctnu diagonal fused). Measured ~2.2x faster than np.kron at
    # NC=400.
    Vy = np.empty((NC, NC))
    # dedup block products by scalar value: A often has few distinct entries
    # (e.g. Free model A = hom_g2*ones + diag(V) has C+1), and K*a computed
    # once per distinct value is bitwise-identical to per-block recomputation
    _KA = {}
    for c in range(C):
        for d in range(C):
            a = A[c, d]
            if a not in _KA:
                _KA[a] = K * a
            Vy[c::C, d::C] = _KA[a]
    # += kron(eye(N), B): only the N diagonal CxC blocks have nonzero products
    Vy4 = Vy.reshape(N, C, N, C)
    Vy4[np.arange(N), :, np.arange(N), :] += B
    # += diag(ctnu.flatten())
    Vy.flat[::NC + 1] += ctnu.flatten()
    for key in random_covars_ZZT.keys():
        ZZT = random_covars_ZZT[key]
        if isinstance(r2[key], float):
            # kron(ZZT, ones(C,C)) * r2: products ZZT[i,j]*1.0 are exact;
            # 4-D broadcast add avoids materializing (NC,NC) repeat temporaries
            Vy4 += ZZT[:, None, :, None] * r2[key] # shared random effect
        else:
            # kron(ZZT, diag(r2[key])): nonzero products only at c == d
            rk = r2[key]
            for c in range(C):
                Vy[c::C, c::C] += ZZT * rk[c] # cell type-specific random effect

    if K2 is not None:
        assert hom2_g2 is not None and V2 is not None
        At = hom2_g2 * np.ones((C,C), dtype='int8') + V2
        _KAt = {}
        for c in range(C):
            for d in range(C):
                a = At[c, d]
                if a not in _KAt:
                    _KAt[a] = K2 * a
                Vy[c::C, d::C] += _KAt[a]

    return Vy


def LL(y: np.ndarray, K: np.ndarray, X: np.ndarray, ctnu: np.ndarray, 
       random_covars_ZZT:dict, hom_g2: float, hom_e2: float, V: np.ndarray, 
       W: np.ndarray, r2:dict, K2: Optional[np.ndarray] = None, 
       hom2_g2: Optional[float] = None, V2: Optional[np.ndarray] = None, 
       ) -> float:
    """
    Loglikelihood function

    Parameters:
        y:  vectorized cell type-specific pseudobulk, vec(Y^T)
        K:  kinship matrix
        X:  design matrix for fixed effects
        ctnu:   cell type-specific noise variance
        random_covars_ZZT:  Z @ Z.T of design matrices for Extra random effects
        hom_g2: variance of genetic effect shared across cell types
        hom_e2: variance of env effect shared across cell types
        V:  covariance matrix of cell type-specific genetic effect
        W:  covariance matrix of cell type-specific env effect
        r2: variance of Extra random effect
        K2: kinship matrix for a second set of SNPs e.g. trans
        hom2_g2: variance of genetic effect shared across cell types for a second set of SNPs e.g. trans
        V2: covariance matrix of cell type-specific genetic effect for a second set of SNPs e.g. trans
    Returns:
        loglikelihood
    """

    from scipy import linalg as sla  # deferred: not on HE path
    N, C = ctnu.shape
    Vy = cal_Vy(hom_g2, hom_e2, V, W, r2, K, ctnu, random_covars_ZZT, 
                K2, hom2_g2, V2)

    # inverse variance
    w, v = sla.eigh(Vy, check_finite=False)
    if ( np.amax(w)/np.amin(w) ) > 1e8 or np.amin(w) < 0:
        return 1e12
    
    # calculate B matrix
    m1 = X.T @ v @ np.diag(1/w) @ v.T 
    m2 = m1 @ X

    # calculate loglikelihood
    det_Vy = np.sum( np.log(w) )
    det_XVyX = np.linalg.slogdet(m2)[1]
    yBy = y @ v @ np.diag(1/w) @ v.T @ y - y @ m1.T @ sla.inv(m2, check_finite=False) @ m1 @ y
    L = det_Vy + det_XVyX + yBy
    L = 0.5 * L

    return L



def LL_chol(y: np.ndarray, K: np.ndarray, X: np.ndarray, ctnu: np.ndarray, 
       random_covars_ZZT:dict, hom_g2: float, hom_e2: float, V: np.ndarray, 
       W: np.ndarray, r2:dict, K2: Optional[np.ndarray] = None, 
       hom2_g2: Optional[float] = None, V2: Optional[np.ndarray] = None, 
       ) -> float:
    """
    Loglikelihood function

    Parameters:
        y:  vectorized cell type-specific pseudobulk, vec(Y^T)
        K:  kinship matrix
        X:  design matrix for fixed effects
        ctnu:   cell type-specific noise variance
        random_covars_ZZT:  Z @ Z.T of design matrices for Extra random effects
        hom_g2: variance of genetic effect shared across cell types
        hom_e2: variance of env effect shared across cell types
        V:  covariance matrix of cell type-specific genetic effect
        W:  covariance matrix of cell type-specific env effect
        r2: variance of Extra random effect
        K2: kinship matrix for a second set of SNPs e.g. trans
        hom2_g2: variance of genetic effect shared across cell types for a second set of SNPs e.g. trans
        V2: covariance matrix of cell type-specific genetic effect for a second set of SNPs e.g. trans
    Returns:
        loglikelihood
    """

    from scipy import linalg as sla  # deferred: not on HE path
    N, C = ctnu.shape
    Vy = cal_Vy(hom_g2, hom_e2, V, W, r2, K, ctnu, random_covars_ZZT, 
                K2, hom2_g2, V2)

    # cholesky decomposition, in place via LAPACK potrf on the F-contiguous
    # transpose view: Vy is symmetric by construction (cal_Vy sums symmetric
    # terms) and freshly allocated per eval, so Vy.T holds the same values in
    # Fortran order and can be factorized in place. UPLO='L' on the view reads
    # the same triangle as scipy.cholesky(lower=True) -> bitwise-identical L,
    # while skipping scipy's C->F copy of the NCxNC matrix and its wrapper
    # overhead. clean=0 leaves the view's upper triangle untouched; it is
    # never read (potrs UPLO='L' and np.diag only).
    try:
        L = Vy.T
        _potrf, = sla.get_lapack_funcs(('potrf',), (L,))
        L, _info = _potrf(L, lower=1, overwrite_a=1, clean=0)
        if _info != 0:
            raise np.linalg.LinAlgError(f'Vy not positive definite (potrf info={_info})')
    except np.linalg.LinAlgError:
        return 1e12

    # calculate B matrix; X and y solved in one dpotrs call (column-independent
    # triangular solves -> bitwise-identical columns), skipping per-call finite
    # validation (inputs are finite by construction in the optimizer loop)
    _potrs, = sla.get_lapack_funcs(('potrs',), (L,))
    sol, _info = _potrs(L, np.column_stack([X, y]), lower=1, overwrite_b=0)
    m1 = sol[:, :-1]
    m2 = X.T @ m1

    # calculate loglikelihood
    det_Vy = 2 * np.sum(np.log(np.diag(L)))
    det_XVyX = np.linalg.slogdet(m2)[1]
    yBy = y @ sol[:, -1] - y @ m1 @ sla.inv(m2, check_finite=False) @ m1.T @ y
    L = det_Vy + det_XVyX + yBy
    L = 0.5 * L

    return L


def _get_r2(r2: Union[list, np.ndarray], random_covars:dict, C:int) -> dict:
    """
    Covert list of r2 to dictionary
    """
    if len(r2) == len(random_covars.keys()):
        shared = True
    else:
        shared = False

    r2_d = {}
    for i, key in enumerate(sorted(random_covars.keys())):
        if shared:
            r2_d[key] = r2[i]
        else:
            r2_d[key] = r2[(i * C):((i + 1) * C)]
    return r2_d


def extract(out: dict, model: str, Y: np.ndarray, K: np.ndarray, P: np.ndarray,
            ctnu: np.ndarray, fixed_covars: dict, random_covars:dict, 
            K2: Optional[np.ndarray] = None, fixed_shared: bool = True) -> dict:
    """
    Extract REML optimization resutls

    Parameters:
        out:    OptimizationResult from optimize.minimize
        model:  cell type-specific gene expression model, hom/free/full
        Y:  matrix of cell type-specific pseudobulk
        K:  kinship matrix
        P:  matrix of cell type proportions
        ctnu:   cell type-specific noise variance
        fixed_covars:   design matrices for extra fixed effects
        random_covars:  design matrices for extra random effects
        K2: second kinship matrix (only for free model)
        fixed_shared:  whether to use shared fixed effects or not 
    Returns:
        a dict of model parameters and statistics
    """

    N, C = Y.shape
    ngam = C*(C+1) // 2

    x = out['x']
    if model == 'hom':
        hom_g2, hom_e2 = x[:2]
        V = W = np.zeros((C,C))
        ct_overall_g_var = ct_overall_e_var = 0
        r2 = _get_r2(x[2:], random_covars, C)

    elif model == 'free':
        if K2 is None:
            hom_g2 = x[0]
            hom_e2 = x[1]
            V = np.diag( x[2:(2+C)] )
            W = np.diag( x[(2+C):(2+2*C)] )
            r2 = _get_r2(x[(2+2*C):], random_covars, C)
            V2 = np.zeros_like(V)
        else:
            hom_g2 = x[0]
            hom2_g2 = x[1]
            hom_e2 = x[2]
            V = np.diag( x[3:(3+C)] )
            V2 = np.diag( x[(3 + C):(3 + 2 * C)] )
            W = np.diag( x[(3 + 2 * C):(3 + 3 * C)] )
            r2 = _get_r2(x[(3 + 3 * C):], random_covars, C)
        ct_overall_g_var, ct_specific_g_var = util.ct_random_var( V, P )
        ct_overall_g2_var, ct_specific_g2_var = util.ct_random_var( V2, P )
        ct_overall_e_var, ct_specific_e_var = util.ct_random_var( W, P )

    elif model == 'freeW':
        hom_g2 = x[0]
        hom_e2 = x[1]
        W = np.diag( x[2:(2+C)] )
        V = np.zeros((C,C))
        ct_overall_g_var, ct_specific_g_var = util.ct_random_var( V, P )
        ct_overall_e_var, ct_specific_e_var = util.ct_random_var( W, P )
        r2 = _get_r2(x[(2+C):], random_covars, C)

    elif model == 'full':
        hom_g2 = 0
        hom_e2 = 0
        V = np.zeros((C,C))
        V[np.tril_indices(C)] = x[:ngam]
        V = V + V.T
        W = np.zeros((C,C))
        W[np.tril_indices(C)] = x[ngam:(2*ngam)]
        W = W + W.T
        ct_overall_g_var, ct_specific_g_var = util.ct_random_var( V, P )
        ct_overall_e_var, ct_specific_e_var = util.ct_random_var( W, P )
        r2 = _get_r2(x[(2*ngam):], random_covars, C)

    # beta
    y = Y.flatten()
    X = util.get_X(fixed_covars, N, C, fixed_shared=fixed_shared)

    if model == 'free' and K2 is not None:
        Vy = cal_Vy(hom_g2, hom_e2, V, W, r2, K, ctnu, _ZZT(random_covars), 
                    K2, hom2_g2, V2)
    else:
        Vy = cal_Vy( hom_g2, hom_e2, V, W, r2, K, ctnu, _ZZT(random_covars))
    beta = util.glse( Vy, X, y )
    beta, fixed_vars = util.cal_variance( beta, P, fixed_covars, r2, 
                                         random_covars )[:2]

    out = {   'hom_g2':hom_g2, 'hom_e2':hom_e2, 'V':V, 'W':W, 'beta':beta, 
                'ct_overall_g_var':ct_overall_g_var, 
                'ct_overall_e_var':ct_overall_e_var, 
                'fixed_vars':fixed_vars } 
    if model == 'free' and K2 is not None:
        out['hom2_g2'] = hom2_g2
        out['V2'] = V2
        out['ct_overall_g2_var'] = ct_overall_g2_var

    return out


def read_R_out(out_f, y, X, P, K, ctnu, K2, fixed_covars, random_covars
               ) -> Tuple[dict, dict]:
    import rdata
    out = rdata.read_rda(out_f)['out']
    res = {'hom_g2': out['g1'][0], 'hom_e2': out['e'][0],
            'V': out['V1'], 'W': out['W'], 'l': out['l'][0]}
    res['ct_overall_g_var'] = util.ct_random_var(out['V1'], P)[0]
    res['ct_overall_e_var'] = util.ct_random_var(out['W'], P)[0]
    if out['r2'] is None:
        batch = None
    else:
        batch = out['r2'][0]
        res['r2'] = {'batch': batch}
    Vy = cal_Vy(res['hom_g2'], res['hom_e2'], res['V'], res['W'], 
                {'batch': batch}, K, ctnu, _ZZT(random_covars))
    if K2 is not None:
        res['hom2_g2'] = out['g2'][0]
        res['V2'] = out['V2']
        res['ct_overall_g2_var'] = util.ct_random_var(res['V2'], P)[0]
        Vy = cal_Vy(res['hom_g2'], res['hom_e2'], res['V'], res['W'], 
                    {'batch': batch}, K, ctnu, _ZZT(random_covars),
                    K2, res['hom2_g2'], res['V2'])
    beta = util.glse(Vy, X, y)
    beta, fixed_vars = util.cal_variance(beta, P, fixed_covars, {'batch': batch}, 
                                        random_covars)[:2]
    res['beta'] = beta
    res['fixed_vars'] = fixed_vars
    opt = {'success': True if out['convergence'][0] == 0 else False, 
            'status': out['convergence'][0], 'message': out['message'],
            'l': out['l'][0], 'initial': out['par'],
            'niter': out['niter']}
    res['success'] = opt['success']
    return opt, res


def _pMp(X:np.ndarray, X_inv: np.ndarray, A:np.ndarray, B:np.ndarray) -> np.ndarray:
    """
    Compute proj @ np.kron(A,B) @ proj
    """
    M = np.kron(A, B)
    p_M = M - X @ X_inv @ (X.T @ M)
    M = p_M - p_M @ X @ X_inv @ X.T
    return M


def he_ols_dense(Y: np.ndarray, K: np.ndarray, X: np.ndarray, ctnu: np.ndarray,
           model: str, random_covars: dict={}, Kt:Optional[np.ndarray] = None,
           random_shared:bool = True, dtype:Optional[str] = None) -> np.ndarray:
    """
    Perform OLS in HE.

    Reference/fallback implementation: builds an explicit (N*C) x (N*C) matrix per basis
    direction via np.kron. he_ols (the default) is a matrix-free rewrite of this exact
    algorithm that never materializes an (N*C)x(N*C) matrix, and is preferred for actual
    use; this dense version is kept only for cross-checking against he_ols on new inputs.

    Parameters:
        Y:  N * C matrix of Cell Type-specific Pseudobulk
        K:  kinship matrix
        X:  design matrix for fixed effects
        ctnu: cell type-specific noise variance
        model:  free / full
        random_covars:  design matrices for Extra random effects
        Kt: kinship matrix for trans-eQTLs (currently only for Free model)
        random_shared: whether Extra random effects are shared across cell types
        dtype:  data type e.g. float32 to save memory
    Returns:
        OLS estimates
    """

    from scipy import linalg as sla  # deferred: not on HE path
    if dtype is None:
        dtype = 'float64'
    else:
        Y, K, X, ctnu = Y.astype(dtype), K.astype(dtype), X.astype(dtype), ctnu.astype(dtype)
        if Kt is not None:
            Kt = Kt.astype(dtype)

    N, C = Y.shape
    y = Y.flatten()
    X_inv = sla.inv(X.T @ X)
    # n_random = len(random_covars.keys()) if random_shared else len(random_covars.keys()) * C

    # projection matrix
    proj = np.eye(N * C, dtype='int8') - X @ X_inv @ X.T # X: 1_N \otimes I_C append sex \otimes 1_C 

    # vec(M @ A @ M)^T @ vec(M @ B @ M) = vec(M @ A)^T @ vec((M @ B)^T)
    # when A, B, and M are symmetric
    # proj @ y @ y^T @ proj - proj @ D @ proj
    y_p = proj @ y
    ctnu_p = proj * ctnu.flatten()
    t = np.outer( y_p, y_p ) - ctnu_p + ctnu_p @ X @ X_inv @ X.T

    # build Q: list of coefficients
    if model == 'iid':
        Q = []
        # shared
        Q.append(_pMp(X, X_inv, K, np.ones((C,C), dtype='int8')))
        if Kt is not None:
            Q.append(_pMp(X, X_inv, Kt, np.ones((C,C), dtype='int8')))
        Q.append(_pMp(X, X_inv, np.eye(N, dtype='int8'), np.ones((C,C), dtype='int8')))

        # cell type-specific
        Q.append(_pMp(X, X_inv, K, np.eye(C, dtype='int8')))
        if Kt is not None:
            Q.append(_pMp(X, X_inv, Kt, np.eye(C, dtype='int8')))
        Q.append(_pMp(X, X_inv, np.eye(N, dtype='int8'), np.eye(C, dtype='int8')))

        # extra
        for key in sorted(random_covars.keys()):
            Z = random_covars[key]
            ZZT = Z @ Z.T
            if random_shared:
                Q.append(_pMp(X, X_inv, ZZT, np.ones((C,C), dtype='int8')))
            else:
                for c in range(C):
                    L = util.L_f(C, c, c)
                    Q.append(_pMp(X, X_inv, ZZT, L))

        QTQ = np.tensordot(Q, Q, axes=([1, 2], [1, 2]))
        QTt = np.tensordot(Q, t, axes=([1, 2], [0, 1]))

    elif model == 'free':
        Q = []
        # shared
        Q.append(_pMp(X, X_inv, K, np.ones((C,C), dtype='int8')))
        if Kt is not None:
            Q.append(_pMp(X, X_inv, Kt, np.ones((C,C), dtype='int8')))
        Q.append(_pMp(X, X_inv, np.eye(N, dtype='int8'), np.ones((C,C), dtype='int8')))

        # cell type-specific
        # cis-eQTLs
        for c in range(C):
            L = util.L_f(C, c, c)
            Q.append(_pMp(X, X_inv, K, L))
        
        # trans-eQTLs
        if Kt is not None:
            for c in range(C):
                L = util.L_f(C, c, c)
                Q.append(_pMp(X, X_inv, Kt, L))
        
        # environment
        for c in range(C):
            L = util.L_f(C, c, c)
            Q.append(_pMp(X, X_inv, np.eye(N, dtype='int8'), L))
        
        # extra
        for key in sorted(random_covars.keys()):
            Z = random_covars[key]
            ZZT = Z @ Z.T
            if random_shared:
                Q.append(_pMp(X, X_inv, ZZT, np.ones((C,C), dtype='int8')))
            else:
                for c in range(C):
                    L = util.L_f(C, c, c)
                    Q.append(_pMp(X, X_inv, ZZT, L))

        QTQ = np.tensordot(Q, Q, axes=([1, 2], [1, 2]))
        QTt = np.tensordot(Q, t, axes=([1, 2], [0, 1]))

    elif model == 'full':
        Q = []
        for c in range(C):
            L = util.L_f(C, c, c)
            Q.append( _pMp(X, X_inv, K, L) )
        for c in range(C):
            L = util.L_f(C, c, c)
            Q.append( _pMp(X, X_inv, np.eye(N, dtype='int8'), L) )
        for i in range(C - 1):
            for j in range(i + 1, C):
                L = util.L_f(C, i, j) + util.L_f(C, j, i)
                Q.append( _pMp(X, X_inv, K, L) )
        for i in range(C - 1):
            for j in range(i + 1, C):
                L = util.L_f(C, i, j) + util.L_f(C, j, i)
                Q.append( _pMp(X, X_inv, np.eye(N, dtype='int8'), L) )

        for key in sorted(random_covars.keys()):
            Z = random_covars[key]
            ZZT = Z @ Z.T
            if random_shared:
                Q.append( _pMp(X, X_inv, ZZT, np.ones((C, C), dtype='int8')) )
            else:
                for c in range(C):
                    L = util.L_f(C, c, c)
                    Q.append( _pMp(X, X_inv, ZZT, L) )

        QTQ = np.tensordot(Q, Q, axes=([1, 2], [1, 2]))
        QTt = np.tensordot(Q, t, axes=([1, 2], [0, 1]))


    # theta
    try:
        theta = sla.inv(QTQ) @ QTt
        # theta = sla.solve(QTQ, QTt)
        # print(theta)
    except np.linalg.LinAlgError as e:
        print(e)
        print(QTQ)
        print(sla.eigh(QTQ, eigvals_only=True))
        sys.exit(-1)

    if isinstance(Q, np.memmap):
        del Q
        os.remove(tmpfn)

    return theta


def _mf_factor_apply(factor, blocks: np.ndarray) -> np.ndarray:
    """
    factor @ blocks[:,c,:] for each c, where blocks is (N,C,p): 
    Apply an N-side factor to a (N,C,p)-shaped block array along the N axis, without ever
    materializing an (N*C)x(N*C) matrix.

    factor: None (= I_N, no-op), a dense (N,N) ndarray (e.g. K, Kt), or a
            ('lr', R) tuple meaning the low-rank Gram matrix R @ R.T (R: N x k, e.g. an
            extra random effect's design matrix) -- applied as R @ (R.T @ blocks), O(N*k)
            instead of O(N^2).
    """
    if factor is None:
        return blocks
    N, C, p = blocks.shape
    flat = blocks.reshape(N, -1)
    if isinstance(factor, tuple):
        R = factor[1]
        return (R @ (R.T @ flat)).reshape(N, C, p)
    return (factor @ flat).reshape(N, C, p)


def _mf_trace_single(factor, N: int) -> float:
    """tr(factor), handling the None (=I_N) and ('lr', R) (=R @ R.T) cases without forming
    the N x N matrix."""
    if factor is None:
        return float(N)
    if isinstance(factor, tuple):
        R = factor[1]
        return float(np.sum(R * R))
    return float(np.trace(factor))


def _mf_trace_product(Ai, Aj, N: int) -> float:
    """tr(Ai @ Aj) for Ai, Aj each None (=I_N), dense, or ('lr', R) (=R @ R.T)."""
    if Ai is None and Aj is None:
        return float(N)
    if Ai is None:
        return _mf_trace_single(Aj, N)
    if Aj is None:
        return _mf_trace_single(Ai, N)
    if isinstance(Ai, tuple) and isinstance(Aj, tuple):
        Ri, Rj = Ai[1], Aj[1]
        return float(np.sum((Ri.T @ Rj) ** 2))
    if isinstance(Ai, tuple):
        Ri = Ai[1]
        return float(np.sum(Ri * (Aj @ Ri)))
    if isinstance(Aj, tuple):
        return _mf_trace_product(Aj, Ai, N)
    return float(np.sum(Ai * Aj))  # tr(Ai@Aj) via elementwise sum, valid since both symmetric


def _cdpq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    einsum('ncp,ndq->cdpq', A, B) as one BLAS gemm: (C*p, N) @ (N, C*p), with the
    (c,p),(d,q) result unflattened to (C,C,p,q). Same contraction, BLAS summation
    order (HE path regression standard rtol=1e-6/atol=1e-8 applies).
    """
    N, C, p = A.shape
    M = A.reshape(N, C * p).T @ B.reshape(N, C * p)
    return M.reshape(C, p, C, p).transpose(0, 2, 1, 3)


def _gcd_cdpq(F_stack: np.ndarray, Pmat: np.ndarray) -> np.ndarray:
    """
    einsum('gcd,cdpq->gpq', F_stack, Pmat) as one BLAS gemm over the flattened
    (c,d) contraction axis.
    """
    G = F_stack.shape[0]
    p = Pmat.shape[2]
    return (F_stack.reshape(G, -1) @ Pmat.reshape(-1, p * p)).reshape(G, p, p)


def he_ols(Y: np.ndarray, K: np.ndarray, X: np.ndarray, ctnu: np.ndarray,
           model: str, random_covars: dict={}, Kt:Optional[np.ndarray] = None,
           random_shared:bool = True, dtype:Optional[str] = None) -> np.ndarray:
    """
    Perform OLS in HE. Matrix-free: no (N*C) x (N*C) matrix is ever built.

    he_ols_dense's setup builds an explicit (N*C)x(N*C) matrix per basis direction
    (O(N^2*C^2) each, O(n_Q) of them held simultaneously) and its QTQ/QTt assembly costs
    O(n_Q^2*(N*C)^2). This function reduces that to O(n_factors*N^2*C*p) -- where
    n_factors is the small, fixed number of distinct N-side relatedness factors in play
    (K, Kt, I_N, plus one low-rank R per random_covars key), independent of how many
    Sigma-side variance parameters (n_Q, which grows with C) are being fit -- by reducing
    every Frobenius inner product <Q_i, Q_j>_F and <Q_i, t>_F needed for the OLS normal
    equations to a small number of (C,C,p,p)-shaped tensor contractions, built by applying
    each distinct factor to X's C columns once (O(N^2*p) per factor) rather than forming
    kron(factor, F) (O(N^2*C^2)) once per parameter.

    Parameters:
        Y:  N * C matrix of Cell Type-specific Pseudobulk
        K:  kinship matrix
        X:  design matrix for fixed effects
        ctnu: cell type-specific noise variance
        model:  iid / free / full
        random_covars:  design matrices for Extra random effects
        Kt: kinship matrix for trans-eQTLs (currently only for iid/Free model)
        random_shared: whether Extra random effects are shared across cell types
        dtype:  unused
    Returns:
        OLS estimates, in the same parameter order as he_ols_dense
    """

    N, C = Y.shape
    p = X.shape[1]
    y = Y.flatten()
    vs = ctnu

    Ainv = np.linalg.inv(X.T @ X)
    X_blocks = X.reshape(N, C, p)
    y_p = y - X @ (Ainv @ (X.T @ y))
    Y_p = y_p.reshape(N, C)

    # ---- ordered list of (factor, F) Sigma-side basis terms, matching he_ols_dense's Q order ----
    ones_C = np.ones((C, C))
    eye_C = np.eye(C)
    terms = []

    if model == 'iid':
        terms.append((K, ones_C))
        if Kt is not None:
            terms.append((Kt, ones_C))
        terms.append((None, ones_C))
        terms.append((K, eye_C))
        if Kt is not None:
            terms.append((Kt, eye_C))
        terms.append((None, eye_C))
    elif model == 'free':
        terms.append((K, ones_C))
        if Kt is not None:
            terms.append((Kt, ones_C))
        terms.append((None, ones_C))
        for c in range(C):
            terms.append((K, util.L_f(C, c, c)))
        if Kt is not None:
            for c in range(C):
                terms.append((Kt, util.L_f(C, c, c)))
        for c in range(C):
            terms.append((None, util.L_f(C, c, c)))
    elif model == 'full':
        for c in range(C):
            terms.append((K, util.L_f(C, c, c)))
        for c in range(C):
            terms.append((None, util.L_f(C, c, c)))
        for i in range(C - 1):
            for j in range(i + 1, C):
                L = util.L_f(C, i, j) + util.L_f(C, j, i)
                terms.append((K, L))
        for i in range(C - 1):
            for j in range(i + 1, C):
                L = util.L_f(C, i, j) + util.L_f(C, j, i)
                terms.append((None, L))
    else:
        raise ValueError(f'model {model} not supported by matrix-free he_ols')

    for key in sorted(random_covars.keys()):
        Z = random_covars[key]
        lr = ('lr', Z)
        if random_shared:
            terms.append((lr, ones_C))
        else:
            for c in range(C):
                terms.append((lr, util.L_f(C, c, c)))

    n_terms = len(terms)

    # ---- t-target machinery shared across all terms (D-weighted quadratic form) ----
    # R_mat = X^T D X (D := diag(vec(ctnu)))
    R_mat = np.zeros((p, p))
    for c in range(C):
        Xc = X_blocks[:, c, :]
        R_mat += Xc.T @ (vs[:, c][:, None] * Xc)

    # ---- caches: the same factor (e.g. K) is reused across many terms (V_1..V_C all use K) ----
    def _fkey(factor):
        """
        Hashable cache key identifying a factor by object identity, not value: numpy arrays
        aren't hashable so can't be dict keys directly, and id() lets repeated uses of the
        *same* K/Kt/R object hit the cache.
        """
        if factor is None:
            return ('I',)
        if isinstance(factor, tuple):
            return ('lr', id(factor[1]))
        return ('dense', id(factor))

    _ax_cache, _single_pmat_cache, _ay_cache, _diag_cache = {}, {}, {}, {}

    def get_AX(factor):
        """
        Cached $ factor @ X_blocks[:,c,:] $ for every c at once, i.e. kron(factor,I_C)@X as an
        (N,C,p) tensor -- computed once per distinct factor and reused by every term that
        shares it (e.g. all V_c terms share factor=K).
        """
        key = _fkey(factor)
        if key not in _ax_cache:
            _ax_cache[key] = _mf_factor_apply(factor, X_blocks)
        return _ax_cache[key]

    def get_single_pmat(factor):
        """
        Cached (C,C,p,p) tensor Pmat_factor[c,d] = X_blocks[:,c,:].T @ factor @ X_blocks[:,d,:]
        -- Lemma 1's Pmat_M(F), before contracting with any particular F. Contracting this
        against a C,C basis F (via einsum('cd,cdpq->pq', F, Pmat)) gives X^T kron(factor,F) X
        without ever forming kron(factor,F).
        """
        key = _fkey(factor)
        if key not in _single_pmat_cache:
            AX = get_AX(factor)
            _single_pmat_cache[key] = _cdpq(X_blocks, AX)
        return _single_pmat_cache[key]

    def get_AY(factor):
        key = _fkey(factor)
        if key not in _ay_cache:
            if factor is None:
                _ay_cache[key] = Y_p
            elif isinstance(factor, tuple):
                R = factor[1]
                _ay_cache[key] = R @ (R.T @ Y_p)
            else:
                _ay_cache[key] = factor @ Y_p
        return _ay_cache[key]

    def get_diag(factor):
        key = _fkey(factor)
        if key not in _diag_cache:
            if factor is None:
                _diag_cache[key] = np.ones(N)
            elif isinstance(factor, tuple):
                R = factor[1]
                _diag_cache[key] = np.sum(R * R, axis=1)
            else:
                _diag_cache[key] = np.diag(factor)
        return _diag_cache[key]

    _pair_pmat_cache = {}

    def get_pair_pmat(Ai, Aj):
        key = (_fkey(Ai), _fkey(Aj))
        if key not in _pair_pmat_cache:
            ABX = _mf_factor_apply(Aj, X_blocks)
            ABX = _mf_factor_apply(Ai, ABX)
            _pair_pmat_cache[key] = _cdpq(X_blocks, ABX)
        return _pair_pmat_cache[key]

    # ---- grouped/batched assembly of QTQ and QTt ----
    # Terms sharing the same N-side factor are grouped so their small (C,C)-basis
    # contractions run as one batched einsum per factor group instead of one tiny
    # call per term (QTt) or per term pair (QTQ). Math unchanged; only call
    # structure differs.
    term_groups = {}
    for i, (Ai, Fi) in enumerate(terms):
        term_groups.setdefault(_fkey(Ai), []).append(i)

    Si_all = [None] * n_terms
    term_i_all = np.zeros(n_terms)
    term_a_all = np.zeros(n_terms)
    term_b_all = np.zeros(n_terms)

    for k, idxs in term_groups.items():
        Ai = terms[idxs[0]][0]
        F_stack = np.stack([terms[i][1] for i in idxs])              # (G,C,C)

        # Si_all[i] = X^T kron(Ai,Fi) X (Lemma 1's Pmat_M(F) contracted with F)
        Si_g = _gcd_cdpq(F_stack, get_single_pmat(Ai))
        for g, i in enumerate(idxs):
            Si_all[i] = Si_g[g]

        # QTt[i] = <Q(Ai,Fi), y_p y_p^T>_F - <Q(Ai,Fi), proj@D@proj>_F
        # einsum('gcd,nc,nd->g', F, Y_p, AY) == F.reshape(G,-1) @ (Y_p.T @ AY).ravel()
        term_i_all[idxs] = F_stack.reshape(len(idxs), -1) @ (Y_p.T @ get_AY(Ai)).ravel()

        # term_a := tr(D M) (M := kron(Ai,Fi))
        dv = get_diag(Ai) @ vs                                       # (C,)
        diagF_stack = np.stack([np.diag(terms[i][1]) for i in idxs])  # (G,C)
        term_a_all[idxs] = np.sum(diagF_stack * dv[None, :], axis=1)

        # term_b := tr(H D M)
        # einsum('gcd,ndp->gncp', F, AX) as one batched gemm
        temp_b = (get_AX(Ai).transpose(0, 2, 1).reshape(N * p, C)
                  @ F_stack.transpose(0, 2, 1)).reshape(len(idxs), N, p, C).transpose(0, 1, 3, 2)
        # einsum('ncp,gncq->gpq', X, W) == X.reshape(NC,p).T @ W.reshape(G,NC,p)
        S_b = X_blocks.reshape(N * C, p).T @ (vs[None, :, :, None] * temp_b).reshape(len(idxs), N * C, p)
        term_b_all[idxs] = np.trace(Ainv[None, :, :] @ S_b, axis1=1, axis2=2)

    # term_c := tr(Ainv R Ainv Si); Ainv@R_mat@Ainv hoisted (was recomputed per term)
    ARA = Ainv @ R_mat @ Ainv
    Si_stack = np.stack(Si_all)                                      # (n_terms,p,p)
    term_c_all = np.trace(ARA[None, :, :] @ Si_stack, axis1=1, axis2=2)

    QTt = term_i_all - (term_a_all - 2 * term_b_all + term_c_all)

    # T3[i,j] = tr(Ainv Si Ainv Sj) = tr(B_i B_j), B_i = Ainv Si: one batched matmul
    # + one Frobenius contraction for the whole (n_terms, n_terms) matrix.
    B_all = Ainv[None, :, :] @ Si_stack                              # (n_terms,p,p)
    # einsum('ipq,jqp->ij', B, B) == B.reshape(n,-1) @ B.transpose(0,2,1).reshape(n,-1).T
    T3_all = B_all.reshape(n_terms, -1) @ B_all.transpose(0, 2, 1).reshape(n_terms, -1).T

    QTQ = np.zeros((n_terms, n_terms))
    _tp_memo = {}

    # group term pairs (i <= j) by ordered factor pair so S2/T2 run as one batched
    # einsum per factor pair instead of one per pair
    pair_groups = {}
    for i in range(n_terms):
        for j in range(i, n_terms):
            pk = (_fkey(terms[i][0]), _fkey(terms[j][0]))
            pair_groups.setdefault(pk, []).append((i, j))

    for pk, pairs in pair_groups.items():
        Ai, Aj = terms[pairs[0][0]][0], terms[pairs[0][1]][0]

        # T1 = tr(Ai Aj) * tr(Fi Fj); tr(Ai Aj) memoized per factor pair
        if pk not in _tp_memo:
            _tp_memo[pk] = _mf_trace_product(Ai, Aj, N)
        FF_stack = np.stack([terms[i][1] @ terms[j][1] for i, j in pairs])  # (G,C,C)
        T1_g = _tp_memo[pk] * np.trace(FF_stack, axis1=1, axis2=2)

        # T2 = tr(Ainv S2), S2 = <FiFj, Pmat_{Ai,Aj}>_F
        S2_g = _gcd_cdpq(FF_stack, get_pair_pmat(Ai, Aj))
        T2_g = np.trace(Ainv[None, :, :] @ S2_g, axis1=1, axis2=2)

        for g, (i, j) in enumerate(pairs):
            val = T1_g[g] - 2 * T2_g[g] + T3_all[i, j]
            QTQ[i, j] = val
            QTQ[j, i] = val

    try:
        theta = np.linalg.inv(QTQ) @ QTt
    except np.linalg.LinAlgError as e:
        print(e)
        print(QTQ)
        print(np.linalg.eigvalsh(QTQ))
        sys.exit(-1)

    return theta


def _factor_key(factor):
    """Same identity-based keying as he_ols' nested _fkey."""
    if factor is None:
        return ('I',)
    if isinstance(factor, tuple):
        return ('lr', id(factor[1]))
    return ('dense', id(factor))


def he_ols_jk(Y, K, X, ctnu, model, random_covars={}, Kt=None, random_shared=True):
    """Return (N, n_terms) array of leave-one-out refit thetas for he_ols."""
    N, C = Y.shape
    p = X.shape[1]
    y = Y.flatten()
    vs = ctnu
    Xb = X.reshape(N, C, p)

    # ---- terms (identical construction to he_ols / he_ols_jk) ----
    ones_C = np.ones((C, C))
    eye_C = np.eye(C)
    terms = []
    if model == 'iid':
        terms.append((K, ones_C))
        if Kt is not None:
            terms.append((Kt, ones_C))
        terms.append((None, ones_C))
        terms.append((K, eye_C))
        if Kt is not None:
            terms.append((Kt, eye_C))
        terms.append((None, eye_C))
    elif model == 'free':
        terms.append((K, ones_C))
        if Kt is not None:
            terms.append((Kt, ones_C))
        terms.append((None, ones_C))
        for c in range(C):
            terms.append((K, util.L_f(C, c, c)))
        if Kt is not None:
            for c in range(C):
                terms.append((Kt, util.L_f(C, c, c)))
        for c in range(C):
            terms.append((None, util.L_f(C, c, c)))
    elif model == 'full':
        for c in range(C):
            terms.append((K, util.L_f(C, c, c)))
        for c in range(C):
            terms.append((None, util.L_f(C, c, c)))
        for i in range(C - 1):
            for j in range(i + 1, C):
                L = util.L_f(C, i, j) + util.L_f(C, j, i)
                terms.append((K, L))
        for i in range(C - 1):
            for j in range(i + 1, C):
                L = util.L_f(C, i, j) + util.L_f(C, j, i)
                terms.append((None, L))
    else:
        raise ValueError(f'model {model} not supported by matrix-free he_ols')
    for key in sorted(random_covars.keys()):
        Z = random_covars[key]
        lr = ('lr', Z)
        if random_shared:
            terms.append((lr, ones_C))
        else:
            for c in range(C):
                terms.append((lr, util.L_f(C, c, c)))
    n_terms = len(terms)

    # ---- distinct factors, in first-appearance order ----
    fkeys, factors = [], []
    for Ai, _ in terms:
        k = _factor_key(Ai)
        if k not in fkeys:
            fkeys.append(k)
            factors.append(Ai)

    # ---- full-data context (once per gene) ----
    XtX = X.T @ X
    Ainv = np.linalg.inv(XtX)
    XTy = X.T @ y
    R_mat = np.zeros((p, p))
    for c in range(C):
        Xc = Xb[:, c, :]
        R_mat += Xc.T @ (vs[:, c][:, None] * Xc)

    ACOL, AX, AY, DIA, TRA, PMfull, AXD = {}, {}, {}, {}, {}, {}, {}
    for f, k in zip(factors, fkeys):
        AX[k] = _mf_factor_apply(f, Xb)
        PMfull[k] = _cdpq(Xb, AX[k])
        if f is None:
            ACOL[k] = np.eye(N)
            AY[k] = Y
            DIA[k] = np.ones(N)
            TRA[k] = float(N)
        elif isinstance(f, tuple):
            R = f[1]
            ACOL[k] = R @ R.T
            AY[k] = R @ (R.T @ Y)
            DIA[k] = np.sum(R * R, axis=1)
            TRA[k] = float(np.sum(R * R))
        else:
            ACOL[k] = f
            AY[k] = f @ Y
            DIA[k] = np.diag(f)
            TRA[k] = float(np.trace(f))
        # AXD[k][n] = AX[k][n] - A[n,i->n]... per-refit correction factor:
        # (AX[k][i] - A[i,i] X[i]) rows; AXD[k][i] = AX[k][i] - DIA[k][i] * Xb[i]
        AXD[k] = AX[k] - DIA[k][:, None, None] * Xb

    # ---- ordered factor pairs + uniform product-factor caches ----
    pair_keys = []
    for i in range(n_terms):
        for j in range(i, n_terms):
            pk = (_factor_key(terms[i][0]), _factor_key(terms[j][0]))
            if pk not in pair_keys:
                pair_keys.append(pk)
    TRAB, ABII, A2X, BAX, PPfull = {}, {}, {}, {}, {}
    for (ka, kb) in pair_keys:
        fa = factors[fkeys.index(ka)]
        fb = factors[fkeys.index(kb)]
        TRAB[(ka, kb)] = _mf_trace_product(fa, fb, N)
        # (AB)[i,i] and (BA)[i,i] for all i at once (dense factor matrices)
        ABII[(ka, kb)] = np.einsum('ij,ji->i', ACOL[ka], ACOL[kb])
        if (kb, ka) not in ABII:
            ABII[(kb, ka)] = np.einsum('ij,ji->i', ACOL[kb], ACOL[ka])
        if fa is None and fb is None:
            A2X[(ka, kb)] = Xb
            BAX[(ka, kb)] = Xb
        elif fa is None:
            A2X[(ka, kb)] = AX[kb]
            BAX[(ka, kb)] = AX[kb]
        elif fb is None:
            A2X[(ka, kb)] = AX[ka]
            BAX[(ka, kb)] = AX[ka]
        else:
            A2X[(ka, kb)] = _mf_factor_apply(fa, AX[kb])
            BAX[(ka, kb)] = _mf_factor_apply(fb, AX[ka])
        PPfull[(ka, kb)] = _cdpq(Xb, A2X[(ka, kb)])

    # ---- per-gene hoisted assembly structure (data-independent) ----
    term_groups = {}
    for i, (Ai, Fi) in enumerate(terms):
        term_groups.setdefault(_factor_key(Ai), []).append(i)
    group_F = {k: np.stack([terms[i][1] for i in idxs]) for k, idxs in term_groups.items()}
    group_diagF = {k: np.stack([np.diag(terms[i][1]) for i in idxs])
                   for k, idxs in term_groups.items()}
    pair_groups = {}
    for i in range(n_terms):
        for j in range(i, n_terms):
            pk = (_factor_key(terms[i][0]), _factor_key(terms[j][0]))
            pair_groups.setdefault(pk, []).append((i, j))
    group_FF = {}
    group_trFF = {}
    group_S2full = {}
    for pk, pairs in pair_groups.items():
        FF = np.stack([terms[i][1] @ terms[j][1] for i, j in pairs])
        group_FF[pk] = FF
        group_trFF[pk] = np.trace(FF, axis1=1, axis2=2)
        group_S2full[pk] = _gcd_cdpq(FF, PPfull[pk])
    group_Si_full = {k: _gcd_cdpq(group_F[k], PMfull[k]) for k in term_groups}

    # ================= vectorized per-refit computation =================
    ar = np.arange(N)
    # Woodbury downdate of Ainv for all i; residual weights
    T1w = np.einsum('pq,icq->icp', Ainv, Xb)                    # Xi @ Ainv (N,C,p)
    M_all = np.einsum('icp,idp->icd', Xb, T1w)                 # Xi @ Ainv @ Xi.T (N,C,C)
    Minv = np.linalg.inv(eye_C - M_all)                        # (N,C,C)
    Ainv_jk = Ainv[None] + T1w.transpose(0, 2, 1) @ Minv @ T1w  # (N,p,p)
    XTy_jk = XTy[None] - np.einsum('icp,ic->ip', Xb, Y)        # (N,p)
    w_all = np.einsum('ipq,iq->ip', Ainv_jk, XTy_jk)           # (N,p)

    # residuals for all (refit i, kept individual n): Yp_full[i,n,c]
    Xw_all = np.einsum('ncp,ip->inc', Xb, w_all)               # (N,N,C)
    Yp_full = Y[None] - Xw_all
    Yp_ii = Yp_full[ar, ar]                                    # (N,C)
    Xw_ii = Xw_all[ar, ar]                                     # (N,C)

    # R_mat downdate and ARA = Ainv_jk @ R_mat_jk @ Ainv_jk
    Xvs = Xb * vs[:, :, None]                                  # (N,C,p)
    Rm_all = R_mat[None] - np.einsum('icp,icq->ipq', Xvs, Xb)  # (N,p,p)
    ARA = np.einsum('ipq,iqr->ipr', Ainv_jk, Rm_all)
    ARA = np.einsum('ipr,isr->ips', ARA, Ainv_jk)              # (N,p,p)

    Si_terms = np.zeros((N, n_terms, p, p))
    term_i_all = np.zeros((N, n_terms))
    term_a_all = np.zeros((N, n_terms))
    term_b_all = np.zeros((N, n_terms))

    for f, k in zip(factors, fkeys):
        idxs = term_groups[k]
        F = group_F[k]                                          # (G,C,C)
        # Si downdate: Si_jk[i,g] = Si_full[g] - FXA - FAX + DIA[i]*FXX
        t1 = np.einsum('gcd,icp->igdp', F, Xb, optimize=True)
        FXA = np.einsum('igdp,idq->igpq', t1, AX[k], optimize=True)
        t2 = np.einsum('gcd,icp->igdp', F, AX[k], optimize=True)
        FAX = np.einsum('igdp,idq->igpq', t2, Xb, optimize=True)
        FXX = np.einsum('igdp,idq->igpq', t1, Xb, optimize=True)
        Si_terms[:, idxs] = (group_Si_full[k][None] - FXA - FAX
                             + DIA[k][:, None, None, None] * FXX)

        # term_i: YY[i,c,d] = sum_{n!=i} Yp[i,n,c] * AY_jk[i,n,d]
        AXw = np.einsum('ndp,ip->ind', AX[k], w_all, optimize=True)   # (N,N,C)
        AXw_ii = AXw[ar, ar]
        inner = np.einsum('inc,ni->ic', Yp_full, ACOL[k], optimize=True) - Yp_ii * DIA[k][:, None]
        YA = (np.einsum('inc,nd->icd', Yp_full, AY[k], optimize=True)
              - Yp_ii[:, :, None] * AY[k][ar][:, None, :])
        YAXw = (np.einsum('inc,ind->icd', Yp_full, AXw, optimize=True)
                - Yp_ii[:, :, None] * AXw_ii[:, None, :])
        YY = (YA - inner[:, :, None] * Y[:, None, :] - YAXw
              + inner[:, :, None] * Xw_ii[:, None, :])
        term_i_all[:, idxs] = np.einsum('gcd,icd->ig', F, YY, optimize=True)

        # term_a: dv[i,c] = sum_{n!=i} DIA[k][n] vs[n,c]
        dv = (DIA[k] @ vs)[None] - DIA[k][:, None] * vs
        term_a_all[:, idxs] = np.einsum('gc,ic->ig', group_diagF[k], dv, optimize=True)

        # term_b: S_b[i,g,q,r] = sum_{n!=i,c,d} Xvs[n,c,q] F[g,c,d] AX_jk[i,n,d,r]
        v1 = np.einsum('ncq,gcd->gndq', Xvs, F, optimize=True)
        full1 = np.einsum('gndq,ndr->gqr', v1, AX[k], optimize=True)
        w1 = np.einsum('icq,gcd->igdq', Xvs, F, optimize=True)
        m1 = np.einsum('igdq,idr->igqr', w1, AX[k], optimize=True)
        XA = (np.einsum('ncq,ni->icq', Xvs, ACOL[k], optimize=True)
              - Xvs * DIA[k][:, None, None])
        z1 = np.einsum('icq,gcd->igdq', XA, F, optimize=True)
        P2 = np.einsum('igdq,idr->igqr', z1, Xb, optimize=True)
        S_b = full1[None] - m1 - P2
        term_b_all[:, idxs] = np.einsum('iqr,igrq->ig', Ainv_jk, S_b, optimize=True)

    # term_c: tr(ARA[i] @ Si[i,t])
    term_c_all = np.einsum('iqr,itrq->it', ARA, Si_terms, optimize=True)

    QTt = term_i_all - (term_a_all - 2 * term_b_all + term_c_all)

    # T3: B[i,t] = Ainv_jk[i] @ Si[i,t]; T3[t,u] = tr(B_t @ B_u)
    B_all = np.einsum('ipq,itqr->itpr', Ainv_jk, Si_terms, optimize=True)
    T3_all = np.einsum('itpq,iuqp->itu', B_all, B_all, optimize=True)

    # QTQ assembly
    QTQ = np.zeros((N, n_terms, n_terms))
    for pk, pairs in pair_groups.items():
        ka, kb = pk
        FF = group_FF[pk]
        trAB_jk = (TRAB[pk] - ABII[(ka, kb)] - ABII[(kb, ka)]
                   + DIA[ka] * DIA[kb])                        # (N,)
        T1 = trAB_jk[:, None] * group_trFF[pk][None]           # (N,G)
        t1 = np.einsum('gcd,icp->igdp', FF, Xb, optimize=True)
        S2 = (group_S2full[pk][None]
              - np.einsum('igdp,idq->igpq', t1, A2X[pk], optimize=True)
              - np.einsum('igdp,idq->igpq',
                          np.einsum('gcd,icp->igdp', FF, BAX[pk], optimize=True),
                          Xb, optimize=True)
              + ABII[(ka, kb)][:, None, None, None]
              * np.einsum('igdp,idq->igpq', t1, Xb, optimize=True)
              - np.einsum('igdp,idq->igpq',
                          np.einsum('gcd,icp->igdp', FF, AXD[ka], optimize=True),
                          AXD[kb], optimize=True))
        T2 = np.einsum('iqr,igrq->ig', Ainv_jk, S2, optimize=True)
        ii_idx = np.array([pr[0] for pr in pairs])
        jj_idx = np.array([pr[1] for pr in pairs])
        val = T1 - 2 * T2 + T3_all[:, ii_idx, jj_idx]
        QTQ[:, ii_idx, jj_idx] = val
        QTQ[:, jj_idx, ii_idx] = val

    return (np.linalg.inv(QTQ) @ QTt[:, :, None])[:, :, 0]


def _reml(model:str, par: Optional[list], Y: np.ndarray, K: np.ndarray,
          P: np.ndarray, ctnu: np.ndarray, fixed_covars: dict, 
          random_covars: dict, shared: bool, method: Optional[str], nrep: int,
          force_rep: bool=False, K2: Optional[np.ndarray]=None,
          chol: bool=True, R: bool=False, maxit: Optional[int]=None,
          reltol: Optional[float]=None, sd: bool=False) -> dict:
    """
    Wrapper for running REML

    Parameters:
        model:  cell type-specific gene expression model
        par:    initial parameters
        Y:  cell type-specific pseudobulk
        K:  kinship matrix
        P:  cell type proportions
        ctnu:   cell type-specific noise variance
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:   design matrices for Extra random effects
        shared: whether Extra fixed and random effects are shared across cell types
        method: optimization method, e.g. BFGS
        nrep:   number of optimization repeats when initial optimization failed
        force_rep: whether to force repeats even if initial optimization succeeded
        K2:  second kinship matrix # NOTE: only supported in Free model
        chol:   whether to use Cholesky decomposition for optimization
        R:  whether to use R for optimization
        maxit:  maximum number of iterations for optimization
        reltol: relative tolerance for optimization
        sd:  whether to use standard deviation for optimization initialization
    Returns:
        a dict of model parameters and statistics
    """

    N, C = Y.shape
    y = Y.flatten()
    X = util.get_X( fixed_covars, N, C, fixed_shared=shared )

    if R:
        start = time.time()
        log.logger.info('Using R for optimization')
        tmp_dir = tempfile.mkdtemp()
        tmp_f = os.path.join(tmp_dir, '%.10f'%(np.var(y)) + '%.10f'%(np.var(ctnu)))
        y_f = tmp_f + '.y.gz'
        X_f = tmp_f + '.X.gz'
        K_f = tmp_f + '.K.gz'
        ctnu_f = tmp_f + '.ctnu.gz'
        out_f = tmp_f + '.rda'
        np.savetxt(y_f, Y, delimiter='\t')
        np.savetxt(X_f, X, delimiter='\t')
        np.savetxt(K_f, K, delimiter='\t')
        np.savetxt(ctnu_f, ctnu, delimiter='\t')

        cmd = ['Rscript', 'src/cigma/reml.R', '--Y', y_f, '--X', X_f, 
               '--K1', K_f, '--vs', ctnu_f, '--out', out_f]
        if K2 is not None:
            K2_f = tmp_f + '.K2.gz'
            np.savetxt(K2_f, K2, delimiter='\t')
            cmd += ['--K2', K2_f]
        else:
            K2_f = None
        if len(random_covars.keys()) > 0:
            random_covars_f = tmp_f + '.batch'
            np.savetxt(random_covars_f, random_covars['batch'], delimiter='\t')
            cmd += ['--batch', random_covars_f]
        if par is not None:
            par_f = tmp_f + '.par'
            # print(par)
            np.savetxt(par_f, par, delimiter='\t')
            cmd += ['--par', par_f]
        if maxit is not None:
            cmd += ['--maxit', str(maxit)]
        if reltol is not None:
            cmd += ['--reltol', str(reltol)]
        if not chol:
            cmd += ['--noChol']

        util.subprocess_popen(cmd)
        opt, res = read_R_out(out_f, y, X, P, K, ctnu, K2, 
                                   fixed_covars, random_covars)
        opt['time'] = time.time() - start
        log.logger.info(f"loglike: {opt['l']}")

    else:
        funs = {
            'hom':  hom_REML_loglike,
            'freeW':    freeW_REML_loglike,
            'free': free_REML_loglike,
            'full': full_REML_loglike,
        }
        if chol:
            funs['free'] = free_REML_loglike_chol

        loglike_fun = funs[model]
        args = (y, K, X, ctnu, _ZZT(random_covars), shared, K2)  # NOTE: K2 is only supported in Free

        out, opt = util.optim(loglike_fun, par, args, method)
        res = extract(out, model, Y, K, P, ctnu, fixed_covars, random_covars, 
                    K2, fixed_shared=shared)

    vars = {'hom_g2': res['hom_g2'], 'hom_e2': res['hom_e2'], 
            'ct_overall_g_var': res['ct_overall_g_var'], 
            'ct_overall_e_var': res['ct_overall_e_var'],}
    if K2 is not None:
        vars['ct_overall_g2_var'] = res['ct_overall_g2_var']
    if nrep > 0:
        if force_rep or util.check_optim(opt, vars, res['fixed_vars']):
            if R:
                ress = {'rep0': res.copy()}
                opts = {'rep0': opt.copy()}
                rng = np.random.default_rng(int(np.var(y) + ctnu.sum()))
                for i in range(nrep):
                    start = time.time()
                    seed = str(rng.integers(10000))
                    if sd:
                        par_sd = [0.001, 0.005, 0.01, 0.05][i]
                        util.subprocess_popen(cmd + ['--seed', str(seed)] 
                                              + ['--sd', str(par_sd)])
                    else:
                        util.subprocess_popen(cmd + ['--seed', str(seed)])
                    opt_, res_ = read_R_out(out_f, y, X, P, K, ctnu, K2, 
                                            fixed_covars, random_covars)
                    ress[f'rep{i+1}'] = res_.copy()
                    opt_['time'] = time.time() - start
                    opt_['seed'] = seed
                    if sd:
                        opt_['sd'] = par_sd
                    opts[f'rep{i+1}'] = opt_.copy()
                    log.logger.info(f"loglike: {opt_['l']}")
                    if (not opt['success']) and opt_['success']:
                        opt, res = opt_.copy(), res_.copy()
                    elif (opt['success'] == opt_['success']) and (opt['l'] < opt_['l']):
                        opt, res = opt_.copy(), res_.copy()
                if force_rep:
                    res['reps'] = ress
                    opt['reps'] = opts
            else:
                out, opt = util.re_optim(out, opt, loglike_fun, par, args, method, 
                                        nrep)
                res = extract(out, model, Y, K, P, ctnu, fixed_covars, random_covars, 
                            K2, fixed_shared=shared)
    
    if R:
        shutil.rmtree(tmp_dir)
        # os.remove(tmp_f + '.y.gz')
        # os.remove(tmp_f + '.X.gz')
        # os.remove(tmp_f + '.K.gz')
        # os.remove(tmp_f + '.ctnu.gz')
        # os.remove(tmp_f + '.rda')
        # if K2_f is not None:
        #     os.remove(tmp_f + '.K2.gz')
        # if len(random_covars.keys()) > 0:
        #     os.remove(tmp_f + '.batch')
    
    res['opt'] = opt
    #res['out'] = out

    return res


def hom_REML_loglike(par: list, y: np.ndarray, K: np.ndarray, X: np.ndarray,
                     ctnu: np.ndarray, random_covars_ZZT: dict, shared: bool) -> float:
    """
    Loglikelihood for REML under Hom model
    """

    N, C = ctnu.shape
    hom_g2, hom_e2 = par[:2]
    V = np.zeros((C,C))
    W = np.zeros((C,C))
    r2 = {}
    for i, key in enumerate(sorted(random_covars_ZZT.keys())):
        if shared:
            r2[key] = par[2 + i]
        else:
            r2[key] = par[(2 + i * C):(2 + i * (C + 1))]

    l = LL(y, K, X, ctnu, random_covars_ZZT, hom_g2, hom_e2, V, W, r2)
    return l


def hom_REML(Y: np.ndarray, K: np.ndarray, P: np.ndarray, ctnu: np.ndarray,
             fixed_covars: dict={}, random_covars: dict={}, shared: bool=True,
             par: list=None, method: str=None, nrep: int=10) -> dict:
    """
    Fit Hom model with REML

    Parameters:
        Y:  cell type-specific pseudobulk
        K:  kinship matrix
        P:  cell type proportioons
        ctnu:   cell type-specific noise variance
        fixed_covars:   design matrix for Extra fixed effects
        random_covars:   design matrix for Extra random effects
        shared: whether Extra fixed and random effects are shared across cell types
        par:    initial parameters
        method: optimization method, e.g. BFGS
        nrep:   number of optimization repeats when initial optimization failed
    Returns:
        a dict of model parameters and statistics
    """

    from scipy import linalg as sla  # deferred: not on HE path
    log.logger.info('Fitting Hom model with REML')

    N, C = Y.shape
    y = Y.flatten()
    n_random = len(random_covars.keys()) if shared else len(random_covars.keys()) * C
    X = util.get_X( fixed_covars, N, C, fixed_shared=shared )

    if par is None:
        beta = sla.inv( X.T @ X ) @ ( X.T @ y )
        hom_g2 = np.var(y - X @ beta) / (2 + len(random_covars.keys()))
        par = [hom_g2] * (2 + n_random)

    res = _reml( 'hom', par, Y, K, P, ctnu, fixed_covars,
                 random_covars, shared, method, nrep=nrep )

    return res


def freeW_REML_loglike(par:list, y:np.ndarray, K:np.ndarray, X:np.ndarray, ctnu:np.ndarray,
                       random_covars_ZZT: dict, shared: bool) -> float:
    """
    Loglikelihood function for REML under FreeW model, where env is Free and genetic is Hom
    """
    N, C = ctnu.shape
    hom_g2 = par[0]
    hom_e2 = par[1]
    W = np.diag(par[2:(C+2)])
    V = np.zeros((C,C))
    r2 = {}
    for i, key in enumerate(sorted(random_covars_ZZT.keys())):
        if shared:
            r2[key] = par[C+2+i]
        else:
            r2[key] = par[(C + 2 + i * C):(C + 2 + (i+1) * C)]

    l = LL(y, K, X, ctnu, random_covars_ZZT, hom_g2, hom_e2, V, W, r2)
    return l


def freeW_REML(Y:np.ndarray, K:np.ndarray, P:np.ndarray, ctnu:np.ndarray,
               fixed_covars:dict={}, random_covars:dict={}, shared: bool = True,
               par:Optional[list]=None, method:Optional[str]=None, nrep:int=10
               ) -> dict:
    """
    Fit FreeW model using REML, where env is Free and genetic is Hom

    Parameters:
        Y:  cell type-specific pseudobulk
        K:  kinship matrix
        P:  cell type proportions
        ctnu:   cell type-specific noise variance
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:   design matrices for Extra random effects
        shared: whether Extra fixed and random effects are shared across cell types
        par:    initial parameters
        method: optimization method, e.g. BFGS
        nrep:   number of optimization repeats when initial optimization failed
    Returns:
        dict of model parameters and statistics
    """

    from scipy import linalg as sla  # deferred: not on HE path
    log.logger.info('Fitting FreeW model with REML')

    N, C = Y.shape
    y = Y.flatten()
    X = util.get_X( fixed_covars, N, C, fixed_shared=shared )
    n_random = len(random_covars.keys()) if shared else len(random_covars.keys()) * C
    n_par = 2 + 2 * C + X.shape[1] + n_random

    if par is None:
        beta = sla.inv( X.T @ X ) @ ( X.T @ y )
        hom_g2 = np.var(y - X @ beta) / (3 + len(random_covars.keys()))
        par = [hom_g2] * (2 + 2 * C + n_random) # NOTE: stardardize design matrix for extra random effect as needed

    res = _reml('freeW', par, Y, K, P, ctnu, fixed_covars,
                random_covars, shared, method, nrep=nrep)

    return res


def free_REML_loglike(par:list, y:np.ndarray, K:np.ndarray, X:np.ndarray, 
                      ctnu:np.ndarray, random_covars_ZZT: dict, shared: bool, 
                      K2: Optional[np.ndarray]=None) -> float:
    """
    Loglikelihood function for REML under Free model
    """

    N, C = ctnu.shape
    if K2 is None:
        hom_g2 = par[0]
        hom_e2 = par[1]
        V = np.diag(par[2 : (C + 2)])
        W = np.diag(par[(C + 2) : (2 * C + 2)])
        k = 2 * C + 2
        V2 = None
        hom2_g2 = None
    else:
        hom_g2 = par[0]
        hom2_g2 = par[1]
        hom_e2 = par[2]
        V = np.diag(par[3 : (C + 3)])
        V2 = np.diag(par[(C + 3) : (2 * C + 3)])
        W = np.diag(par[(2 * C + 3) : (3 * C + 3)])
        k = 3 * C + 3
    r2 = {}
    for i, key in enumerate(sorted(random_covars_ZZT.keys())):
        # NOTE: can remove the shared here
        if shared:
            r2[key] = par[k + i]
        else:
            r2[key] = par[(k + i * C) : (k + (i + 1) * C)]

    l = LL(y, K, X, ctnu, random_covars_ZZT, hom_g2, hom_e2, V, W, r2, 
        K2, hom2_g2, V2)

    return l


def free_REML_loglike_chol(par:list, y:np.ndarray, K:np.ndarray, X:np.ndarray, 
                      ctnu:np.ndarray, random_covars_ZZT: dict, shared: bool, 
                      K2: Optional[np.ndarray]=None) -> float:
    """
    Loglikelihood function for REML under Free model
    """

    N, C = ctnu.shape
    if K2 is None:
        hom_g2 = par[0]
        hom_e2 = par[1]
        V = np.diag(par[2 : (C + 2)])
        W = np.diag(par[(C + 2) : (2 * C + 2)])
        k = 2 * C + 2
        V2 = None
        hom2_g2 = None
    else:
        hom_g2 = par[0]
        hom2_g2 = par[1]
        hom_e2 = par[2]
        V = np.diag(par[3 : (C + 3)])
        V2 = np.diag(par[(C + 3) : (2 * C + 3)])
        W = np.diag(par[(2 * C + 3) : (3 * C + 3)])
        k = 3 * C + 3
    r2 = {}
    for i, key in enumerate(sorted(random_covars_ZZT.keys())):
        # NOTE: can remove the shared here
        if shared:
            r2[key] = par[k + i]
        else:
            r2[key] = par[(k + i * C) : (k + (i + 1) * C)]

    l = LL_chol(y, K, X, ctnu, random_covars_ZZT, hom_g2, hom_e2, V, W, r2, 
                K2, hom2_g2, V2)

    return l


def free_REML(Y:np.ndarray, K:np.ndarray, P:np.ndarray, ctnu:np.ndarray, 
              fixed_covars:dict={}, random_covars:dict={}, 
              K2: Optional[np.ndarray]=None, shared: bool=True,
              par:Optional[list]=None, method:Optional[str]=None, nrep:int=10, 
              force_rep:bool=False, jk:bool=False, 
              chol:bool=False, R:bool=False, maxit:Optional[int]=None, 
              reltol:Optional[float]=None, sd:bool=False) -> Tuple[dict,dict]:
    '''
    Fit Free model using REML

    Parameters:
        Y:  cell type-specific pseudobulk
        K:  kinship matrix
        P:  cell type proportions
        ctnu:   cell type-specific noise variance
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:   design matrices for Extra random effects
        K2:  second kinship matrix
        shared: whether Extra fixed and random effects are shared across cell types
        par:    initial parameters
        method: optimization method, e.g. BFGS
        nrep:   number of optimization repeats when initial optimization failed
        force_rep: whether to force repeats even if initial optimization succeeded
        jk: jackknife to estimation dispersion matirx
        chol: whether to use cholesky decomposition for optimization
        R: whether to use R for optimization
        maxit: maximum number of iterations for optimization
        reltol: relative tolerance for optimization convergence
        sd: whether to control sd for optimization initialization
    Returns:
        a tuple of 
            #.  dict of model parameters and statistics
            #.  dict of p values
    '''

    log.logger.info('Fitting Free model with REML')

    N, C = Y.shape
    y = Y.flatten()
    X = util.get_X( fixed_covars, N, C, fixed_shared=shared )
    n_random = len(random_covars.keys()) if shared else len(random_covars.keys()) * C
    n_par = 2 + 2 * C + X.shape[1] + n_random


    if not R:
        if par is None:
            # beta = sla.inv( X.T @ X ) @ ( X.T @ y )
            if K2 is None:
                hom_g2 = Y.var(axis=0).mean() / (4 + len(random_covars.keys()))
                par = [hom_g2] * (2 + 2 * C + n_random)
            else:
                hom_g2 = Y.var(axis=0).mean() / (6 + len(random_covars.keys()))
                par = [hom_g2] * (3 + 3 * C + n_random)


    res = _reml('free', par, Y, K, P, ctnu, fixed_covars,
                random_covars, shared, method, nrep=nrep, force_rep=force_rep, 
                maxit=maxit, reltol=reltol, sd=sd,
                K2=K2, chol=chol, R=R)

    if jk:
        jacks = {'ct_beta':[], 'V':[], 'W':[], 'VW':[]}
        for i in range(N):
            if K2 is None:
                Y_jk, K_jk, ctnu_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, K, ctnu, fixed_covars, random_covars, P)
                K2_jk = None
            else:
                Y_jk, K_jk, ctnu_jk, fixed_covars_jk, random_covars_jk, P_jk, K2_jk = util.jk_rmInd(
                    i, Y, K, ctnu, fixed_covars, random_covars, P, K2)

            res_jk = _reml('free', par, Y_jk, K_jk, P_jk, ctnu_jk,
                    fixed_covars_jk, random_covars_jk, shared, method, nrep=nrep,
                    K2=K2_jk, chol=chol, R=R)

            jacks['ct_beta'].append( res_jk['beta']['ct_beta'] )
            jacks['V'].append( np.diag(res_jk['V']) )
            jacks['W'].append( np.diag(res_jk['W']) )
            jacks['VW'].append( np.append( np.diag(res_jk['V']), np.diag(res_jk['W']) ) )

        var_V = (N-1) * np.cov( np.array(jacks['V']).T, bias=True )
        var_W = (N-1) * np.cov( np.array(jacks['W']).T, bias=True )
        var_VW = (N-1) * np.cov( np.array(jacks['VW']).T, bias=True )
        var_ct_beta = (N-1) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        p = {
                'V': wald.mvwald_test(np.diag(res['V']), np.zeros(C), var_V, n=N, P=n_par),
                'W': wald.mvwald_test(np.diag(res['W']), np.zeros(C), var_W, n=N, P=n_par),
                'VW': wald.mvwald_test( np.append(np.diag(res['V']),np.diag(res['W'])), np.zeros(2*C), 
                    var_VW, n=N, P=n_par),
                'ct_beta': util.wald_ct_beta(res['beta']['ct_beta'], var_ct_beta, n=N, P=n_par),
                }
    else:
        p = {}

    return res, p


def full_REML_loglike(par:list, y:np.ndarray, K:np.ndarray, X:np.ndarray, ctnu:np.ndarray,
                      random_covars_ZZT:dict, shared:bool) -> float:
    """
    Loglikelihood function for REML under Full model
    """
    N, C = ctnu.shape
    ngam = C*(C+1) // 2
    hom_g2 = 0
    hom_e2 = 0
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = par[:ngam]
    V = V + V.T
    W = np.zeros((C,C))
    W[np.tril_indices(C)] = par[ngam:(2*ngam)]
    W = W + W.T
    r2 = {}
    for i, key in enumerate(sorted(random_covars_ZZT.keys())):
        if shared:
            r2[key] = par[2*ngam+i]
        else:
            r2[key] = par[(2*ngam+i*C):(2*ngam+(i+1)*C)]

    l = LL(y, K, X, ctnu, random_covars_ZZT, hom_g2, hom_e2, V, W, r2)
    return l


def full_REML(Y:np.ndarray, K:np.ndarray, P:np.ndarray, ctnu:np.ndarray,
              fixed_covars:dict={}, random_covars:dict={}, shared:bool=True,
              par:Optional[list]=None, method:Optional[str]=None, nrep:int=10
              ) -> dict:
    """
    Fit Full model using REML

    Parameters:
        Y:  cell type-specific pseudobulk
        K:  kinship matrix
        P:  cell type proportions
        ctnu:   cell type-specific noise variance
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:   design matrices for Extra random effects
        shared: whether Extra fixed and random effects are shared across cell types
        par:    initial parameters
        method: optimization method, e.g. BFGS
        nrep:   number of optimization repeats when initial optimization failed
    Returns:
        dict of model parameters and statistics
    """

    from scipy import linalg as sla  # deferred: not on HE path
    log.logger.info('Fitting Full model with REML')

    N, C = Y.shape
    ngam = C*(C+1) // 2
    y = Y.flatten()
    X = util.get_X( fixed_covars, N, C, fixed_shared=shared )
    n_random = len(random_covars.keys()) if shared else len(random_covars.keys()) * C

    if par is None:
        beta = sla.inv( X.T @ X ) @ ( X.T @ y )
        hom_g2 = np.var(y - X @ beta) / 2
        V = W = np.diag(np.ones(C))[np.tril_indices(C)] * hom_g2
        par = list(V) + list(W) + [hom_g2] * n_random

    res = _reml('full', par, Y, K, P, ctnu, fixed_covars, random_covars, shared,
                method, nrep=nrep)

    return res


def _iid_he(Y: np.ndarray, K: np.ndarray, Kt: Optional[np.ndarray], 
             ctnu: np.ndarray, P: np.ndarray, fixed_covars: dict={}, 
             random_covars: dict={}, dtype:Optional[str] = None) -> dict:

    N, C = Y.shape
    X = util.get_X(fixed_covars, N, C, fixed_shared=True)

    theta = he_ols(Y, K, X, ctnu, 'iid', random_covars=random_covars, Kt=Kt, 
                   random_shared=True, dtype=dtype)
    
    if Kt is None:
        hom_g2, hom_e2 = theta[0], theta[1]
        V , W = theta[2], theta[3]
        r2 = _get_r2(theta[4:], random_covars, C)

        out = {'hom_g2': hom_g2, 'hom_e2': hom_e2, 'V': V, 'W': W}
        out['shared_h2'], out['specific_h2'] = util.compute_h2_pergene(
            hom_g2, V, hom_e2, W)
        out['specificity'] = hom_g2 / (hom_g2 + V)
        out['r2'] = r2
    else:
        hom_g2, hom_g2_b, hom_e2 = theta[0], theta[1], theta[2]
        V, V_b, W = theta[3], theta[4], theta[5]
        r2 = _get_r2(theta[6:], random_covars, C)

        out = {'hom_g2': hom_g2, 'hom_e2': hom_e2, 'V': V, 'W': W,
               'hom_g2_b': hom_g2_b, 'V_b': V_b}
        out['shared_h2'], out['specific_h2'] = util.compute_h2_pergene(
            hom_g2, V, hom_g2_b + hom_e2, V_b + W)
        out['shared_h2_b'], out['specific_h2_b'] = util.compute_h2_pergene(
            hom_g2_b, V_b, hom_g2 + hom_e2, V + W)
        out['shared_h2_total'], out['specific_h2_total'] = util.compute_h2_pergene(
            hom_g2 + hom_g2_b, V + V_b, hom_e2, W)
        out['specificity'] = hom_g2 / (hom_g2 + V)
        out['specificity_b'] = hom_g2_b / (hom_g2_b + V_b)
        out['r2'] = r2

    return out


def iid_HE(Y: np.ndarray, K: np.ndarray, ctnu: np.ndarray, P: np.ndarray, 
           fixed_covars: dict={}, random_covars: dict={}, 
           Kt:Optional[np.ndarray]=None, jk:bool = False,
           dtype: Optional[str]=None) -> Tuple[dict, dict]:
    """
    Fitting IID model with HE

    Parameters:
        Y:    cell type-specific pseudobulk (no header no index)
        K:    kinship matrix
        ctnu: cell type-specific noise variance (no header no index)
        P:    cell type proportions
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:  design matrices for Extra random covars
        Kt: trans kinship matrix
        jk: perform jackknife
        dtype:  data type for computation in he_ols, e.g. float32

    Returns:
        a tuple of
            #.  dictionary of parameter estimates
            #.  dictionary of p values
    """

    log.logger.info('Fitting IID model with HE')

    N, C = Y.shape 
    # ntril = (C-1) * C // 2
    X = util.get_X(fixed_covars, N, C, fixed_shared=True)
    n_random = len(random_covars.keys())
    n_par = 2 + 2 + X.shape[1] + n_random
    if Kt is not None:
        n_par += 2


    out = _iid_he(Y, K, Kt, ctnu, P, fixed_covars, random_covars, dtype=dtype)
    out['ctnu'] = ctnu.mean(axis=0)
    out['nu'] = ( ctnu * (P ** 2) ).sum(axis=1)

    # jackknife
    if jk:
        log.logger.info('Performing jackknife')
        jacks = {'hom_g2': [], 'V': [], 'hom_e2': [], 'W': [], 
                 'shared_h2': [],'specific_h2': [], 'specificity': [],}
        if Kt is not None:
            jacks['V_b'] = []
            jacks['hom_g2_b'] = []
            jacks['shared_h2_b'] = []
            jacks['specific_h2_b'] = []
            jacks['specificity_b'] = []
            jacks['shared_h2_total'] = []
            jacks['specific_h2_total'] = []
        
        for i in range(N):
            if i % 10 == 0:
                log.logger.info(i)

            if Kt is None:
                Y_jk, K_jk, ctnu_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                                        i, Y, K, ctnu, fixed_covars, random_covars, P=P)

                out_jk = _iid_he(Y_jk, K_jk, None, ctnu_jk, P_jk, fixed_covars_jk,
                               random_covars_jk, dtype=dtype)

            else:
                Y_jk, K_jk, ctnu_jk, fixed_covars_jk, random_covars_jk, P_jk, Kt_jk = util.jk_rmInd(
                        i, Y, K, ctnu, fixed_covars, random_covars, P=P, K2=Kt)

                out_jk = _iid_he(Y_jk, K_jk, Kt_jk, ctnu_jk, P_jk, fixed_covars_jk,
                               random_covars_jk, dtype=dtype)
                
                jacks['hom_g2_b'].append(out_jk['hom_g2_b'])
                jacks['V_b'].append(out_jk['V_b'])
                jacks['shared_h2_b'].append(out_jk['shared_h2_b'])
                jacks['specific_h2_b'].append(out_jk['specific_h2_b'])
                jacks['shared_h2_total'].append(out_jk['shared_h2_total'])
                jacks['specific_h2_total'].append(out_jk['specific_h2_total'])
                jacks['specificity_b'].append(out_jk['specificity_b'])

            jacks['hom_g2'].append(out_jk['hom_g2'])
            jacks['V'].append(out_jk['V'])
            jacks['hom_e2'].append(out_jk['hom_e2'])
            jacks['W'].append(out_jk['W'])
            jacks['shared_h2'].append(out_jk['shared_h2'])
            jacks['specific_h2'].append(out_jk['specific_h2'])
            jacks['specificity'].append(out_jk['specificity'])

        var_hom_g2 = (N - 1) * np.var(jacks['hom_g2'])
        var_V = (N - 1) * np.var(jacks['V'])
        var_hom_e2 = (N - 1) * np.var(jacks['hom_e2'])
        var_W = (N - 1) * np.var(jacks['W'])
        var_shared_h2 = (N - 1) * np.var(jacks['shared_h2'])
        var_specific_h2 = (N - 1) * np.var(jacks['specific_h2'])
        var_specificity = (N - 1) * np.var(jacks['specificity'])

        p = {   
                'hom_g2': wald.wald_test(out['hom_g2'], 0, var_hom_g2, N - n_par),
                'V': wald.wald_test(out['V'], 0, var_V, N - n_par),
                'hom_e2': wald.wald_test(out['hom_e2'], 0, var_hom_e2, N - n_par),
                'W': wald.wald_test(out['W'], 0, var_W, N - n_par),
                'var_hom_g2': var_hom_g2,
                'var_V': var_V,
                'var_hom_e2': var_hom_e2,
                'var_W': var_W,
                'var_shared_h2': var_shared_h2,
                'var_specific_h2': var_specific_h2,
                'var_specificity': var_specificity,
                'jk_hom_g2': np.array(jacks['hom_g2']),  # NOTE: tmp
                'jk_V': np.array(jacks['V']),  # NOTE: tmp
                'jk_hom_e2': np.array(jacks['hom_e2']),  # NOTE: tmp
                'jk_W': np.array(jacks['W']),  # NOTE: tmp
                'jk_shared_h2': np.array(jacks['shared_h2']),  # NOTE: tmp
                'jk_specific_h2': np.array(jacks['specific_h2']),  # NOTE: tmp
                }

        if Kt is not None:
            p['var_hom_g2_b'] = (N - 1) * np.var(jacks['hom_g2_b'])
            p['hom_g2_b'] = wald.wald_test(out['hom_g2_b'], 0, p['var_hom_g2_b'], N - n_par),
            p['var_V_b'] = (N - 1) * np.var(jacks['V_b'])
            p['V_b'] = wald.wald_test(out['V_b'], 0, p['var_V_b'], N - n_par)
            p['var_shared_h2_b'] = (N - 1) * np.var(jacks['shared_h2_b'])
            p['var_specific_h2_b'] = (N - 1) * np.var(jacks['specific_h2_b'])
            p['var_shared_h2_total'] = (N - 1) * np.var(jacks['shared_h2_total'])
            p['var_specific_h2_total'] = (N - 1) * np.var(jacks['specific_h2_total'])
            p['jk_hom_g2_b'] = np.array(jacks['hom_g2_b'])  # NOTE: tmp
            p['jk_V_b'] = np.array(jacks['V_b'])  # NOTE: tmp
            p['jk_shared_h2_b'] = np.array(jacks['shared_h2_b'])  # NOTE: tmp
            p['jk_specific_h2_b'] = np.array(jacks['specific_h2_b'])  # NOTE: tmp
            p['var_specificity_b'] = (N - 1) * np.var(jacks['specificity_b'])

    else:
        p = {}

    # OLS to get beta
    ols_model = _sm().OLS(Y.flatten(), X)
    ols_res = ols_model.fit()
    beta = ols_res.params
    # beta, fixed_vars, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
    out['ct_beta'] = beta[:C]
    # out['ct_beta2'] = (sla.inv(X.T @ X) @ X.T @ Y.flatten())[:C]  # NOTE: to check OLS
    wald_M = []
    for c in range(C-1):
        m = np.zeros(X.shape[1])
        m[c] = 1
        m[C-1] = -1
        wald_M.append(m)

    p['ct_beta'] = ols_res.wald_test(wald_M).pvalue
    
    return out, p


def _free_he(Y: np.ndarray, K: np.ndarray, Kt: Optional[np.ndarray], 
             ctnu: np.ndarray, P: np.ndarray, fixed_covars: dict={}, 
             random_covars: dict={}, fixed_shared: bool=True, 
             random_shared: bool=True, 
             dtype:Optional[str] = None) -> dict:
    
    # NOTE: add specificity

    N, C = Y.shape
    X = util.get_X(fixed_covars, N, C, fixed_shared=fixed_shared)

    theta = he_ols(Y, K, X, ctnu, 'free', random_covars, Kt=Kt, 
                   random_shared=random_shared, dtype=dtype)
    return _free_he_post(theta, C, Kt, random_covars)


def _free_he_post(theta: np.ndarray, C: int, Kt: Optional[np.ndarray],
                  random_covars: dict) -> dict:
    """
    Map a free-model HE theta vector (from he_ols) to the _free_he output
    dict. Split out so the jackknife fast path (he_ols_jk) reuses the
    identical theta -> out mapping without re-running he_ols' setup.
    P is unused by the mapping.
    """
    if Kt is None:
        cis_hom_g2, hom_e2 = theta[0], theta[1]
        cis_V, W = np.diag(theta[2:(C + 2)]), np.diag(theta[(C + 2):(C * 2 + 2)])
        r2 = _get_r2(theta[(C * 2 + 2):], random_covars, C)

        out = {'hom_g2': cis_hom_g2, 'hom_e2': hom_e2, 'V': cis_V, 'W': W}
        out['shared_h2'], out['specific_h2'] = util.compute_h2_pergene(
            cis_hom_g2, cis_V, hom_e2, W)
        out['r2'] = r2
    else:
        hom_g2, hom_g2_b, hom_e2 = theta[0], theta[1], theta[2]
        V, V_b, W = np.diag(theta[3:(C + 3)]), np.diag(theta[(C + 3):(C * 2 + 3)]), np.diag(theta[(C * 2 + 3):(C * 3 + 3)])
        r2 = _get_r2(theta[(C * 3 + 3):], random_covars, C)

        out = {'hom_g2': hom_g2, 'hom_e2': hom_e2, 'V': V, 'W': W,
               'hom_g2_b': hom_g2_b, 'V_b': V_b}
        out['shared_h2'], out['specific_h2'] = util.compute_h2_pergene(
            hom_g2, V, hom_g2_b + hom_e2, V_b + W)
        out['shared_h2_b'], out['specific_h2_b'] = util.compute_h2_pergene(
            hom_g2_b, V_b, hom_g2 + hom_e2, V + W)
        out['shared_h2_total'], out['specific_h2_total'] = util.compute_h2_pergene(
            hom_g2 + hom_g2_b, V + V_b, hom_e2, W)
        out['r2'] = r2

        # ct_overall_gt_var, ct_specific_gt_var = util.ct_random_var(trans_V, P)
        # out['ct_overall_gt_var'] = ct_overall_gt_var
        # out['ct_specific_gt_var'] = ct_specific_gt_var
    # ct specific effect variance
    # cis_ct_overall_g_var, cis_ct_specific_g_var = util.ct_random_var(cis_V, P)
    # ct_overall_e_var, ct_specific_e_var = util.ct_random_var(W, P)

            # 'cis_ct_overall_g_var':cis_ct_overall_g_var, 'cis_ct_specific_g_var':cis_ct_specific_g_var, 
            # 'cis_ct_overall_e_var':cis_ct_overall_e_var, 'cis_ct_specific_e_var':cis_ct_specific_e_var}


    return out


def free_HE(Y: np.ndarray, K: np.ndarray, ctnu: np.ndarray, P: np.ndarray, 
        fixed_covars: dict={}, random_covars: dict={}, 
        Kt:Optional[np.ndarray] = None, fixed_shared:bool=True, 
        random_shared:bool=True, jk:bool = True, prop: float = 1, 
        dtype: Optional[str] = None, verbose: bool = False) -> Tuple[dict, dict]:
    """
    Fitting Free model with HE

    Parameters:
        Y:    cell type-specific pseudobulk (no header no index)
        K:    kinship matrix
        ctnu: cell type-specific noise variance (no header no index)
        P:    cell type proportions
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:  design matrices for Extra random effects
        Kt:   kinship matrix for trans effect
        fixed_shared: whether Extra fixed effects are shared across cell types
        random_shared: whether Extra random effects are shared across cell types
        jk: perform jackknife
        prop: number of replicates for jackknife prop * N
        dtype:  data type for he_ols, e.g. float32
        verbose: whether to output detailed information

    Returns:
        a tuple of
            #.  dictionary of parameter estimates
            #.  dictionary of p values
    """

    log.logger.info('Fitting Free model with HE')

    N, C = Y.shape 
    X = util.get_X(fixed_covars, N, C, fixed_shared=fixed_shared)
    n_random = len(random_covars.keys()) if random_shared else len(random_covars.keys()) * C
    n_par = 2 + 2 * C + X.shape[1] + n_random  # NOTE: when fixed and random are shared
    if Kt is not None:
        n_par += C + 1


    out = _free_he(Y, K, Kt, ctnu, P, fixed_covars, random_covars=random_covars,
                   fixed_shared=fixed_shared, random_shared=random_shared, 
                   dtype=dtype)
    out['ctnu'] = ctnu.mean(axis=0)
    out['nu'] = ( ctnu * (P ** 2) ).sum(axis=1)
    # log.logger.info(out['nu'].dtype)

    # jackknife
    if jk:
        log.logger.info('Jackknife')
        jacks = {'hom_g2': [], 'V': [], 'hom_e2': [], 'W': [], 
                 'shared_h2': [],'specific_h2': [], 'specificity': [],
                 't': []}  # t contains all variance components used for h2
        if Kt is not None:
            jacks['V_b'] = []
            jacks['hom_g2_b'] = []
            jacks['shared_h2_b'] = []
            jacks['specific_h2_b'] = []
            jacks['specificity_b'] = []
            jacks['shared_h2_total'] = []
            jacks['specific_h2_total'] = []
        
        if prop == 1:
            replicates = range(N)
        else:
            replicates = np.linspace(0, N-1, int(prop * N), dtype='int')

        # Fast path: all-N leave-one-out refits of he_ols by exact downdates from one
        # full-data context per gene, replacing N independent jk_rmInd + _free_he
        # refits. Identical normal equations; reassociation-level FP differences only.
        # Requires prop == 1 (every individual left out once) and float64 (dtype None).
        thetas_jk = None
        if prop == 1 and dtype is None:
            thetas_jk = he_ols_jk(Y, K, X, ctnu, 'free', random_covars, Kt=Kt,
                                  random_shared=random_shared)

        for i in replicates:
            if prop != 1 or i % 10 == 0:
                log.logger.info(i)

            if thetas_jk is not None:
                out_jk = _free_he_post(thetas_jk[i], C, Kt, random_covars)
            elif Kt is None:
                Y_jk, K_jk, ctnu_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                                        i, Y, K, ctnu, fixed_covars, random_covars, P=P)

                out_jk = _free_he(Y_jk, K_jk, None, ctnu_jk, P_jk, fixed_covars_jk,
                               random_covars_jk, fixed_shared, random_shared, dtype=dtype)
            else:
                Y_jk, K_jk, ctnu_jk, fixed_covars_jk, random_covars_jk, P_jk, Kt_jk = util.jk_rmInd(
                        i, Y, K, ctnu, fixed_covars, random_covars, P=P, K2=Kt)

                out_jk = _free_he(Y_jk, K_jk, Kt_jk, ctnu_jk, P_jk, fixed_covars_jk,
                               random_covars_jk, fixed_shared, random_shared, dtype=dtype)

            if Kt is None:
                jacks['t'].append([out_jk['hom_g2']] + np.diag(out_jk['V']).tolist()
                                + [out_jk['hom_e2']] + np.diag(out_jk['W']).tolist())
            else:
                jacks['hom_g2_b'].append(out_jk['hom_g2_b'])
                jacks['V_b'].append(np.diag(out_jk['V_b']))
                jacks['shared_h2_b'].append(out_jk['shared_h2_b'])
                jacks['specific_h2_b'].append(out_jk['specific_h2_b'])
                jacks['shared_h2_total'].append(out_jk['shared_h2_total'])
                jacks['specific_h2_total'].append(out_jk['specific_h2_total'])
                out_jk_V_b_bar = np.mean(np.diag(out_jk['V_b']))
                jacks['specificity_b'].append(out_jk_V_b_bar / (out_jk['hom_g2_b'] + out_jk_V_b_bar))  # NOTE: move to _free_he

                jacks['t'].append([out_jk['hom_g2']] + np.diag(out_jk['V']).tolist()
                                + [out_jk['hom_g2_b']] + np.diag(out_jk['V_b']).tolist()
                                + [out_jk['hom_e2']] + np.diag(out_jk['W']).tolist())

            jacks['hom_g2'].append(out_jk['hom_g2'])
            jacks['V'].append(np.diag(out_jk['V']))
            jacks['hom_e2'].append(out_jk['hom_e2'])
            jacks['W'].append(np.diag(out_jk['W']))
            jacks['shared_h2'].append(out_jk['shared_h2'])
            jacks['specific_h2'].append(out_jk['specific_h2'])
            out_jk_V_bar = np.mean(np.diag(out_jk['V']))
            jacks['specificity'].append(out_jk_V_bar / (out_jk['hom_g2'] + out_jk_V_bar))

        var_hom_g2 = (N - 1) * np.var(jacks['hom_g2'])
        var_V = (N - 1) * np.cov(np.array(jacks['V']).T, bias=True)
        var_hom_e2 = (N - 1) * np.var(jacks['hom_e2'])
        var_W = (N - 1) * np.cov(np.array(jacks['W']).T, bias=True)
        var_t = (N - 1) * np.cov(np.array(jacks['t']).T, bias=True)
        var_shared_h2 = (N - 1) * np.var(jacks['shared_h2'])
        var_specific_h2 = (N - 1) * np.var(jacks['specific_h2'])
        var_specificity = (N - 1) * np.var(jacks['specificity'])

        p = {   
                'hom_g2': wald.wald_test(out['hom_g2'], 0, var_hom_g2, N - n_par),
                'V': wald.mvwald_test(np.diag(out['V']), np.zeros(C), var_V, n=N, P=n_par),
                'vc': np.array([wald.wald_test(out['V'][i, i], 0, var_V[i, i], N - n_par) for i in range(C)]), # p for each cell type
                'hom_e2': wald.wald_test(out['hom_e2'], 0, var_hom_e2, N - n_par),
                'W': wald.mvwald_test(np.diag(out['W']), np.zeros(C), var_W, n=N, P=n_par),
                }
        
        if verbose:
            p['var_hom_g2'] = var_hom_g2
            p['var_V'] = var_V
            p['var_hom_e2'] = var_hom_e2
            p['var_W'] = var_W
            p['var_shared_h2'] = var_shared_h2
            p['var_specific_h2'] = var_specific_h2
            p['var_specificity'] = var_specificity
            p['jk_hom_g2'] = np.array(jacks['hom_g2'])
            p['jk_V'] = np.array(jacks['V'])
            p['jk_hom_e2'] = np.array(jacks['hom_e2'])
            p['jk_W'] = np.array(jacks['W'])
            p['jk_shared_h2'] = np.array(jacks['shared_h2'])
            p['jk_specific_h2'] = np.array(jacks['specific_h2'])
            # p['jk_specificity'] = np.array(jacks['specificity'])

        # if Kt is None:
            # h2
            # t = [out['hom_g2']] + np.diag(out['V']).tolist() + [out['hom_e2']] + np.diag(out['W']).tolist()
            # t = np.array(t)
            # formula = [f'~(x1 + x{2 + i})/(x1 + x{2 + i} + x{2 + C} + x{3 + C + i})' for i in range(C)]
            # p['h2'] = util.h2_equal_test(formula, t, var_t, out['h2'])
        # else:
        if Kt is not None:
            p['var_hom_g2_b'] = (N - 1) * np.var(jacks['hom_g2_b'])
            p['hom_g2_b'] = wald.wald_test(out['hom_g2_b'], 0, p['var_hom_g2_b'], N - n_par),
            p['var_V_b'] = (N - 1) * np.cov(np.array(jacks['V_b']).T, bias=True)
            p['V_b'] = wald.mvwald_test(np.diag(out['V_b']), np.zeros(C), 
                                        p['var_V_b'], n=N, P=n_par)
            p['vc_b'] = np.array([wald.wald_test(out['V_b'][i, i], 0, p['var_V_b'][i, i], N - n_par) for i in range(C)]), # p for each cell type
            if verbose:
                p['var_shared_h2_b'] = (N - 1) * np.var(jacks['shared_h2_b'])
                p['var_specific_h2_b'] = (N - 1) * np.var(jacks['specific_h2_b'])
                p['var_shared_h2_total'] = (N - 1) * np.var(jacks['shared_h2_total'])
                p['var_specific_h2_total'] = (N - 1) * np.var(jacks['specific_h2_total'])
                p['jk_hom_g2_b'] = np.array(jacks['hom_g2_b'])  # NOTE: tmp
                p['jk_V_b'] = np.array(jacks['V_b'])  # NOTE: tmp
                p['jk_shared_h2_b'] = np.array(jacks['shared_h2_b'])  # NOTE: tmp
                p['jk_specific_h2_b'] = np.array(jacks['specific_h2_b'])  # NOTE: tmp
                p['var_specificity_b'] = (N - 1) * np.var(jacks['specificity_b'])

            # h2
            # t = [out['hom_g2']] + np.diag(out['V']).tolist() 
            # t += [out['hom_g2_b']] + np.diag(out['V_b']).tolist()
            # t += [out['hom_e2']] + np.diag(out['W']).tolist()
            # t = np.array(t)
            # formula = [f'~(x1 + x{2 + i}) / (x1 + x{2 + i} + x{2 + C} + x{3 + C + i} + x{3 + 2 * C} + x{4 + 2 * C + i})' for i in range(C)]
            # formula_b = [f'~(x{2 + C} + x{3 + C + i}) / (x1 + x{2 + i} + x{2 + C} + x{3 + C + i} + x{3 + 2 * C} + x{4 + 2 * C + i})' for i in range(C)]
            # both_formula = [f'~(x1 + x{2 + i} + x{2 + C} + x{3 + C + i}) / (x1 + x{2 + i} + x{2 + C} + x{3 + C + i} + x{3 + 2 * C} + x{4 + 2 * C + i})' for i in range(C)]
            # p['h2'] = util.h2_equal_test(formula, t, var_t, out['h2'])
            # p['h2_b'] = util.h2_equal_test(formula_b, t, var_t, out['h2_b'])
            # p['h2_t'] = util.h2_equal_test(both_formula, t, var_t, out['h2_t'])
        
    else:
        p = {}

    # OLS to get beta. Replicates statsmodels OLS(method='pinv').fit().params
    # exactly (pinv_extended with rcond=1e-15: SVD of X, reciprocal of singular
    # values above cutoff, vt.T @ (s[:,None] * u.T) @ y) using numpy only, so
    # the ~0.5s statsmodels import (which also pulls in pandas) is avoided on
    # the non-verbose path. Bitwise-identical to the statsmodels result.
    _u, _s, _vt = np.linalg.svd(X, False)
    _cutoff = 1e-15 * np.maximum.reduce(_s)
    for _i in range(min(X.shape[1], _u.shape[0])):
        if _s[_i] > _cutoff:
            _s[_i] = 1.0 / _s[_i]
        else:
            _s[_i] = 0.0
    beta = np.dot(np.dot(np.transpose(_vt), np.multiply(_s[:, np.newaxis], np.transpose(_u))), Y.flatten())
    out['ct_beta'] = beta[:C]
    if verbose:
        ols_res = _sm().OLS(Y.flatten(), X).fit()
        out['beta'] = beta
        if fixed_shared and random_shared:
            beta, fixed_vars, random_vars = util.cal_variance(beta, P, fixed_covars, out['r2'], random_covars)
            out['op_fixed_vars'] = fixed_vars
            out['op_random_vars'] = random_vars
    # wald test for ct_beta
    wald_M = []
    for c in range(C-1):
        m = np.zeros(X.shape[1])
        m[c] = 1
        m[C-1] = -1
        wald_M.append(m)

    if verbose:
        p['ct_beta'] = ols_res.wald_test(wald_M).pvalue
    
    return out, p


def full_HE(Y: np.ndarray, K: np.ndarray, ctnu: np.ndarray, P: np.ndarray, fixed_covars: dict={},
        random_covars: dict={}, shared:bool=True, dtype: Optional[str]=None) -> dict:
    """
    Fitting Full model with HE

    Parameters:
        Y:    cell type-specific pseudobulk (no header no index)
        K:    kinship matrix
        ctnu: cell type-specific noise variance (no header no index)
        P:    cell type proportions
        fixed_covars:   design matrices for Extra fixed effects
        random_covars:  design matrices for Extra random covars
        shared: whether Extra fixed and random effects are shared across cell types
        dtype:  data type for computation in he_ols, e.g. float32

    Returns:
        a dictionary of parameter estimates
    """

    log.logger.info('Fitting Full model with HE')

    N, C = Y.shape 
    ntril = (C-1) * C // 2
    X = util.get_X(fixed_covars, N, C, fixed_shared=shared)

    theta = he_ols(Y, K, X, ctnu, 'full', random_covars=random_covars,
                   random_shared=shared, dtype=dtype)
    log.logger.info(theta.dtype)
    V, W = np.diag(theta[:C]), np.diag(theta[C:(C*2)])
    V[np.triu_indices(C,k=1)] = theta[(C*2):(C*2 + ntril)]
    V = V + V.T - np.diag(theta[:C])
    W[np.triu_indices(C,k=1)] = theta[(C*2 + ntril):(C*2 + ntril*2)]
    W = W + W.T - np.diag(theta[C:(C*2)])

    ct_overall_g_var, ct_specific_g_var = util.ct_random_var( V, P )
    ct_overall_e_var, ct_specific_e_var = util.ct_random_var( W, P )

    he = {'V': V, 'W': W, 'ct_overall_g_var':ct_overall_g_var, 
          'ct_overall_e_var':ct_overall_e_var}
    return he

