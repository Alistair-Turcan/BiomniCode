from typing import Tuple, Optional, Union

import sys, math
import numpy as np

# ---------------------------------------------------------------------------
# Vendored scalar kernels replacing scipy.special.{stdtr,chdtrc,fdtrc}.
# Same mathematical functions (regularized incomplete beta / complemented
# incomplete gamma, Cephes-equivalent algorithms), accurate to ~1e-13
# relative -- far inside the authorized rtol=1e-6/atol=1e-8 gate.  Vendoring
# them here removes the ~0.23s scipy.special import from the HE-only path
# (wald.py was the last top-level scipy importer reachable from free_HE).
_MACH_EPS = 2.220446049250313e-16   # DBL_EPSILON (Cephes MACHEP)
_BIG = 4503599627370496.0           # 2**52 (Cephes igamc rescale threshold)
_MAXLOG = 709.782712893384          # ~log(DBL_MAX)
_BETACF_TOL = 3.0e-14
_FPMIN = 1.0e-300


def _betacf(a, b, x):
    """Continued fraction for incbet (Lentz; Numerical Recipes 6.4)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 1001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_TOL:
            break
    return h


def _incbet(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) + lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _igam(a, x):
    """Regularized lower incomplete gamma P(a, x) -- Cephes igam series."""
    if x > 1.0 and x > a:
        return 1.0 - _igamc(a, x)
    ax = a * math.log(x) - x - math.lgamma(a)
    if ax < -_MAXLOG:
        return 0.0
    ax = math.exp(ax)
    r = a
    c = 1.0
    ans = 1.0
    for _ in range(10000):
        r += 1.0
        c *= x / r
        ans += c
        if abs(c) <= abs(ans) * _MACH_EPS:
            break
    return ans * ax / a


def _igamc(a, x):
    """Complemented regularized upper incomplete gamma Q(a, x) -- Cephes."""
    if x < 1.0 or x < a:
        return 1.0 - _igam(a, x)
    ax = a * math.log(x) - x - math.lgamma(a)
    if ax < -_MAXLOG:
        return 0.0
    ax = math.exp(ax)
    y = 1.0 - a
    z = x + y + 1.0
    c = 0.0
    pkm2 = 1.0
    qkm2 = x
    pkm1 = x + 1.0
    qkm1 = z * x
    ans = pkm1 / qkm1
    for _ in range(10000):
        c += 1.0
        y += 1.0
        z += 2.0
        yc = y * c
        pk = pkm1 * z - pkm2 * yc
        qk = qkm1 * z - qkm2 * yc
        if qk != 0.0:
            r = pk / qk
            t = abs((ans - r) / r)
            ans = r
        else:
            t = 1.0
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk
        if abs(pk) > _BIG:
            pkm2 /= _BIG
            pkm1 /= _BIG
            qkm2 /= _BIG
            qkm1 /= _BIG
        if t <= _MACH_EPS:
            break
    return ans * ax


def stdtr(df, t):
    """Student t CDF -- same function as scipy.special.stdtr.

    Two formulations to avoid cancellation: for |t| <= 2 use
    F - 0.5 = +-0.5 * I_w(1/2, df/2) with w = t^2/(df+t^2); in the tails use
    F = 0.5 * I_z(df/2, 1/2) with z = df/(df+t^2) (full relative accuracy
    in the lower tail, which is what wald_test evaluates).
    """
    if t == 0.0:
        return 0.5
    tt = t * t
    if tt <= 4.0:
        w = tt / (df + tt)
        p = 0.5 * _incbet(0.5, 0.5 * df, w)
        return 0.5 + p if t > 0.0 else 0.5 - p
    z = df / (df + tt)
    p = 0.5 * _incbet(0.5 * df, 0.5, z)
    return 1.0 - p if t > 0.0 else p


def chdtrc(df, x):
    """Chi-square survival function -- same function as scipy.special.chdtrc."""
    if x <= 0.0:
        return 1.0
    return _igamc(0.5 * df, 0.5 * x)


def fdtrc(dfn, dfd, x):
    """F survival function -- same function as scipy.special.fdtrc."""
    if x <= 0.0:
        return 1.0
    z = dfd / (dfd + dfn * x)
    return _incbet(0.5 * dfd, 0.5 * dfn, z)


def check_singular(X, eigtol=1e32):
    eval, evec = np.linalg.eig(X)
    if ((max(eval) / (min(eval) + 1e-99)) > eigtol) or ( min(eval) < 0 ):
    # if min(eval) < 0:
        print( eval )
        #print( evec[:, np.argmin(eval)] )
        return True # singular
    else:
        return False

def asymptotic_dispersion_matrix(X, Z, V):
    '''
    Asymptotic dispersion matrix from expected Information matrix (i.e Fisher information)
    Refer to 6.3 Asymptotic dispersion matrix in Variance Components
    X : ndarray
        design matrix for fixed effects
    Z : list
        list of design matrixs for random effects
    V : ndarray
        covariance matrix of y
    D : ndarray
        dispersion matrix
    '''
    if check_singular(V):
        #print(V)
        sys.exit('Singular V in asymptotic dispersion matrix!\n')
    #eval, evec = np.linalg.eig(V)
    #if min(eval) < 0 or (max(eval) / (min(eval) + 1e-99)) > 1e8:
        #sys.exit(f'Singular V in asymptotic_dispersion_matrix: {max(eval)} {min(eval)}\n')
    #V_inv = evec @ np.diag(1/eval) @ evec.T
    V_inv = np.linalg.inv(V)
    # information matrix elements
    minus_E_l_beta_beta = X.T @ V_inv @ X
    #print(V_inv[V_inv < 0 ])
    #print(X.T @ V_inv)
    #print(minus_E_l_beta_beta)
    minus_E_l_sigma_sigma = []
    for i in range(len(Z)):
        tmp = []
        for j in range(len(Z)):
            tmp.append(np.trace(V_inv @ Z[i] @ Z[i].T @ V_inv @ Z[j] @ Z[j].T) / 2)
        minus_E_l_sigma_sigma.append(tmp)
    minus_E_l_sigma_sigma = np.array(minus_E_l_sigma_sigma)

    # asymptotic dispersion matrix
    if check_singular(minus_E_l_beta_beta):
        print(minus_E_l_beta_beta)
        sys.exit('Singular minus_E_l_beta_beta!\n')
    minus_E_l_beta_beta_inv = np.linalg.inv(minus_E_l_beta_beta)
    #print(minus_E_l_beta_beta)
    #print(minus_E_l_beta_beta_inv)
    #if not np.allclose( minus_E_l_beta_beta @ minus_E_l_beta_beta_inv, np.identity(minus_E_l_beta_beta.shape[0]) ):
    #    print( minus_E_l_beta_beta @ minus_E_l_beta_beta_inv )
    #    sys.exit('Singular matrix?\n')
    #print( np.linalg.eigvals(minus_E_l_beta_beta) )
    #print(minus_E_l_beta_beta @ minus_E_l_beta_beta_inv @ minus_E_l_beta_beta)
    #print(minus_E_l_beta_beta @ np.linalg.pinv(minus_E_l_beta_beta) @ minus_E_l_beta_beta)
    if check_singular( minus_E_l_sigma_sigma ):
        print(minus_E_l_sigma_sigma)
        sys.exit('Singular minus_E_l_sigma_sigma!\n')
    minus_E_l_sigma_sigma_inv = np.linalg.inv(minus_E_l_sigma_sigma)
    D = np.block([
        [minus_E_l_beta_beta_inv, np.zeros((minus_E_l_beta_beta_inv.shape[0], minus_E_l_sigma_sigma_inv.shape[1]))],
        [np.zeros((minus_E_l_sigma_sigma_inv.shape[0], minus_E_l_beta_beta_inv.shape[1])), minus_E_l_sigma_sigma_inv]
        ])

    #print('Asymptotic dispersion matrix:\n', D)
    return D

def reml_asymptotic_dispersion_matrix(X, Z, V):
    '''
    Asymptotic dispersion matrix for REML from expected Information matrix (i.e Fisher information)
    Refer to 6.6 Restricted maximum likelihood
    X : ndarray
        design matrix for fixed effects
    Z : list
        list of design matrixs for random effects
    V : ndarray
        covariance matrix of y
    D : ndarray
        dispersion matrix
    '''
    if check_singular(V):
        print(V)
        sys.exit('Singular V in REML asymptotic dispersion matrix!\n')
    V_inv = np.linalg.inv(V)

    A = X.T @ V_inv @ X
    if check_singular(A):
        print(A)
        sys.exit('Singular A!\n')

    P = V_inv - V_inv @ X @ np.linalg.inv(A) @ X.T @ V_inv 
    # information matrix elements
    I = []
    for i in range(len(Z)):
        tmp = []
        for j in range(len(Z)):
            sesq = np.sum( (Z[i].T @  P @ Z[j])**2 )
            tmp.append(sesq)
        I.append(tmp)
    I = np.array(I)

    # asymptotic dispersion matrix
    if check_singular(I):
        print(I)
        sys.exit('Singular I!\n')
    D = 2 * np.linalg.inv(I)

    return D

def wald_test(estimate, expectation, var, df, two_sided=False):
    '''
    estimate : scalar
        ML estimate of parameter
    expectation : scalar
        expectation of parameter
    var : scalar
        variance of parameter normal distribution
    df : scalar
        degree of freedom
    two_sided : boolen
        two-sided or one-sided test (estiamte > expectation)
    '''
    if var <= 0:
        sys.exit(f'Positive variance expected, but get {var}\n')
    W = (estimate-expectation) / math.sqrt(var)

    if two_sided:
        return 2 * stdtr(df, -abs(W))
    else:
        return stdtr(df, -W)
    

def mvwald_test(estimate: np.ndarray, expectation: np.ndarray, var: np.ndarray, 
                eigtol: float=1e32, Ftest: bool=True, 
                n: int=None, P: int=None, df: int=None) -> float:
    """
    estimate:   estimates of parameters
    expectation:    expectation of parameters
    var:    variance-covariance matrix of parameter normal distribution
    Ftest:  Ftest or chi squire test (refer to Wiki Wald test - test on multiple parameters)
    n:  sample size
    P:  number of estimated parameters
    """
    
    # chi square test
    if check_singular(var, eigtol):
        print(var)
        sys.exit('Singular matrix in mvwald!\n')
        #return float('nan')
    W2 = (estimate-expectation) @ np.linalg.inv(var) @ (estimate-expectation)
    p = chdtrc(len(estimate), W2)
    
    # F test
    if Ftest:
        F = W2/len(estimate)
        if not df:
            df = n-P
        p = fdtrc(len(estimate), df, F)

    return p

if __name__ == '__main__':
    pass
