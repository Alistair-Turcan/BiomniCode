"""Numba kernels for the dynamical-model fitting hot path.

BITWISE-EXACTNESS CONTRACT: every kernel in this module must reproduce the
exact IEEE-754 float64 results of the NumPy expressions it replaces —
same operations, same order, no fast-math, no reassociation. argmin kernels
replicate np.argmin first-occurrence semantics, including NaN propagation
(NaN counts as the minimum; the first NaN index wins).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def project_argmin(xu, xs, x_obs):
    """Row-wise argmin of squared L2 distances from x_obs rows to (xu, xs).

    Replicates, bitwise:
        ((xt[None, :, :] - x_obs[:, None, :]) ** 2).sum(axis=2).argmin(axis=1)

    with ``xt = np.column_stack([xu, xs])``, for float64 xu/xs of shape (m,)
    and x_obs of shape (n, 2). The sum over the last (size-2) axis is a single
    addition with the first column's term on the left, matching np.sum's
    pairwise reduction for length-2 axes.

    The distance evaluation and the argmin scan are split into two loops over
    a per-row buffer: the branch-free distance loop auto-vectorizes, while the
    scan preserves np.argmin first-occurrence/NaN semantics exactly. Each
    distance value is computed with the identical float64 expression as the
    fused form, so results are unchanged bit-for-bit.
    """
    n = x_obs.shape[0]
    m = xu.shape[0]
    out = np.empty(n, dtype=np.int64)
    dbuf = np.empty(m, dtype=np.float64)
    # A NaN distance requires a NaN operand: du/dv of FINITE values are
    # finite or +-inf, their squares are >= 0 or +inf, and the sum of two
    # non-negative terms is never NaN (but inf - inf = NaN, so the gate must
    # reject non-finite curve/row values, not just NaN). A single cheap curve
    # precheck plus two scalar row checks select between the exact
    # NaN-semantics scan and a 4-lane blocked scan whose lane-combine
    # (strict <, lanes in ascending block order, tail last) reproduces
    # np.argmin first-occurrence exactly.
    curve_nan = False
    for j in range(m):
        curve_nan = (
            curve_nan | (not np.isfinite(xu[j])) | (not np.isfinite(xs[j]))
        )
    L = m // 4
    for i in range(n):
        uo = x_obs[i, 0]
        so = x_obs[i, 1]
        for j in range(m):
            du = xu[j] - uo
            ds = xs[j] - so
            dbuf[j] = du * du + ds * ds
        best = np.inf
        bidx = 0
        if curve_nan or (not np.isfinite(uo)) or (not np.isfinite(so)):
            for j in range(m):
                d = dbuf[j]
                # np.argmin semantics: strict < keeps the first minimum;
                # a NaN d beats a non-NaN best (NaN propagates to the
                # result), but a later NaN does not displace an earlier NaN.
                # Branchless (identical truth table to the if-form) so the
                # scan compiles to conditional moves.
                cond = (d < best) or (np.isnan(d) and not np.isnan(best))
                best = d if cond else best
                bidx = j if cond else bidx
        else:
            # NaN-free fast path: four independent blocked accumulators
            # break the loop-carried cmov chain; identical selection logic.
            b0 = np.inf
            b1 = np.inf
            b2 = np.inf
            b3 = np.inf
            i0 = 0
            i1 = 0
            i2 = 0
            i3 = 0
            for j in range(L):
                d0 = dbuf[j]
                d1 = dbuf[j + L]
                d2 = dbuf[j + 2 * L]
                d3 = dbuf[j + 3 * L]
                c0 = d0 < b0
                b0 = d0 if c0 else b0
                i0 = j if c0 else i0
                c1 = d1 < b1
                b1 = d1 if c1 else b1
                i1 = j + L if c1 else i1
                c2 = d2 < b2
                b2 = d2 if c2 else b2
                i2 = j + 2 * L if c2 else i2
                c3 = d3 < b3
                b3 = d3 if c3 else b3
                i3 = j + 3 * L if c3 else i3
            best = b0
            bidx = i0
            if b1 < best:
                best = b1
                bidx = i1
            if b2 < best:
                best = b2
                bidx = i2
            if b3 < best:
                best = b3
                bidx = i3
            for j in range(4 * L, m):
                d = dbuf[j]
                if d < best:
                    best = d
                    bidx = j
        out[i] = bidx
    return out


# ---------------------------------------------------------------------------
# compute_divergence hot path (mode="assign_timepoints") fused kernels.
#
# These kernels replicate, bitwise, the NumPy expression tree evaluated in
# compute_divergence for the constraint_time_increments + fit_steady_states +
# chi path. Contract per element (all array operands guarded float64 except
# `s`, which may be float32; scalar operands keep their NumPy dtype so numba's
# C-like scalar promotion matches NEP50 array-scalar promotion op-for-op):
#   * np.exp inputs are precomputed in NumPy (identical ufunc results);
#   * x**2 is x*x (both correctly rounded => identical bits);
#   * bool masks multiply as float 1.0/0.0 (never select), preserving
#     signed-zero and NaN propagation;
#   * argmin replicates np.argmin first-occurrence/NaN semantics;
#   * error_model="numpy" so division by zero yields inf/NaN, not exceptions.
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def _cd_curves_dists2(expu, exps, expu_, exps_, u, s, std_u, std_s,
                      ab, ag, c, a0b, a0g, c_, u0_, s0_, out):
    """Fused curve+distance pass for the constraint block.

    Replicates bitwise (per cell j):
        ut  = 0*expu + ab*(1-expu);  st  = 0*exps + ag*(1-exps) + c*(exps-expu)
        ut_ = u0_*expu_ + a0b*(1-expu_); st_ = s0_*exps_ + a0g*(1-exps_) + c_*(exps_-expu_)
        distu  = (u-ut)/std_u  etc.
        out[j,0] = distu_**2 + dists_**2 ; out[j,1] = distu**2 + dists**2
    `out` is the (n,2) C-contiguous twin of res.T in the original code, so
    out.ravel() equals res.T.ravel() byte-for-byte (input to csr_matvecs).
    """
    for j in range(len(u)):
        eu = expu[j]
        es = exps[j]
        ut = 0.0 * eu + ab * (1.0 - eu)
        st = 0.0 * es + ag * (1.0 - es) + c * (es - eu)
        eu_ = expu_[j]
        es_ = exps_[j]
        ut_ = u0_ * eu_ + a0b * (1.0 - eu_)
        st_ = s0_ * es_ + a0g * (1.0 - es_) + c_ * (es_ - eu_)
        du = (u[j] - ut) / std_u
        du_ = (u[j] - ut_) / std_u
        ds = (s[j] - st) / std_s
        ds_ = (s[j] - st_) / std_s
        out[j, 0] = du_ * du_ + ds_ * ds_
        out[j, 1] = du * du + ds * ds


@njit(cache=True, error_model="numpy")
def _cd_curves_dists4(expu, exps, expu_, exps_, u, s, std_u, std_s,
                      ab, ag, c, a0b, a0g, c_, u0_, s0_, out):
    """Fused curve+distance+steady-state pass for the main block.

    Same as _cd_curves_dists2 plus the steady-state distances:
        distx_steady  = ((u-ab)/std_u)**2 + ((s-ag)/std_s)**2
        distx_steady_ = (u/std_u)**2 + (s/std_s)**2
    out columns: [distx_, distx, distx_steady_, distx_steady] (== res rows).
    """
    for j in range(len(u)):
        eu = expu[j]
        es = exps[j]
        ut = 0.0 * eu + ab * (1.0 - eu)
        st = 0.0 * es + ag * (1.0 - es) + c * (es - eu)
        eu_ = expu_[j]
        es_ = exps_[j]
        ut_ = u0_ * eu_ + a0b * (1.0 - eu_)
        st_ = s0_ * es_ + a0g * (1.0 - es_) + c_ * (es_ - eu_)
        du = (u[j] - ut) / std_u
        du_ = (u[j] - ut_) / std_u
        ds = (s[j] - st) / std_s
        ds_ = (s[j] - st_) / std_s
        dx = du * du + ds * ds
        dx_ = du_ * du_ + ds_ * ds_
        ust = (u[j] - ab) / std_u
        sst = (s[j] - ag) / std_s
        dxst = ust * ust + sst * sst
        ust_ = u[j] / std_u
        sst_ = s[j] / std_s
        dxst_ = ust_ * ust_ + sst_ * sst_
        out[j, 0] = dx_
        out[j, 1] = dx
        out[j, 2] = dxst_
        out[j, 3] = dxst


@njit(cache=True)
def _cd_argmin_rows(Y):
    """np.argmin(Y.T, axis=0) for C-contiguous (n,k) Y: per-row argmin with
    first-occurrence/NaN semantics identical to project_argmin's scan."""
    n = Y.shape[0]
    K = Y.shape[1]
    o = np.empty(n, dtype=np.int64)
    for j in range(n):
        best = np.inf
        bidx = 0
        for k in range(K):
            d = Y[j, k]
            cond = (d < best) or (np.isnan(d) and not np.isnan(best))
            best = d if cond else best
            bidx = k if cond else bidx
        o[j] = bidx
    return o


@njit(cache=True, error_model="numpy")
def _cd_assign_final(Y, tau, tau_, t_):
    """Fused argmin + state masking + time assembly for mode=assign_timepoints.

    Replicates bitwise, per cell j (o_j = argmin over the 4 rows of res):
        tau_[j] *= (o_j == 0);  tau[j] *= (o_j == 1)   # pre-remap, float mul
        o_j: 2 -> 1, 3 -> 0                              # post-mask remap
        t[j] = tau[j]*(o_j == 1) + (tau_[j] + t_)*(o_j == 0)
    tau/tau_ are mutated in place exactly like the NumPy in-place ops.
    """
    n = Y.shape[0]
    K = Y.shape[1]
    o = np.empty(n, dtype=np.int64)
    t = np.empty(n, dtype=np.float64)
    for j in range(n):
        best = np.inf
        bidx = 0
        for k in range(K):
            d = Y[j, k]
            cond = (d < best) or (np.isnan(d) and not np.isnan(best))
            best = d if cond else best
            bidx = k if cond else bidx
        tau_[j] = tau_[j] * (1.0 if bidx == 0 else 0.0)
        tau[j] = tau[j] * (1.0 if bidx == 1 else 0.0)
        if bidx == 2:
            bidx = 1
        elif bidx == 3:
            bidx = 0
        o[j] = bidx
        m1 = 1.0 if bidx == 1 else 0.0
        m0 = 1.0 if bidx == 0 else 0.0
        t[j] = tau[j] * m1 + (tau_[j] + t_) * m0
    return t, o


@njit(cache=True, error_model="numpy")
def _gd_curve_diffs(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb,
                    u, s, std_u, std_s, scaling, regc, has_ssr,
                    udiff, sdiff, reg):
    """Fused evaluation tail of BaseDynamics.get_dists (vectorize per-cell
    combos + SplicingDynamics.get_solution(stacked=False) + udiff/sdiff/reg).

    Replicates bitwise, per element i (o_i = int(t_i < t_), computed in the
    driver; expu/exps = np.exp(-beta*tau)/np.exp(-gamma*tau) precomputed in
    NumPy since numba's exp is not bitwise-identical to NumPy's):
        a   = alpha*o_i + 0*(1-o_i)          # vectorize (alpha_=0)
        u0i = 0*o_i + u0_*(1-o_i)            # vectorize (u0=0)
        s0i = 0*o_i + s0_*(1-o_i)            # vectorize (s0=0)
        ut  = u0i*expu + (a/beta)*(1-expu)
        c   = (a - u0i*beta) * inv_gb        # inv_gb = invert(gamma-beta)
        st  = s0i*exps + (a/gamma)*(1-exps) + c*(exps-expu)
        udiff = (ut - u)/std_u*scaling ; sdiff = (st - s)/std_s
        reg   = regc*s/std_s                 # regc = gamma/beta - ssr
    NEP50 emulation notes: numpy result_type(float32 scalar, int64 array) is
    float64, so the vectorize combos are float64 regardless of scalar dtype;
    numba's scalar*int64 promotion agrees (f4 is widened exactly to f8), so
    the combos are written as plain f8 arithmetic on the int64 mask. The int
    `0 * (1 - o_i)` term reproduces the original's -0.0 -> +0.0 sign fix.
    """
    n = expu.shape[0]
    for i in range(n):
        o_i = o[i]
        a = alpha * o_i + 0 * (1 - o_i)
        u0i = 0 * o_i + u0_ * (1 - o_i)
        s0i = 0 * o_i + s0_ * (1 - o_i)
        eu = expu[i]
        es = exps[i]
        ut = u0i * eu + (a / beta) * (1.0 - eu)
        cc = (a - u0i * beta) * inv_gb
        st = s0i * es + (a / gamma) * (1.0 - es) + cc * (es - eu)
        udiff[i] = (ut - u[i]) / std_u * scaling
        sdiff[i] = (st - s[i]) / std_s
        if has_ssr:
            reg[i] = regc * s[i] / std_s
    return udiff, sdiff, reg


@njit(cache=True, error_model="numpy")
def _at_clip_args(u, s, inv_u, c1, c2, c3, c4, c5, arg1, arg2):
    """Clipped log-arguments of the two tau_inv calls in assign_tau's
    else branch (bitwise; same FP ops in the same order as NumPy).

    inv_u (gamma >= beta), from tau_inv(u, s, 0, 0, alpha, beta, gamma) and
    tau_inv(u, s, u0_, s0_, 0, beta, gamma):
        arg1 = clip((u - c1) / c2)   c1 = alpha/beta, c2 = 0 - alpha/beta
        arg2 = clip((u - c3) / c4)   c3 = 0/beta,     c4 = u0_ - 0/beta
    inv_us (gamma < beta), same two calls:
        arg1 = clip((s - c1*u - c2) / c3)   c1 = beta*invert(gamma-beta),
                                            c2 = alpha/gamma - c1*(alpha/beta),
                                            c3 = 0 - c1*0 - c2
        arg2 = clip((s - c1*u - c4) / c5)   c4 = 0/gamma - c1*(0/beta),
                                            c5 = s0_ - c1*u0_ - c4
    clip bounds are clipped_log's defaults: lb + eps = 1e-6,
    ub - eps = 1 - 1e-6. np.minimum/np.maximum reproduce np.clip bitwise
    (incl. NaN propagation). c1*u is computed once per element: both tau_inv
    calls evaluate the identical beta_*u product, so the value is the same.
    """
    n = u.shape[0]
    for i in range(n):
        if inv_u:
            a1 = (u[i] - c1) / c2
            a2 = (u[i] - c3) / c4
        else:
            b = c1 * u[i]
            a1 = (s[i] - b - c2) / c3
            a2 = (s[i] - b - c4) / c5
        arg1[i] = np.minimum(np.maximum(a1, 1e-6), 1.0 - 1e-6)
        arg2[i] = np.minimum(np.maximum(a2, 1e-6), 1.0 - 1e-6)


@njit(cache=True, error_model="numpy")
def _at_scale_clip(log1, log2, m1, m2, t_, s, tau, tau_):
    """Post-log tail of assign_tau's else branch (bitwise):
        tau  = np.clip(m1 * log1, 0, t_)          m1 = -1/beta or -1/gamma
        tau_ = np.clip(m2 * log2, 0, np.max(tau_[s > 0]))
    The masked maximum uses np.maximum semantics starting from -inf, which
    equals np.max on the (nonempty) masked set bitwise, NaN included.
    Returns the number of cells with s > 0; 0 signals the caller to fall
    back to the NumPy path (which raises in np.max on an empty reduction).
    """
    n = log1.shape[0]
    m = -np.inf
    cnt = 0
    for i in range(n):
        tau[i] = np.minimum(np.maximum(m1 * log1[i], 0.0), t_)
        v = m2 * log2[i]
        tau_[i] = v
        if s[i] > 0.0:
            m = np.maximum(m, v)
            cnt += 1
    if cnt > 0:
        for i in range(n):
            tau_[i] = np.minimum(np.maximum(tau_[i], 0.0), m)
    return cnt


@njit(cache=True, error_model="numpy")
def _gdx_curve_distx(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb,
                     u, s, std_u, std_s, scaling, regc, has_ssr, distx):
    """get_distx variant of _gd_curve_diffs (bitwise): identical per-cell
    curve evaluation, then directly
        distx = udiff**2 + sdiff**2 (+ reg**2 if has_ssr)
    with udiff/sdiff/reg computed exactly as in _gd_curve_diffs. The
    regularize=False / ssr=None `distx += 0**2` of the original is a bitwise
    no-op (sum of squares is >= +0.0), so it is skipped. When reg is
    float32, reg**2 is float32 in NumPy and upcast on +=; the kernel keeps
    regc*s/std_s in numba's natural (matching) precision before squaring.
    """
    n = expu.shape[0]
    for i in range(n):
        o_i = o[i]
        a = alpha * o_i + 0 * (1 - o_i)
        u0i = 0 * o_i + u0_ * (1 - o_i)
        s0i = 0 * o_i + s0_ * (1 - o_i)
        eu = expu[i]
        es = exps[i]
        ut = u0i * eu + (a / beta) * (1.0 - eu)
        cc = (a - u0i * beta) * inv_gb
        st = s0i * es + (a / gamma) * (1.0 - es) + cc * (es - eu)
        uu = (ut - u[i]) / std_u * scaling
        ss = (st - s[i]) / std_s
        d = uu * uu + ss * ss
        if has_ssr:
            rr = regc * s[i] / std_s
            d = d + rr * rr
        distx[i] = d
    return distx


@njit(cache=True)
def adjust_outliers(tau, tau_new, tau_ord, dtau, ub):
    """Apply adjust_increments' outlier shifts, bitwise-exactly.

    Replicates:
        idx = np.where(dtau > ub)[0]
        for i in idx:
            tau_new[tau >= tau_ord[i]] -= dtau[i]

    ``np.where`` yields ascending indices and ``tau_ord`` is sorted, so the
    outliers qualifying for an element form a prefix of the outlier list;
    the early break skips exactly the iterations that could not modify the
    element. Per element, subtractions accumulate in ascending-index order,
    matching the original in-place loop's operation sequence bit-for-bit.
    NaN semantics match: NaN > ub is False (no outlier), NaN >= ti is False
    (element untouched).
    """
    m = tau_ord.shape[0]
    n = tau.shape[0]
    idxs = np.empty(m, dtype=np.int64)
    K = 0
    for i in range(m):
        if dtau[i] > ub:
            idxs[K] = i
            K += 1
    # k-outer / j-inner replicates the original loop exactly: each outlier
    # iteration is a full in-place pass over tau_new. For float32 tau_new
    # (dtau is always float64 via np.diff's prepend upcast), the float64
    # subtraction result is rounded to float32 on every store, matching
    # numpy's masked `-=` per-iteration rounding.
    for k in range(K):
        i = idxs[k]
        ti = tau_ord[i]
        dti = dtau[i]
        for j in range(n):
            if tau[j] >= ti:
                tau_new[j] = tau_new[j] - dti
    return tau_new


@njit(cache=True)
def _ad_tmax(T, tj, idx):
    """align_dynamics 'align_total_time' column statistics, bitwise-exact.

    Replicates, per selected column j = idx[jj] (tj_j = t_[j]):
        m1[jj] = np.max(T[:, j] * (T[:, j] < tj_j))
        m2[jj] = np.max((T[:, j] - tj_j) * (T[:, j] > tj_j))
        cnt[jj] = np.sum((T[:, j] == tj_j) | (T[:, j] == 0))
    Max is order-independent except signed-zero ties, which the caller's
    `T_max += T_max == 0` guard erases downstream (both +-0.0 become 1.0);
    NaN propagates through np.maximum exactly as through np.max. The bool
    multiply reproduces NumPy's float*bool -> float*0.0/1.0 (incl. -0.0).
    """
    k = idx.shape[0]
    n = T.shape[0]
    m1 = np.full(k, -np.inf, dtype=np.float64)
    m2 = np.full(k, -np.inf, dtype=np.float64)
    cnt = np.zeros(k, dtype=np.int64)
    # row-major traversal: T is C-order (n_cells, n_vars), so iterating rows
    # outer keeps the gather prefetch-friendly; per-column accumulation order
    # stays i-ascending, identical to the column-major equivalent.
    for i in range(n):
        for jj in range(k):
            j = idx[jj]
            v = T[i, j]
            tjj = tj[j]
            m1[jj] = np.maximum(m1[jj], v * (v < tjj))
            dv = v - tjj
            m2[jj] = np.maximum(m2[jj], dv * (v > tjj))
            if (v == tjj) or (v == 0.0):
                cnt[jj] += 1
    return m1, m2, cnt


@njit(cache=True)
def _ad_scale_cols(T, Tau, Tau_, idx, m_):
    """In-place T[:, idx] *= m_, Tau[:, idx] *= m_, Tau_[:, idx] *= m_
    (bitwise: identical per-element multiplies, no copy)."""
    k = idx.shape[0]
    n = T.shape[0]
    # row-major traversal (T/Tau/Tau_ are C-order): rows outer keeps the
    # strided column gather prefetch-friendly.
    for i in range(n):
        for jj in range(k):
            j = idx[jj]
            mj = m_[jj]
            T[i, j] = T[i, j] * mj
            Tau[i, j] = Tau[i, j] * mj
            Tau_[i, j] = Tau_[i, j] * mj


@njit(cache=True)
def _otau(t, t_):
    """Bitwise-exact fused replica of the two NumPy statements
        o = np.array(t < t_, dtype=int)          # int64 mask
        tau = t * o + (t - t_) * (1 - o)         # f64 elementwise chain
    with a scalar ``t_`` (the hot call sites all pass a Python/NumPy float).
    Each element evaluates the identical expression ``t*oi + (t-t_)*(1-oi)``
    with the same IEEE-754 double operations in the same order (int operands
    promote to float64 exactly as NumPy's int64-array broadcast does), so
    NaN/inf/signed-zero propagation matches the original bit-for-bit while
    the six temporary arrays are eliminated.
    """
    n = t.shape[0]
    o = np.empty(n, dtype=np.int64)
    tau = np.empty(n, dtype=np.float64)
    for i in range(n):
        ti = t[i]
        oi = 1 if ti < t_ else 0
        o[i] = oi
        tau[i] = ti * oi + (ti - t_) * (1 - oi)
    return o, tau


@njit(cache=True)
def _count_on_off(o):
    """(count of o == 1, count of o == 0) in one pass; replaces the
    np.any(on) / np.any(off) branch tests after o == 0 / o == 1 masking."""
    n_on = 0
    n_off = 0
    for i in range(o.shape[0]):
        if o[i] == 1:
            n_on += 1
        elif o[i] == 0:
            n_off += 1
    return n_on, n_off


@njit(cache=True)
def _gather_eq(o, arr, val):
    """arr[o == val] — boolean-mask fancy gather in row order (bitwise:
    pure data movement, same order as NumPy's mask indexing)."""
    n = 0
    for i in range(o.shape[0]):
        if o[i] == val:
            n += 1
    out = np.empty(n, dtype=arr.dtype)
    j = 0
    for i in range(o.shape[0]):
        if o[i] == val:
            out[j] = arr[i]
            j += 1
    return out


@njit(cache=True)
def _scatter_eq(o, arr, sub, val):
    """arr[o == val] = sub — boolean-mask scatter in row order (bitwise:
    pure data movement, same positions as NumPy's mask assignment)."""
    j = 0
    for i in range(o.shape[0]):
        if o[i] == val:
            arr[i] = sub[j]
            j += 1


@njit(cache=True)
def _diff_prepend0_f64(a):
    """np.diff(a, prepend=0) for float64 input, bitwise-exact:
    out[0] = a[0] - 0.0 == a[0] for every double (incl. -0.0, NaN, inf);
    out[i] = a[i] - a[i-1] in order. Only valid for float64 input — the
    NumPy original upcasts float32 via the int64 prepend, so callers gate
    on dtype and keep the NumPy call for non-float64 input."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = a[0]
    for i in range(1, n):
        out[i] = a[i] - a[i - 1]
    return out


# --- AOT kernel overrides (iter18) -------------------------------------
# Precompiled exports of the kernels above (pycc; see
# harness/build_kernels_aot.py). Each wrapper guards every argument
# against the exported signature -- pycc wrappers silently accept
# mismatched scalar dtypes AND mismatched layouts -- and falls back to
# the JIT dispatcher on any deviation, so behavior is identical with
# or without the .so modules present.
try:
    from scvelo.tools import _em_kernels_aot_np as _aot_np
    from scvelo.tools import _em_kernels_aot_py as _aot_py
    _AOT_KERNELS = True
except ImportError:
    _AOT_KERNELS = False

if _AOT_KERNELS:
    _jit_cd_curves_dists2 = _cd_curves_dists2
    def _cd_curves_dists2(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out):
        if (expu.flags.c_contiguous and expu.ndim == 1 and expu.dtype == np.float64 and expu.flags.aligned) and (exps.flags.c_contiguous and exps.ndim == 1 and exps.dtype == np.float64 and exps.flags.aligned) and (expu_.flags.c_contiguous and expu_.ndim == 1 and expu_.dtype == np.float64 and expu_.flags.aligned) and (exps_.flags.c_contiguous and exps_.ndim == 1 and exps_.dtype == np.float64 and exps_.flags.aligned) and (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned) and (s.flags.c_contiguous and s.ndim == 1 and s.dtype == np.float32 and s.flags.aligned) and isinstance(std_u, float) and isinstance(std_s, np.float32) and isinstance(ab, float) and isinstance(c, float) and isinstance(a0b, float) and isinstance(a0g, float) and isinstance(c_, float) and isinstance(u0_, float) and isinstance(s0_, float) and (out.ndim == 2 and out.dtype == np.float64 and out.flags.aligned):
            if isinstance(ag, np.float32):
                try:
                    return _aot_np._cd_curves_dists2__s0(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out)
                except TypeError:
                    pass
            elif isinstance(ag, float):
                try:
                    return _aot_np._cd_curves_dists2__s1(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out)
                except TypeError:
                    pass
        return _jit_cd_curves_dists2(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out)

    _jit_cd_curves_dists4 = _cd_curves_dists4
    def _cd_curves_dists4(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out):
        if (expu.flags.c_contiguous and expu.ndim == 1 and expu.dtype == np.float64 and expu.flags.aligned) and (exps.flags.c_contiguous and exps.ndim == 1 and exps.dtype == np.float64 and exps.flags.aligned) and (expu_.flags.c_contiguous and expu_.ndim == 1 and expu_.dtype == np.float64 and expu_.flags.aligned) and (exps_.flags.c_contiguous and exps_.ndim == 1 and exps_.dtype == np.float64 and exps_.flags.aligned) and (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned) and (s.flags.c_contiguous and s.ndim == 1 and s.dtype == np.float32 and s.flags.aligned) and isinstance(std_u, float) and isinstance(std_s, np.float32) and isinstance(ab, float) and isinstance(c, float) and isinstance(a0b, float) and isinstance(a0g, float) and isinstance(c_, float) and isinstance(u0_, float) and isinstance(s0_, float) and (out.ndim == 2 and out.dtype == np.float64 and out.flags.aligned):
            if isinstance(ag, np.float32):
                try:
                    return _aot_np._cd_curves_dists4__s0(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out)
                except TypeError:
                    pass
            elif isinstance(ag, float):
                try:
                    return _aot_np._cd_curves_dists4__s1(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out)
                except TypeError:
                    pass
        return _jit_cd_curves_dists4(expu, exps, expu_, exps_, u, s, std_u, std_s, ab, ag, c, a0b, a0g, c_, u0_, s0_, out)

    _jit_cd_assign_final = _cd_assign_final
    def _cd_assign_final(Y, tau, tau_, t_):
        if (Y.ndim == 2 and Y.dtype == np.float64 and Y.flags.aligned) and (tau.flags.c_contiguous and tau.ndim == 1 and tau.dtype == np.float64 and tau.flags.aligned) and (tau_.flags.c_contiguous and tau_.ndim == 1 and tau_.dtype == np.float64 and tau_.flags.aligned) and isinstance(t_, float):
            try:
                return _aot_np._cd_assign_final__s0(Y, tau, tau_, t_)
            except TypeError:
                pass
        return _jit_cd_assign_final(Y, tau, tau_, t_)

    _jit_gd_curve_diffs = _gd_curve_diffs
    def _gd_curve_diffs(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, udiff, sdiff, reg):
        if (o.flags.c_contiguous and o.ndim == 1 and o.dtype == np.int64 and o.flags.aligned) and (expu.flags.c_contiguous and expu.ndim == 1 and expu.dtype == np.float64 and expu.flags.aligned) and (exps.flags.c_contiguous and exps.ndim == 1 and exps.dtype == np.float64 and exps.flags.aligned) and isinstance(beta, float) and isinstance(u0_, float) and isinstance(s0_, float) and isinstance(inv_gb, float) and (s.flags.c_contiguous and s.ndim == 1 and s.dtype == np.float32 and s.flags.aligned) and isinstance(std_u, np.float32) and isinstance(std_s, np.float32) and isinstance(regc, float) and (udiff.flags.c_contiguous and udiff.ndim == 1 and udiff.dtype == np.float64 and udiff.flags.aligned) and (sdiff.flags.c_contiguous and sdiff.ndim == 1 and sdiff.dtype == np.float64 and sdiff.flags.aligned) and (reg.flags.c_contiguous and reg.ndim == 1 and reg.dtype == np.float64 and reg.flags.aligned):
            if isinstance(alpha, np.float32) and isinstance(gamma, np.float32) and (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned) and isinstance(scaling, float):
                try:
                    return _aot_np._gd_curve_diffs__s0(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, udiff, sdiff, reg)
                except TypeError:
                    pass
            elif isinstance(alpha, float) and isinstance(gamma, np.float32) and (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned) and isinstance(scaling, float):
                try:
                    return _aot_np._gd_curve_diffs__s1(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, udiff, sdiff, reg)
                except TypeError:
                    pass
            elif isinstance(alpha, float) and isinstance(gamma, float) and (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float32 and u.flags.aligned) and isinstance(scaling, np.float32):
                try:
                    return _aot_np._gd_curve_diffs__s2(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, udiff, sdiff, reg)
                except TypeError:
                    pass
            elif isinstance(alpha, float) and isinstance(gamma, float) and (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned) and isinstance(scaling, float):
                try:
                    return _aot_np._gd_curve_diffs__s3(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, udiff, sdiff, reg)
                except TypeError:
                    pass
        return _jit_gd_curve_diffs(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, udiff, sdiff, reg)

    _jit_at_clip_args = _at_clip_args
    def _at_clip_args(u, s, inv_u, c1, c2, c3, c4, c5, arg1, arg2):
        if (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned) and (s.flags.c_contiguous and s.ndim == 1 and s.dtype == np.float32 and s.flags.aligned) and isinstance(c1, float) and isinstance(c2, float) and isinstance(c3, float) and isinstance(c4, float) and isinstance(c5, float) and (arg1.flags.c_contiguous and arg1.ndim == 1 and arg1.dtype == np.float64 and arg1.flags.aligned) and (arg2.flags.c_contiguous and arg2.ndim == 1 and arg2.dtype == np.float64 and arg2.flags.aligned):
            try:
                return _aot_np._at_clip_args__s0(u, s, inv_u, c1, c2, c3, c4, c5, arg1, arg2)
            except TypeError:
                pass
        return _jit_at_clip_args(u, s, inv_u, c1, c2, c3, c4, c5, arg1, arg2)

    _jit_at_scale_clip = _at_scale_clip
    def _at_scale_clip(log1, log2, m1, m2, t_, s, tau, tau_):
        if (log1.flags.c_contiguous and log1.ndim == 1 and log1.dtype == np.float64 and log1.flags.aligned) and (log2.flags.c_contiguous and log2.ndim == 1 and log2.dtype == np.float64 and log2.flags.aligned) and isinstance(m1, float) and isinstance(m2, float) and isinstance(t_, float) and (s.flags.c_contiguous and s.ndim == 1 and s.dtype == np.float32 and s.flags.aligned) and (tau.flags.c_contiguous and tau.ndim == 1 and tau.dtype == np.float64 and tau.flags.aligned) and (tau_.flags.c_contiguous and tau_.ndim == 1 and tau_.dtype == np.float64 and tau_.flags.aligned):
            try:
                return _aot_np._at_scale_clip__s0(log1, log2, m1, m2, t_, s, tau, tau_)
            except TypeError:
                pass
        return _jit_at_scale_clip(log1, log2, m1, m2, t_, s, tau, tau_)

    _jit_gdx_curve_distx = _gdx_curve_distx
    def _gdx_curve_distx(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, distx):
        if (o.flags.c_contiguous and o.ndim == 1 and o.dtype == np.int64 and o.flags.aligned) and (expu.flags.c_contiguous and expu.ndim == 1 and expu.dtype == np.float64 and expu.flags.aligned) and (exps.flags.c_contiguous and exps.ndim == 1 and exps.dtype == np.float64 and exps.flags.aligned) and isinstance(alpha, float) and isinstance(beta, float) and isinstance(gamma, float) and isinstance(u0_, float) and isinstance(s0_, float) and isinstance(inv_gb, float) and (s.flags.c_contiguous and s.ndim == 1 and s.dtype == np.float32 and s.flags.aligned) and isinstance(std_u, float) and isinstance(std_s, np.float32) and isinstance(scaling, float) and isinstance(regc, float) and (distx.flags.c_contiguous and distx.ndim == 1 and distx.dtype == np.float64 and distx.flags.aligned):
            if (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float32 and u.flags.aligned):
                try:
                    return _aot_np._gdx_curve_distx__s0(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, distx)
                except TypeError:
                    pass
            elif (u.flags.c_contiguous and u.ndim == 1 and u.dtype == np.float64 and u.flags.aligned):
                try:
                    return _aot_np._gdx_curve_distx__s1(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, distx)
                except TypeError:
                    pass
        return _jit_gdx_curve_distx(o, expu, exps, alpha, beta, gamma, u0_, s0_, inv_gb, u, s, std_u, std_s, scaling, regc, has_ssr, distx)

    _jitproject_argmin = project_argmin
    def project_argmin(xu, xs, x_obs):
        if (xu.flags.c_contiguous and xu.ndim == 1 and xu.dtype == np.float64 and xu.flags.aligned) and (xs.flags.c_contiguous and xs.ndim == 1 and xs.dtype == np.float64 and xs.flags.aligned) and (x_obs.flags.f_contiguous and x_obs.ndim == 2 and x_obs.dtype == np.float64 and x_obs.flags.aligned):
            try:
                return _aot_py.project_argmin__s0(xu, xs, x_obs)
            except TypeError:
                pass
        return _jitproject_argmin(xu, xs, x_obs)

    _jit_cd_argmin_rows = _cd_argmin_rows
    def _cd_argmin_rows(Y):
        if (Y.ndim == 2 and Y.dtype == np.float64 and Y.flags.aligned):
            try:
                return _aot_py._cd_argmin_rows__s0(Y)
            except TypeError:
                pass
        return _jit_cd_argmin_rows(Y)

    _jitadjust_outliers = adjust_outliers
    def adjust_outliers(tau, tau_new, tau_ord, dtau, ub):
        if (dtau.flags.c_contiguous and dtau.ndim == 1 and dtau.dtype == np.float64 and dtau.flags.aligned) and isinstance(ub, float):
            if (tau.flags.c_contiguous and tau.ndim == 1 and tau.dtype == np.float32 and tau.flags.aligned) and (tau_new.flags.c_contiguous and tau_new.ndim == 1 and tau_new.dtype == np.float32 and tau_new.flags.aligned) and (tau_ord.flags.c_contiguous and tau_ord.ndim == 1 and tau_ord.dtype == np.float32 and tau_ord.flags.aligned):
                try:
                    return _aot_py.adjust_outliers__s0(tau, tau_new, tau_ord, dtau, ub)
                except TypeError:
                    pass
            elif (tau.flags.c_contiguous and tau.ndim == 1 and tau.dtype == np.float64 and tau.flags.aligned) and (tau_new.flags.c_contiguous and tau_new.ndim == 1 and tau_new.dtype == np.float64 and tau_new.flags.aligned) and (tau_ord.flags.c_contiguous and tau_ord.ndim == 1 and tau_ord.dtype == np.float64 and tau_ord.flags.aligned):
                try:
                    return _aot_py.adjust_outliers__s1(tau, tau_new, tau_ord, dtau, ub)
                except TypeError:
                    pass
        return _jitadjust_outliers(tau, tau_new, tau_ord, dtau, ub)

    _jit_ad_tmax = _ad_tmax
    def _ad_tmax(T, tj, idx):
        if (T.flags.c_contiguous and T.ndim == 2 and T.dtype == np.float64 and T.flags.aligned) and (tj.flags.c_contiguous and tj.ndim == 1 and tj.dtype == np.float64 and tj.flags.aligned) and (idx.flags.c_contiguous and idx.ndim == 1 and idx.dtype == np.int64 and idx.flags.aligned):
            try:
                return _aot_py._ad_tmax__s0(T, tj, idx)
            except TypeError:
                pass
        return _jit_ad_tmax(T, tj, idx)

    _jit_ad_scale_cols = _ad_scale_cols
    def _ad_scale_cols(T, Tau, Tau_, idx, m_):
        if (T.flags.c_contiguous and T.ndim == 2 and T.dtype == np.float64 and T.flags.aligned) and (Tau.flags.c_contiguous and Tau.ndim == 2 and Tau.dtype == np.float64 and Tau.flags.aligned) and (Tau_.flags.c_contiguous and Tau_.ndim == 2 and Tau_.dtype == np.float64 and Tau_.flags.aligned) and (idx.flags.c_contiguous and idx.ndim == 1 and idx.dtype == np.int64 and idx.flags.aligned) and (m_.flags.c_contiguous and m_.ndim == 1 and m_.dtype == np.float64 and m_.flags.aligned):
            try:
                return _aot_py._ad_scale_cols__s0(T, Tau, Tau_, idx, m_)
            except TypeError:
                pass
        return _jit_ad_scale_cols(T, Tau, Tau_, idx, m_)

    _jit_otau = _otau
    def _otau(t, t_):
        if (t.flags.c_contiguous and t.ndim == 1 and t.dtype == np.float64 and t.flags.aligned) and isinstance(t_, float):
            try:
                return _aot_py._otau__s0(t, t_)
            except TypeError:
                pass
        return _jit_otau(t, t_)

    _jit_count_on_off = _count_on_off
    def _count_on_off(o):
        if (o.flags.c_contiguous and o.ndim == 1 and o.dtype == np.int64 and o.flags.aligned):
            try:
                return _aot_py._count_on_off__s0(o)
            except TypeError:
                pass
        return _jit_count_on_off(o)

    _jit_gather_eq = _gather_eq
    def _gather_eq(o, arr, val):
        if (o.flags.c_contiguous and o.ndim == 1 and o.dtype == np.int64 and o.flags.aligned) and (arr.flags.c_contiguous and arr.ndim == 1 and arr.dtype == np.float64 and arr.flags.aligned) and type(val) in (int, np.int64):
            try:
                return _aot_py._gather_eq__s0(o, arr, val)
            except TypeError:
                pass
        return _jit_gather_eq(o, arr, val)

    _jit_scatter_eq = _scatter_eq
    def _scatter_eq(o, arr, sub, val):
        if (o.flags.c_contiguous and o.ndim == 1 and o.dtype == np.int64 and o.flags.aligned) and (arr.flags.c_contiguous and arr.ndim == 1 and arr.dtype == np.float64 and arr.flags.aligned) and (sub.flags.c_contiguous and sub.ndim == 1 and sub.dtype == np.float64 and sub.flags.aligned) and type(val) in (int, np.int64):
            try:
                return _aot_py._scatter_eq__s0(o, arr, sub, val)
            except TypeError:
                pass
        return _jit_scatter_eq(o, arr, sub, val)

    _jit_diff_prepend0_f64 = _diff_prepend0_f64
    def _diff_prepend0_f64(a):
        if (a.flags.c_contiguous and a.ndim == 1 and a.dtype == np.float64 and a.flags.aligned):
            try:
                return _aot_py._diff_prepend0_f64__s0(a)
            except TypeError:
                pass
        return _jit_diff_prepend0_f64(a)
