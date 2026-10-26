
import numba.types as nt
from numba.pycc import CC
from scvelo.tools import _em_model_kernels as K


import numba.pycc.compiler as _pyccc
from numba.core.compiler import Flags as _Flags

def _np_flags():
    f = _Flags()
    f.error_model = "numpy"
    return f

_pyccc.Flags = _np_flags


cc = CC("_em_kernels_aot_np")

def _p(pair):
    ns = vars(nt)
    args = eval(pair[0], dict(ns))
    ret = eval(pair[1], dict(ns))
    if not isinstance(args, tuple):
        args = (args,)
    return ret(*args)

SIGS = {'_cd_curves_dists2': [("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float64, float32, float64, float32, float64, float64, float64, float64, float64, float64, Array(float64, 2, 'A', False, aligned=True))", 'none'), ("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float64, float32, float64, float64, float64, float64, float64, float64, float64, float64, Array(float64, 2, 'A', False, aligned=True))", 'none')], '_cd_curves_dists4': [("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float64, float32, float64, float32, float64, float64, float64, float64, float64, float64, Array(float64, 2, 'A', False, aligned=True))", 'none'), ("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float64, float32, float64, float64, float64, float64, float64, float64, float64, float64, Array(float64, 2, 'A', False, aligned=True))", 'none')], '_cd_assign_final': [("(Array(float64, 2, 'A', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64)", "Tuple((Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True)))")], '_gd_curve_diffs': [("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float32, float64, float32, float64, float64, float64, Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float32, float32, float64, float64, bool, Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", "UniTuple(Array(float64, 1, 'C', False, aligned=True), 3)"), ("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64, float64, float32, float64, float64, float64, Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float32, float32, float64, float64, bool, Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", "UniTuple(Array(float64, 1, 'C', False, aligned=True), 3)"), ("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64, float64, float64, float64, float64, float64, Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float32, float32, float32, float64, bool, Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", "UniTuple(Array(float64, 1, 'C', False, aligned=True), 3)"), ("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64, float64, float64, float64, float64, float64, Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float32, float32, float64, float64, bool, Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", "UniTuple(Array(float64, 1, 'C', False, aligned=True), 3)")], '_at_clip_args': [("(Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), bool, float64, float64, float64, float64, float64, Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", 'none')], '_at_scale_clip': [("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64, float64, float64, Array(float32, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", 'int64')], '_gdx_curve_distx': [("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64, float64, float64, float64, float64, float64, Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float64, float32, float64, float64, bool, Array(float64, 1, 'C', False, aligned=True))", "Array(float64, 1, 'C', False, aligned=True)"), ("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64, float64, float64, float64, float64, float64, Array(float64, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), float64, float32, float64, float64, bool, Array(float64, 1, 'C', False, aligned=True))", "Array(float64, 1, 'C', False, aligned=True)")]}

for _name, _rs in SIGS.items():
    _f = getattr(K, _name).py_func
    for _i, _r in enumerate(_rs):
        cc.export(_name + "__s" + str(_i), _p(_r))(_f)

if __name__ == "__main__":
    cc.compile()
