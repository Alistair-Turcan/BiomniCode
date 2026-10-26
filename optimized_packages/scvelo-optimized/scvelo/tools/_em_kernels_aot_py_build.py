
import numba.types as nt
from numba.pycc import CC
from scvelo.tools import _em_model_kernels as K



cc = CC("_em_kernels_aot_py")

def _p(pair):
    ns = vars(nt)
    args = eval(pair[0], dict(ns))
    ret = eval(pair[1], dict(ns))
    if not isinstance(args, tuple):
        args = (args,)
    return ret(*args)

SIGS = {'project_argmin': [("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 2, 'F', False, aligned=True))", "Array(int64, 1, 'C', False, aligned=True)")], '_cd_argmin_rows': [("(Array(float64, 2, 'A', False, aligned=True),)", "Array(int64, 1, 'C', False, aligned=True)")], 'adjust_outliers': [("(Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64)", "Array(float32, 1, 'C', False, aligned=True)"), ("(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), float64)", "Array(float64, 1, 'C', False, aligned=True)")], '_ad_tmax': [("(Array(float64, 2, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True))", "Tuple((Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True)))")], '_ad_scale_cols': [("(Array(float64, 2, 'C', False, aligned=True), Array(float64, 2, 'C', False, aligned=True), Array(float64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))", 'none')], '_otau': [("(Array(float64, 1, 'C', False, aligned=True), float64)", "Tuple((Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True)))")], '_count_on_off': [("(Array(int64, 1, 'C', False, aligned=True),)", 'UniTuple(int64, 2)')], '_gather_eq': [("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), int64)", "Array(float64, 1, 'C', False, aligned=True)")], '_scatter_eq': [("(Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), int64)", 'none')], '_diff_prepend0_f64': [("(Array(float64, 1, 'C', False, aligned=True),)", "Array(float64, 1, 'C', False, aligned=True)")]}

for _name, _rs in SIGS.items():
    _f = getattr(K, _name).py_func
    for _i, _r in enumerate(_rs):
        cc.export(_name + "__s" + str(_i), _p(_r))(_f)

if __name__ == "__main__":
    cc.compile()
