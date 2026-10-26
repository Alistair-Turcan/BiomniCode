#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

static inline int64_t index_at(const Py_buffer *b, Py_ssize_t i) {
    return b->itemsize == 4 ? ((const int32_t *)b->buf)[i]
                            : ((const int64_t *)b->buf)[i];
}

static int index_format(const Py_buffer *b) {
    return b->format &&
        ((b->itemsize == 4 && strcmp(b->format, "i") == 0) ||
         (b->itemsize == 8 &&
          (strcmp(b->format, "l") == 0 || strcmp(b->format, "q") == 0)));
}

static int doubles(const Py_buffer *b) {
    return b->itemsize == sizeof(double) && b->format &&
           strcmp(b->format, "d") == 0;
}

static PyObject *fill(PyObject *self, PyObject *args) {
    PyObject *objects[5];
    Py_buffer b[5];
    int acquired = 0, ok = 0;
    if (!PyArg_ParseTuple(args, "OOOOO", &objects[0], &objects[1],
                          &objects[2], &objects[3], &objects[4]))
        return NULL;
    for (int k = 0; k < 5; ++k) {
        if (PyObject_GetBuffer(objects[k], &b[k],
                              PyBUF_STRIDES | PyBUF_FORMAT) < 0)
            goto done;
        ++acquired;
        if (!PyBuffer_IsContiguous(&b[k], 'C')) {
            PyErr_SetString(PyExc_ValueError, "contiguous buffers required");
            goto done;
        }
    }
    if (b[0].ndim != 1 || b[1].ndim != 1 || b[2].ndim != 1 ||
        b[3].ndim != 2 || b[4].ndim != 2 || b[4].readonly ||
        !index_format(&b[0]) || !index_format(&b[1]) ||
        !doubles(&b[2]) || !doubles(&b[3]) || !doubles(&b[4]) ||
        b[0].shape[0] < 1 || b[1].shape[0] != b[2].shape[0] ||
        b[4].shape[0] != b[0].shape[0] - 1 ||
        b[4].shape[1] != b[3].shape[1]) {
        PyErr_SetString(PyExc_ValueError, "incompatible CSR buffers");
        goto done;
    }
    for (int k = 0; k < 4; ++k) {
        uintptr_t a = (uintptr_t)b[k].buf, z = (uintptr_t)b[4].buf;
        if (a < z + (uintptr_t)b[4].len &&
            z < a + (uintptr_t)b[k].len) {
            PyErr_SetString(PyExc_ValueError, "output must not alias input");
            goto done;
        }
    }
    for (int k = 0; k < 5; ++k) {
        if ((uintptr_t)b[k].buf % (uintptr_t)b[k].itemsize != 0) {
            PyErr_SetString(PyExc_ValueError, "aligned buffers required");
            goto done;
        }
    }
    Py_ssize_t rows = b[4].shape[0], features = b[4].shape[1];
    int64_t nnz = b[1].shape[0];
    if (index_at(&b[0], 0) != 0 || index_at(&b[0], rows) != nnz) {
        PyErr_SetString(PyExc_ValueError, "invalid row pointers");
        goto done;
    }
    for (Py_ssize_t r = 0; r < rows; ++r) {
        int64_t lo = index_at(&b[0], r), hi = index_at(&b[0], r + 1);
        if (lo < 0 || hi < lo || hi > nnz) {
            PyErr_SetString(PyExc_ValueError, "invalid row pointers");
            goto done;
        }
    }
    for (int64_t e = 0; e < nnz; ++e) {
        int64_t col = index_at(&b[1], e);
        if (col < 0 || col >= b[3].shape[0]) {
            PyErr_SetString(PyExc_ValueError, "invalid column index");
            goto done;
        }
    }
    const double *restrict values = b[2].buf;
    const double *restrict input = b[3].buf;
    double *restrict output = b[4].buf;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < rows; ++r) {
        double *restrict target = output + r * features;
        for (Py_ssize_t j = 0; j < features; ++j)
            target[j] = 0.0;
        int64_t lo = index_at(&b[0], r), hi = index_at(&b[0], r + 1);
        for (int64_t e = lo; e < hi; ++e) {
            const double *restrict source =
                input + index_at(&b[1], e) * features;
            double weight = values[e];
            for (Py_ssize_t j = 0; j < features; ++j)
                target[j] += weight * source[j];
        }
    }
    Py_END_ALLOW_THREADS
    ok = 1;
done:
    while (acquired)
        PyBuffer_Release(&b[--acquired]);
    if (!ok)
        return NULL;
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"fill", fill, METH_VARARGS, "Fill owned float64 output from CSR buffers."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_magic_csr_native", NULL, -1, methods
};
PyMODINIT_FUNC PyInit__magic_csr_native(void) {
    return PyModule_Create(&module);
}
