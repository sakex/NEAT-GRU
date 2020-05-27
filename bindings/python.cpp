//
// Created by alexandre on 27.05.20.
//

#include <iostream>
#include "python.h"


PyObject * fit(PyObject *self, PyObject * args)
{
    int iterations, max_individuals, inputs, outputs;
    if (!PyArg_ParseTuple(args, "iiii", &iterations, &max_individuals, &inputs, &outputs))
        return nullptr;
    return PyLong_FromLong(iterations);
}

static PyMethodDef NeatMethods[] = {
        {"fit",  fit, METH_VARARGS,
                              "Execute a shell command."},
        {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

static struct PyModuleDef neatmodule = {
        PyModuleDef_HEAD_INIT,
        "neat",   /* name of module */
        nullptr, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NeatMethods
};

PyMODINIT_FUNC
PyInit_neat(void) {
    return PyModule_Create(&neatmodule);
}
