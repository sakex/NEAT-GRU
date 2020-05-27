//
// Created by alexandre on 27.05.20.
//

#ifndef NEAT_GRU_PYTHON_H
#define NEAT_GRU_PYTHON_H

#include <Python.h>

extern "C" {
static PyObject *fit(PyObject *self, PyObject *);

PyMODINIT_FUNC PyInit_neat(void);
}

#endif //NEAT_GRU_PYTHON_H
