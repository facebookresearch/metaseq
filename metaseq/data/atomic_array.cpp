#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>

static PyObject* kwnames;
static PyObject* format;
struct Owned {
    Owned(PyObject* o)
    : obj(o) {}
    ~Owned() {
        Py_XDECREF(obj);
    }
    PyObject* obj;
};

struct AtomicArray {
    PyObject_HEAD
    Py_buffer buffer;
    Py_ssize_t size;
    /* Type-specific fields go here. */
    static int init(AtomicArray* self, PyObject *args, PyObject *kwds) {
        self->buffer.obj = nullptr;
        PyObject* size;
        const char* names[] = {"size", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**) names, &size)) {
            return -1;
        }
        return init_size(self, size);
    }
    static int init_size(AtomicArray* self, PyObject *size) {
        self->buffer.obj = nullptr;
        Owned multiprocessing = PyImport_ImportModule("multiprocessing");
        if (!multiprocessing.obj) {
            return -1;
        }
        Owned Array = PyObject_GetAttrString(multiprocessing.obj, "Array");
        if  (!Array.obj) {
            return -1;
        }
        PyObject* call_args[] = {format, size, Py_False};
        Owned array = PyObject_Vectorcall(Array.obj, &call_args[0], 2, kwnames);
        if (!array.obj) {
            return -1;
        }
        if (-1 == PyObject_GetBuffer(array.obj, &self->buffer, PyBUF_WRITEABLE | PyBUF_SIMPLE)) {
            return -1;
        }
        self->size = self->buffer.len / self->buffer.itemsize;
        return 0;
    }
    static Py_ssize_t length(AtomicArray* self) {
        return self->size;
    }
    static PyObject * sq_item(AtomicArray * self, Py_ssize_t idx) {
        if (idx < 0 || idx >= self->size) {
            PyErr_Format(PyExc_IndexError, "%n", idx);
            return nullptr;
        }
        int dst;
        __atomic_load((int*)self->buffer.buf + idx, &dst, __ATOMIC_SEQ_CST);
        return PyLong_FromLong(dst);
    }
    static int sq_ass_item(AtomicArray * self, Py_ssize_t idx, PyObject* pyvalue) {
        if (idx < 0 || idx >= self->size) {
            PyErr_Format(PyExc_IndexError, "%n", idx);
            return -1;
        }
        int value = PyLong_AsLong(pyvalue);
        if (PyErr_Occurred()) {
            return -1;
        }
        __atomic_store((int*) self->buffer.buf + idx, &value, __ATOMIC_SEQ_CST);
        return 0;
    }
    static void dealloc(AtomicArray* self) {
        PyBuffer_Release(&self->buffer);
        Py_TYPE(self)->tp_free(self);
    }
    static PyObject* __getstate__(AtomicArray* self) {
        Owned bytes = PyBytes_FromStringAndSize(nullptr, self->buffer.len);
        if (!bytes.obj) {
            return nullptr;
        }
        char* buf = PyBytes_AsString(bytes.obj);
        for (Py_ssize_t i = 0; i < self->size; ++i) {
            int value;
            __atomic_load((int*)self->buffer.buf + i, &value, __ATOMIC_SEQ_CST);
            memcpy(buf + sizeof(int)*i, &value, sizeof(int));
        }
        Owned ln = PyLong_FromLong(self->size);
        if (!ln.obj) {
            return nullptr;
        }
        return PyTuple_Pack(2, ln.obj, bytes.obj);
    }
    static PyObject* __setstate__(AtomicArray* self, PyObject* state) {
        PyObject* ln = PyTuple_GetItem(state, 0);
        if (!ln) {
            return nullptr;
        }
        PyObject* bytes = PyTuple_GetItem(state, 1);
        if (!bytes) {
            return nullptr;
        }
        char* buf = PyBytes_AsString(bytes);
        if (-1 == init_size(self, ln)) {
            return nullptr;
        }
        if (PyBytes_Size(bytes) != sizeof(int)*self->size) {
            PyErr_Format(PyExc_ValueError, "bytes doesn't match array size");
            return nullptr;
        }
        memcpy(self->buffer.buf, buf, self->buffer.len);
        Py_RETURN_NONE;
    }
    static PyObject* from_tensor_data_ptr(AtomicArray* self, PyObject* obj) {
        uint64_t p = PyLong_AsLongLong(obj);
        if (PyErr_Occurred()) {
            return nullptr;
        }
        memcpy(self->buffer.buf, (void*)p, self->buffer.len);
        Py_RETURN_NONE;
    }
};


static PySequenceMethods tp_as_sequence = {
    (lenfunc) AtomicArray::length,
    0,
    0,
    (ssizeargfunc) AtomicArray::sq_item,
    0,
    (ssizeobjargproc) AtomicArray::sq_ass_item,
    0,
    0,
    0,
    0,
};

static PyMethodDef tp_methods[] = {
    {"__getstate__", (PyCFunction) AtomicArray::__getstate__, METH_NOARGS,
            "Pickle the object"
    },
    {"__setstate__", (PyCFunction) AtomicArray::__setstate__, METH_O,
            "Un-pickle the object"
    },
    {"from_tensor_data_ptr", (PyCFunction) AtomicArray::from_tensor_data_ptr, METH_O,
            "load data from a tensor"
    },
    {nullptr}
};

PyTypeObject AtomicArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "metaseq.data.atomic_array.AtomicArray",   /* tp_name */
    sizeof(AtomicArray),                    /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor) AtomicArray::dealloc,           /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,             /* tp_repr */
    0,                              /* tp_as_number */
    &tp_as_sequence,    /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    PyDoc_STR("Atomic Read/Write Shared Memory Array"), /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    tp_methods,                              /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) AtomicArray::init,            /* tp_init */
    0,                                    /* tp_alloc */
    PyType_GenericNew,                      /* tp_new */
};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "atomic_array",
    .m_doc = "atomic memory read write array",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_atomic_array(void) {
    PyObject *m;
    if (PyType_Ready(&AtomicArrayType) < 0)
        return nullptr;

    m = PyModule_Create(&custommodule);
    if (m == nullptr)
        return nullptr;

    Py_INCREF(&AtomicArrayType);
    if (PyModule_AddObject(m, "AtomicArray", (PyObject *) &AtomicArrayType) < 0) {
        Py_DECREF(&AtomicArrayType);
        Py_DECREF(m);
        return nullptr;
    }
    Owned lock = PyUnicode_FromString("lock");
    kwnames = PyTuple_Pack(1, lock.obj);
    format = PyUnicode_FromString("i");
    return m;
}
