#include <pybind11/pybind11.h>

namespace py = pybind11;

void atomic_write(uint64_t buf_ptr, int idx, int value) {
    __atomic_store((int*)buf_ptr + idx, &value, __ATOMIC_SEQ_CST);
}

int atomic_read(uint64_t buf_ptr, int idx) {
    // on x86 this is just a normal load....
    int dst;
    __atomic_load((int*)buf_ptr + idx, &dst, __ATOMIC_SEQ_CST);
    return dst;
}

void atomic_read_all(uint64_t dst_ptr, uint64_t src_ptr,  int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        atomic_write(dst_ptr, i, atomic_read(src_ptr, i));
    }
}

PYBIND11_MODULE(atomic, m) {
    m.def("atomic_write", atomic_write);
    m.def("atomic_read", atomic_read);
    m.def("atomic_read_all", atomic_read_all);
}
