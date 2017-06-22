#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "lib/SymplecticPauli.h"
#include "lib/StabilizerMatrix.h"
#include "lib/generation.h"

namespace py = pybind11;

PYBIND11_MODULE(c_stabilizers, m) {
    py::class_<SymplecticPauli>(m, "SymplecticPauli")
        .def(py::init<const unsigned int nQubits, const unsigned int Num>())
        .def_readonly("n_qubits", &SymplecticPauli::NQubits)
        .def_property("xNum", &SymplecticPauli::XNum, &SymplecticPauli::setX)
        .def_property("zNum", &SymplecticPauli::ZNum, &SymplecticPauli::setZ)
        .def("commutes", &SymplecticPauli::commutes)
        .def("__str__", &SymplecticPauli::toString)
        .def("to_matrix", &SymplecticPauli::toMatrix)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self);

    m.def("commutivity_test", &commutivityTest, "Test mutual commutivity within a set of pauli matrices");

    py::class_<StabilizerMatrix>(m, "StabilizerMatrix")
        .def(py::init<const unsigned int nQubits, std::vector<SymplecticPauli> paulis>())
        .def_readonly("n_qubits", &StabilizerMatrix::NQubits)
        .def("to_canonical_form", &StabilizerMatrix::to_canonical_form)
        .def("linearly_independent", &StabilizerMatrix::linearlyIndependent)
        .def("get_generators", &StabilizerMatrix::Generators)
        .def("get_projector", &StabilizerMatrix::projector)
        .def("get_stabilizer_state", &StabilizerMatrix::stabilizerState)
        .def("__str__", &StabilizerMatrix::toString)
        .def(py::self == py::self)
        .def(py::self != py::self);

    m.def("c_get_stabilizer_groups", &getStabilizerGroups,
          py::arg("n_qubits"), py::arg("n_states"), py::arg("real_only"));

}