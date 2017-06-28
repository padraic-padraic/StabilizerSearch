
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
#include <Eigen/Householder>

#include "lib/SymplecticPauli.h"
#include "lib/StabilizerMatrix.h"
#include "lib/PauliMatrices.h"
#include "lib/generation.h"
#include "lib/orthogonalisation.h"

namespace py = pybind11;

SymplecticPauli pauliFromString(py::str pauli_literals){
    return SymplecticPauli(std::string(pauli_literals));
}

PYBIND11_PLUGIN(c_stabilizers) {
    py::module m("c_stabilizers", "c backend for stabilizer group and state generation.");
    py::class_<SymplecticPauli>(m, "SymplecticPauli")
        .def(py::init<const unsigned int, const unsigned int>())
        .def_property_readonly("n_qubits", &SymplecticPauli::NQubits)
        .def_property("xNum", &SymplecticPauli::XNum, &SymplecticPauli::setX)
        .def_property("zNum", &SymplecticPauli::ZNum, &SymplecticPauli::setZ)
        .def("commutes", &SymplecticPauli::commutes)
        .def("__str__", &SymplecticPauli::toString)
        .def("__repr__", [](const SymplecticPauli& p){
            return "<c_stabilizers.SymplecticPauli: "+ p.toString() + ">";
        })
        .def("is_real", &SymplecticPauli::isReal)
        .def("to_matrix", &SymplecticPauli::toMatrix)
        .def("from_string", &pauliFromString)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self);

    m.def("commutivity_test", &commutivityTest, "Test mutual commutivity within a set of pauli matrices");

    py::class_<StabilizerMatrix>(m, "StabilizerMatrix")
        .def(py::init<const unsigned int, std::vector<SymplecticPauli>>())
        .def_property_readonly("n_qubits", &StabilizerMatrix::NQubits)
        .def_property_readonly("generators", &StabilizerMatrix::Generators)
        .def("to_canonical_form", &StabilizerMatrix::toCanonicalForm)
        .def("linearly_independent", &StabilizerMatrix::linearlyIndependent)
        .def("get_projector", &StabilizerMatrix::projector)
        .def("get_stabilizer_state", &StabilizerMatrix::stabilizerState)
        .def("__str__", &StabilizerMatrix::toString)
        .def(py::self == py::self)
        .def(py::self != py::self);

    m.def("c_get_stabilizer_groups", 
           (std::vector<StabilizerMatrix> (*)(unsigned int, unsigned int, bool, bool))&getStabilizerGroups,
            py::arg("n_qubits"), py::arg("n_states"), py::arg("real_only")=false, py::arg("verbose")=false);
    m.def("c_get_eigenstates", 
        (VectorList (*)(std::vector<StabilizerMatrix>, unsigned int, bool))&getStabilizerStates,
        py::arg("groups"), py::arg("n_states"), py::arg("verbose")=false);
    m.def("c_get_projector", &orthoProjector, py::arg("states"));
    return m.ptr();
}