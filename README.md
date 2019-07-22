# Stabilizer Search
**Generating ~ _Stabilizer States_ ~ and finding the Stabilizer Rank with Python and Cython.**

This package contains code for generating stabilizer states, as numpy matrices, and for searching for stabilizer rank decompositions. The package also includes data files for stabilizer states on 1-4 qubits. 

** Important Note: This package is 'between versions' right now; there are some issues with the implementation of canonical tableaux, and it's not well documented. Fixes are coming shortly! **

---


## Installation

This package has been tested with Python 2.7 and 3.6. After [downloading](https://github.com/padraic-padraic/StabilizerSearch/archive/master.zip) the source from this repository, the package can be installed using
```shell
python setup.py install
```
or else run locally within the directory by first building the cython extensions with
```shell
python setup.py build_ext --inplace
```
