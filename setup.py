from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.extension import Extension

import numpy
import os
import subprocess

NUMPY_INC = numpy.get_include()

def pre_build_dependencies():
    build_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(build_root, 'stabilizer_search', 'clib', 'haar_random'))
    subprocess.call(["make", "clean"])
    subprocess.call(["make", "all"])
    os.chdir(os.path.join(build_root, 'stabilizer_search', 'clib', 'StabilizerCPP'))
    subprocess.call(["cmake", "-DBUILD_TESTING=OFF", "-DBUILD_EXECUTABLE=OFF", "./"])
    subprocess.call(["make"])
    os.chdir(build_root)

class PreBuildInstall(install):

    def run(self):
        pre_build_dependencies()
        install.run(self)


class PreBuildExt(build_ext):

    def run(self):
        pre_build_dependencies()
        build_ext.run(self)

EXTENSIONS = [
    Extension(
        "stabilizer_search.mat.haar_random",
        ["stabilizer_search/mat/haar_random.pyx"],
        include_dirs=['./stabilizer_search/clib/haar_random/'],
        library_dirs=['./stabilizer_search/clib/haar_random/'],
        extra_objects=["./stabilizer_search/clib/haar_random/haarrandom.a"],
        ),
    Extension(
        "stabilizer_search.stabilizers.cy_generators",
        ["stabilizer_search/stabilizers/cy_generators.pyx"],
        include_dirs=[NUMPY_INC, "stabilizer_search/stabilizers/"]),
    Extension(
        "stabilizer_search.stabilizers.cy_eigenstates",
        ["stabilizer_search/stabilizers/cy_eigenstates.pyx"],
        include_dirs=[NUMPY_INC],
        ),
    Extension(
        "stabilizer_search.linalg.cy_gram_schmidt",
        ["stabilizer_search/linalg/cy_gram_schmidt.pyx"],
        include_dirs=[NUMPY_INC]
        ),
    Extension(
        "stabilizer_search.search.cy_do_random_walk",
        ["stabilizer_search/search/cy_do_random_walk.pyx"],
        include_dirs=["./stabilizer_search/linalg/", NUMPY_INC],
        libraries=["m"]
    )
    # Extension(
    #     "stabilizer_search.stabilizers.c_generators",
    #     ["stabilizer_search/stabilizers/c_generators.pyx"],
    #     include_dirs=["./stabilizer_search/clib",
    #                   "./stabilizer_search/clib/StabilizerCPP/src",
    #                   NUMPY_INC],
    #     library_dirs=["./stabilizer_search/clib/StabilizerCPP/out/"],
    #     libraries=["libsymplectic_stabilizer"],
    #     extra_objects=["./stabilizer_search/clib/StabilizerCPP/out/libsymplectic_stabilizer.a"],
    #     language="c++"
    #     )
]


setup(
    name='stabilizer_search',
    version='1.1.0',
    url="https://github.com/padraic-padraic/StabilizerSearch",
    author="Padraic Calpin",
    description='Stabilizer Search',
    license="GPLv3.0",
    keywords='quantum computing stabilizers simulation',
    packages=find_packages(exclude=['tests', '*tests']),
    package_data={
        'stabilizer_search.stabilizers':['/data/*.pkl']
    },
    install_requires=['cython', 'numpy', 'scipy', 'six'],
    ext_modules=cythonize(EXTENSIONS, include_path=["./stabilizer_search/mat/", "./stabilizer_search/linalg/"]),
    cmdclass={'install': PreBuildInstall,
              'build_ext': PreBuildExt}
)
