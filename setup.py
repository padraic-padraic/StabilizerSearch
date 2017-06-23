from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.extension import Extension

import numpy
import os
import setuptools
import subprocess
import sys


NUMPY_INC = numpy.get_include()


class MixedExtension(Extension):
    """Leightweight helper class that tells PreBuildExt and PreBuildInstall 
    whether to cythonize, 'pybindize', or otherwise leave well alone, the 
    extensions."""
    def __init__(self, *args, **kwargs):
        self.ext_type = kwargs.pop('ext_type', None)
        super().__init__(*args, **kwargs)

## The `get_pybind_include` and `has_flag` functions are taken directly
## from the pybind 'example' project 
## https://github.com/pybind/python_example/blob/master/setup.py
## The custom extension command given in the above example has been adapted into
## a cython like 'pybindize' function that sets up the appropriate compiler flags
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


def pre_build_dependencies():
    build_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(build_root, 'stabilizer_search', 'clib', 'haar_random'))
    subprocess.call(["make", "clean"])
    subprocess.call(["make", "all"])
    os.chdir(os.path.join(build_root, 'stabilizer_search', 'clib', 'StabilizerCPP'))
    subprocess.call(["cmake", "-DBUILD_SHARED_LIBS=OFF", "-DBUILD_TESTING=OFF", 
                    "-DBUILD_EXECUTABLE=OFF", "-DBUILD_PYTHON=ON", "./"])
    subprocess.call(["make"])
    os.chdir(build_root)


EXTENSIONS = [
    MixedExtension(
        "stabilizer_search.mat.haar_random",
        ["stabilizer_search/mat/haar_random.pyx"],
        include_dirs=['./stabilizer_search/clib/haar_random/',
                      './stabilizer_search/mat/'],
        library_dirs=['./stabilizer_search/clib/haar_random/'],
        extra_objects=["./stabilizer_search/clib/haar_random/haarrandom.a"],
        ext_type='cython'
        ),
    MixedExtension(
        "stabilizer_search.stabilizers.cy_generators",
        ["stabilizer_search/stabilizers/cy_generators.pyx"],
        include_dirs=[NUMPY_INC, "stabilizer_search/stabilizers/"],
        ext_type='cython'
        ),
    MixedExtension(
        "stabilizer_search.stabilizers.cy_eigenstates",
        ["stabilizer_search/stabilizers/cy_eigenstates.pyx"],
        include_dirs=[NUMPY_INC],
        ext_type='cython'
        ),
    MixedExtension(
        "stabilizer_search.search.cy_do_random_walk",
        ["stabilizer_search/search/cy_do_random_walk.pyx"],
        include_dirs=["./stabilizer_search/linalg/", NUMPY_INC],
        libraries=["m"], 
        ext_type='cython'
    ),
    MixedExtension(
        "stabilizer_search.clib.c_stabilizers",
        ['stabilizer_search/clib/smatrix.cpp'],
        include_dirs=[
            'stabilizer_search/clib/StabilizerCPP/src/',
            'stabilizer_search/clib/eigen/',
            'stabilizer_search/clib/dynamic_bitset/include/',
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        library_dirs=['stabilizer_search/clib/StabilizerCPP/out/'],
        runtime_library_dirs=['stabilizer_search/clib/StabilizerCPP/out/',
                              'stabilizer_search/clib/eigen/',
                              'stabilizer_search/clib/dynamic_bitset/include/'],
        extra_objects=['stabilizer_search/clib/StabilizerCPP/out/libsymplectic_stabilizer.a'],
        libraries=['symplectic_stabilizer'],
        # extra_link_args=['-static'],
        language="c++",
        ext_type='pybind'
    )
]


class PreBuildInstall(install):

    def pybindize(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in pb_extensions:
            ext.extra_compile_args = opts
        return pb_extensions

    def run(self):
        pre_build_dependencies()
        self.pybindize(PYBIND_EXTENSIONS)
        install.run(self)


class PreBuildExt(build_ext):

    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    # if sys.platform == 'darwin':
    #     c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def pybindize(self, pb_extensions):
        ct = build_ext.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in pb_extensions:
            ext.extra_compile_args = opts
        self.extensions += pb_extensions

    def run(self):
        pre_build_dependencies()
        # ct = self.compiler.compiler_type
        for ext in self.extensions:
            if ext.ext_type == 'cython':
                ext = cythonize(ext, include_path=["./stabilizer_search/mat/", 
                          "./stabilizer_search/linalg/"])[0]
            elif ext.ext_type == 'pybind':
                ct='unix'
                opts = self.c_opts.get(ct, [])
                if ct == 'unix':
                    opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
                    opts.append('-std=c++11')
                    # opts.append(cpp_flag(self.compiler))
                    # if has_flag(self.compiler, '-fvisibility=hidden'):
                    #     opts.append('-fvisibility=hidden')
                elif ct == 'msvc':
                    opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
                ext.extra_compile_args = opts
            else:
                continue
        build_ext.run(self)

setup(
    name='stabilizer_search',
    version='2.0.0',
    url="https://github.com/padraic-padraic/StabilizerSearch",
    author="Padraic Calpin",
    description='Stabilizer Search',
    license="GPLv3.0",
    keywords='quantum computing stabilizers simulation',
    packages=find_packages(exclude=['tests', '*tests']),
    package_data={
        'stabilizer_search.stabilizers':['/data/*.states',
                                         '/data/*.generators']
    },
    install_requires=['cython', 'numpy', 'scipy', 'six', 'pybind11'],
    ext_modules=EXTENSIONS,# + PYBIND_EXTENSIONS,
    cmdclass={'install': PreBuildInstall,
              'build_ext': PreBuildExt}
)
