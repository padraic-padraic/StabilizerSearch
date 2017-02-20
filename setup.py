from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.extension import Extension

import os
import subprocess

def pre_build_dependencies():
    build_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(build_root, 'stabilizer_search', 'mat'))
    subprocess.call(["make", "clean"])
    subprocess.call(["make", "all"])
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
        include_dirs=['./stabilizer_search/mat/'],
        library_dirs=['./stabilizer_search/mat/'],
        libraries=["m"],
        extra_objects=["./stabilizer_search/mat/haarrandom.a"]
        )
]


setup(
    name='stabilizer_search',
    version='1.0.3',
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
    ext_modules=cythonize(EXTENSIONS),
    cmdclass={'install': PreBuildInstall,
              'build_ext': PreBuildExt}
)
