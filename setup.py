from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

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
    version='1.0.0',
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
    ext_modules=cythonize(EXTENSIONS)
)
