from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

extensions = [
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
    version='0.1.0',
    packages = find_packages(exclude=['tests', '*tests']),
    install_requires=['cython', 'numpy', 'scipy', 'six'],
    ext_modules=cythonize(extensions)
    )
