import os
import numpy

# See if Cython is installed
try:
    from Cython.Build import cythonize
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    def build(setup_kwargs):
        pass
# Cython is installed. Compile
else:
    from setuptools import Extension
    from setuptools.dist import Distribution
    from distutils.command.build_ext import build_ext

    # This function will be executed in setup.py:
    def build(setup_kwargs):
        # The files you want to compile
        ext_modules = [
            Extension(
            "rwlenspy.lensing",
            ["rwlenspy/lensing.pyx"],
            include_dirs = ['rwlenspy/.'],
            extra_compile_args=['-fopenmp','-ffast-math','-std=c++11'],
            extra_link_args=['-fopenmp'],)
        ]
        
        # Build
        setup_kwargs.update({
            'ext_modules': cythonize(
                ext_modules,
                language_level=3,
                compiler_directives={'linetrace': True},
            ),
            'cmdclass': {'build_ext': build_ext}
            ,
            'include_dirs': [numpy.get_include()]
        }
        )
