#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from setuptools.command.build_ext import build_ext
from setuptools import Extension

from pathlib import Path

root = Path(__file__).parent

print("Build File Imported")

# See if Cython is installed
try:
    from Cython.Build import cythonize
# If Cython is not available
except ImportError:
    def build(setup_kwargs):
        print("Build without Cython")
        ext_modules = [
            Extension(
                "rwlenspy.lensing",
                ["rwlenspy/lensing.cpp"],
            )
        ]
        setup_kwargs.update(
            {
                "ext_modules": ext_modules,
                "include_dirs": [numpy.get_include()],
            }
        )
# Cython is installed, Compile.
else:
    # This function will be executed in setup.py:
    def build(setup_kwargs):
        print("Build with Cython")
        # The files you want to compile
        ext_modules = [
            Extension(
                "rwlenspy.lensing",
                sources=["rwlenspy/lensing.pyx", "rwlenspy/rwlens.cpp"],
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-fopenmp", "-std=c++11"],
                extra_link_args=["-fopenmp"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                language="c++"
            )
        ]

        # Build
        setup_kwargs.update(
            {
                "ext_modules": cythonize(
                    ext_modules,
                    language_level=3,
                    compiler_directives={"linetrace": True},
                ),
                "cmdclass": {"build_ext": build_ext},
            }
        )
