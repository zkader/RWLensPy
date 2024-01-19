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
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    def build(setup_kwargs):
        print("Build without Cython")
# Cython is installed, Compile.
else:
    # This function will be executed in setup.py:
    def build(setup_kwargs):
        print("Build with Cython")
        # The files you want to compile
        ext_modules = [
            Extension(
                "rwlenspy.lensing",
                ["rwlenspy/lensing.pyx"],
                include_dirs=["rwlenspy/."],
                extra_compile_args=["-fopenmp", "-std=c++11"],
                extra_link_args=["-fopenmp"],
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
                "include_dirs": [numpy.get_include()],
            }
        )
