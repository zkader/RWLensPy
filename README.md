# RWLensPy

This is a python package that generates observed morphologies and propagation transfer functions for radio wave propgation recorded by a radio telescope.

The code can be installed with:

`pip install rwlenspy`

## Propagation Transfer Function

## Spatial Images

## Multiplane Systems

## Custom Install

The package is built with Poetry and Cython using C++11 and OpenMP. This requires having a compiler for  If one requires a dev install, this can be done with:

`poetry install --with test,dev`
`poetry run python`

Poetry installs the package within it's virtual environment. Tests can be run with:

`poetry run pytest`

