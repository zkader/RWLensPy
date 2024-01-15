import numpy as np
from time import time

from astropy import units as u
from astropy import constants as c
from astropy import cosmology


def test_import():
    try:
        import rwlenspy
    except ImportError:
        raise ImportError(
            "Could not Import module rwlenspy. Check if module is complied."
        )
    return


import rwlenspy as rwl
from rwlenspy.utils import *

def test_transferfunc():
    pass


def test_analyticgrav():
    pass
