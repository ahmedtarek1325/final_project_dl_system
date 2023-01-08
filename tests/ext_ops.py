import sys
sys.path.append('./python')
import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl
import itertools


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

'''
{"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
'''

bbm_params = [
    {"shape_a": (10,3,5), "shape_b": (10,5,7)},
    
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", bbm_params)
def test_bbm_forward(params, device):
    np.random.seed(0)
    shape_A, shape_B = params['shape_a'], params['shape_b']
    _A = np.random.randn(*shape_A)
    _B = np.random.randn(*shape_B)

    lhs = _A @ _B

    A = nd.NDArray(_A, device=device)
    B = nd.NDArray(_B, device=device)
    rhs = ndl.bmm( A , B)

    assert np.linalg.norm(rhs.numpy() -lhs) < 1e-4






