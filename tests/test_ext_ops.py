import sys
sys.path.append('./python')
import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl
import itertools


def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

'''
{"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
'''

bbm_params = [
    {"shape_a": (10,3,5), "shape_b": (10,5,7)},
    {"shape_a": (10,8,7,9), "shape_b": (10,8,9,2)},
    {"shape_a": (2,4,3,7,9), "shape_b": (2,4,3,9,1)}   
]

@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", bbm_params)
def test_bbm_forward(params, device):
    np.random.seed(0)
    shape_A, shape_B = params['shape_a'], params['shape_b']
    _A = np.random.randn(*shape_A)
    _B = np.random.randn(*shape_B)

    lhs = _A @ _B

    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    rhs = ndl.bmm( A , B)

    assert np.linalg.norm(rhs.numpy() -lhs) < 1e-4


###################################################################################
######################### Concatenate Ops Testing #################################

concatenate_params = [
    {"shape": (4,5,6,7),    "n": 4, "axis": 0},
    {"shape": (5,7,9,3,1), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2},
    {"shape": (2,3,7,8,9,4,2), "n": 5, "axis": 3},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", concatenate_params)
def test_concatenate_forward(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    to_concate_ndl = []
    to_concate_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_concate_ndl += [ndl.Tensor(_A, device=device)]
        to_concate_npy += [_A]

    lhs = np.concatenate(to_concate_npy, axis=axis)
    rhs = ndl.concat(to_concate_ndl, axis=axis)

    assert np.linalg.norm(rhs.numpy() -lhs) < 1e-7
    
stack_back_params = [
    ( (4,5,6,7), 3, 0),
    ( (5,7,9,3,1), 3, 1),
    ( (4, 5, 6), 3, 2)
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape, n, axis", stack_back_params)
def test_concatenate_backward(shape, n, axis, device):
    np.random.seed(0)
    get_tensor = lambda shape: ndl.Tensor(np.random.randn(*shape)*5, device=device)
    backward_check(ndl.concat, [get_tensor(shape) for _ in range(n)], axis=axis)



###################################################################################
######################### Split Ops Testing #################################

split_group_params = [
    {"shape": (4,9,5,3), "axis": 1, "splits":3},
    {"shape": (5,7,9,3,1),"axis": 2,"splits":3},
    {"shape": (4, 5, 6),"axis": 1,"splits":5},
    {"shape": (2,3,7,8,9,4,2), "axis": 3,"splits":2},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", split_group_params)
def test_split_group_forward(params, device):
    np.random.seed(0)
    shape, axis,splits = params['shape'], params['axis'], params['splits']

    _A = np.random.randn(*shape)
    A = ndl.Tensor(_A, device=device)

    lhs = np.split(_A, splits,axis)
    rhs = ndl.split_group(A,axis,splits)

    for i in range(splits):
        assert np.linalg.norm(rhs[i].numpy() -lhs[i]) < 1e-4