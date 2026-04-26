import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import math

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows

    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    t = nl.tile_size.pmax
    t_rows = math.ceil(i_rows / t)
    t_cols = math.ceil(i_cols / t)

    for i in nl.affine_range(t_rows):
        for j in nl.affine_range(t_cols):
            r = i * t
            c = j * t

            in_r_off, in_c_off = nl.mgrid[0:t, 0:t]
            mask = ((r + in_r_off) < i_rows) & ((c + in_c_off) < i_cols)

            tile = nl.load_transpose2d(in_tensor[r + in_r_off, c + in_c_off], mask=mask)
            nl.store(out_tensor[c + in_r_off, r + in_c_off], tile, mask=mask)

    return out_tensor

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    t = nl.tile_size.pmax
    t_rows = math.ceil(BATCH_SIZE / t)

    b_tile = nl.load(b)
    for i in nl.affine_range(t_rows):
        r = i * t

        in_r_off, in_c_off = nl.mgrid[0:t, 0:HIDDEN_SIZE]
        mask = ((r + in_r_off) < BATCH_SIZE) & ((in_c_off) < HIDDEN_SIZE)

        A_tile = nl.load(A[r + in_r_off, in_c_off], mask=mask)
        y = nl.add(A_tile, b_tile)

        if act == 'relu':
            z = nl.maximum(y, 0)
        elif act == 'softmax':
            y_max = nl.max(y, axis=1)
            y_shifted = nl.subtract(y, y_max)
            y_exp = nl.exp(y_shifted)
            y_sum = nl.sum(y_exp, axis=1)
            z = nl.divide(y_exp, y_sum)

        nl.store(result[r + in_r_off, in_c_off], z, mask=mask)

    return result

@nki.jit
def nki_forward(
        X,
        W1,
        b1,
        W2,
        b2,
        matmul_kernel='tiled'
    ):
    """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

    Args:
        X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
        W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
        b1: the bias vector of shape [HIDDEN_SIZE]
        W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
        b2: the bias vector of shape [OUTPUT_SIZE]
    Returns:
        probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
    
    Option:
        matmul_kernel: the matrix multiplication kernel to use 
            - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
    """
    if matmul_kernel == 'tiled':
        nki_matmul = nki_matmul_tiled_
    elif matmul_kernel == 'hoist_load':
        nki_matmul = nki_matmul_hoist_load_
    elif matmul_kernel == 'block_free_dimension':
        nki_matmul = nki_matmul_block_free_dimension_
    elif matmul_kernel == 'fully_optimized':
        nki_matmul = nki_matmul_fully_optimized_
    else:
        raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

    # Layer 1
    # YOUR CODE HERE
    X_t = nki_transpose(X)
    z0 = nki_bias_add_act(nki_matmul(X_t, W1), b1)

    # Layer 2 (output)
    # YOUR CODE HERE
    z0_t = nki_transpose(z0)
    probs = nki_bias_add_act(nki_matmul(z0_t, W2), b2, act='softmax')

    return probs


@nki.jit
def nki_predict(
        X,
        W1,
        b1,
        W2,
        b2,
        matmul_kernel='tiled'
    ):
    """NKI kernel run forward pass and predict the classes of the input tensor.

    Args:
        X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
        W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
        b1: the bias vector of shape [HIDDEN_SIZE]
        W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
        b2: the bias vector of shape [OUTPUT_SIZE]
    Returns:
        predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
    
    Option:
        matmul_kernel: the matrix multiplication kernel to use 
            - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

    Returns:
        predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
    """
    probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel=matmul_kernel) # YOUR CODE HERE
    BATCH_SIZE, OUTPUT_SIZE = probs.shape
    predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

    # YOUR CODE HERE
    max8_prob = nisa.max8(src=probs)
    

    return predictions