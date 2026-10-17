import numpy as np
import pandas as pd
import torch as to


def convert_to_torch(A):
    """
    Converts an array (either Numpy or Pandas) to a torch.Tensor.

    Parameters
    ----------
    A : array_like
        Container to convert.

    Returns
    -------
    B : torch.Tensor
        A converted to torch.

    """
    # FIXME
    # 1. fails with list input
    # 2. Following warning is raised with Pandas objects:
    # kAOV/src/kaov/utils.py:22: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:213.)
    # B = to.from_numpy(A.to_numpy().reshape(-1, 1)).double()
    # 3. refactor function

    if isinstance(A, pd.Series):
        B = to.from_numpy(A.to_numpy().reshape(-1, 1)).double()
    elif isinstance(A, pd.DataFrame):
        B = to.from_numpy(A.to_numpy()).double()

    elif isinstance(A, to.Tensor):
        B = A.double()
    else:
        print("DEBUG")
        try:
            X = A.to_numpy() if not isinstance(A, np.ndarray) else A.copy()
            B = to.from_numpy(X).double()
        except AttributeError:
            print(f'Unknown data type {type(A)}')
            B = None

    return B
