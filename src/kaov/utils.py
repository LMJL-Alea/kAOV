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
    if isinstance(A, pd.Series):
        B = to.from_numpy(A.to_numpy().reshape(-1, 1)).double()
    if isinstance(A, pd.DataFrame):
        B = to.from_numpy(A.to_numpy()).double()

    if isinstance(A, to.Tensor):
        B = A.double()
    else:
        try:
            X = A.to_numpy() if not isinstance(A, np.ndarray) else A.copy()
            B = to.from_numpy(X).double()
        except AttributeError:
            print(f'Unknown data type {type(A)}')
    return B
