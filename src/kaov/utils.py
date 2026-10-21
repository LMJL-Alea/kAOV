import numpy as np
import pandas as pd
import torch as to


def convert_to_torch(
    data: np.ndarray | pd.DataFrame | pd.Series | to.Tensor
) -> to.Tensor:
    """
    Convert a NumPy array, Pandas DataFrame, Pandas Series or PyTorch tensor
    to the corresponding float64 PyTorch tensor.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, pandas.Series or torch.Tensor
        Input array-like data to convert.

    Returns
    -------
    torch.Tensor
        A PyTorch tensor with the same shape as the input (except for Series,
        which are converted to 2D column vectors) and float64 dtype.

    Raises
    ------
    TypeError
        If the input is not a NumPy array, Pandas DataFrame, Pandas Series pr
        torch.Tensor.
    """
    if isinstance(data, to.Tensor):
        return data.double()

    elif isinstance(data, np.ndarray):
        # Ensure the array is writable to avoid PyTorch warnings
        if not data.flags.writeable:
            data = data.copy()
        return to.as_tensor(data, dtype=to.float64)

    elif isinstance(data, pd.Series):
        # Convert Series to a 2D column vector
        data_np = data.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
        return to.as_tensor(data_np, dtype=to.float64)

    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to NumPy array (preserves shape)
        data_np = data.to_numpy(dtype=np.float64, copy=True)
        return to.as_tensor(data_np, dtype=to.float64)

    else:
        raise TypeError(
            f"Unsupported input type: {type(data)}. "
            "Expected numpy.ndarray, pandas.DataFrame, or pandas.Series."
        )
