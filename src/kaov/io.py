import dill
import gzip


def save(obj, file_name: str, compress: bool = True):
    """
    Save AOV object to disk in binary format (pickle/dill) possibly
    with Gzip compression.

    If compression is enabled, a `.gz` extension is added to `file_name`
    if not already present.

    Parameters
    ----------
    obj : kaov.kaov.AOV
        An object to save.
    file_name : str
        Absolute or relative path to file where to store the input object.
    compress : bool, optional
        Flag to enable/disable compression when saving AOV object to disk.
        Defaults to True.

    Returns
    -------
    None
        This function saves the object to disk and returns nothing.
    """
    # compress result file?
    if compress:
        custom_open = gzip.open
        # add `.gz` extension if needed
        if not file_name.endswith('.gz'):
            file_name = file_name + '.gz'
    else:
        custom_open = open

    with custom_open(file_name, 'wb') as f:
        dill.dump(obj, f)


def load(file_name: str, compressed: bool = True):
    """
    Load AOV object that was saved on disk in binary format (pickle/dill)
    possibly with Gzip decompression.

    Parameters
    ----------
    file_name : str
        Absolute or relative path to file where the AOV object was saved
        on disk.
    compressed : bool, optional
        Flag to enable/disable decompression when loading the AOV object
        from disk. Defaults to True.

    Returns
    -------
    kaov.kaov.AOV
        The loaded AOV object.
    """
    if compressed:
        custom_open = gzip.open
    else:
        custom_open = open

    with custom_open(file_name, 'rb') as f:
        return dill.load(f)
