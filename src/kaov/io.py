import dill
import gzip


def save(obj, file_name: str, compress: bool = True):
    """
    Save AOV object to disk in binary format (pickle/dill) possibly
    with Gzip compression.

    If compression is enabled, a `.gz` extension is added to `file_name`
    if not already present.

    Args:
        obj (AOV): an object to save.
        file_name (str): absolute or relative path to file where to store the
            input object.
        compress (bool): flag to enable/disable compression when saving AOV
            object to disk.

    Returns: no return value.
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

    Args:
        file_name (str): absolute or relative path to file where the AOV object
            was saved on disk.
    compress (bool): flag to enable/disable decompression when loading the
        AOV object from disk.
    """
    if compressed:
        custom_open = gzip.open
    else:
        custom_open = open

    with custom_open(file_name, 'rb') as f:
        return dill.load(f)
