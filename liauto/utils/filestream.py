import os
import shutil
import hashlib
from contextlib import contextmanager


__all__ = ["exists", "normpath", "reader", "writer", "makedirs", "delete", "get_md5sum"]


def exists(path):
    """Whether path exists."""
    return os.path.exists(path)


def normpath(path):
    """
    Normalize file path.
    """
    return os.path.normpath(path)


@contextmanager
def reader(
    path, encoding=None, delimiter=None, buffer_size=128 * 1024 * 1024, mode="r"
):
    """
    Return a file reader.

    Parameters
    ----------
    path : str
        File path
    encoding : str
        Encoding type
    delimiter : str
        Delimiter
    buffer_size : int, optional
        Not used.
    """
    with open(path, mode, encoding=encoding, newline=delimiter) as f:
        yield f


@contextmanager
def writer(path, encoding=None, append=False):
    """
    Return a file writer
    """
    w_mode = "w" if append else "a"
    with open(path, w_mode, encoding=encoding) as f:
        yield f


def makedirs(path, mode=0o777):
    """Create a remote directory, recursively if necessary.

    Parameters
    ----------
    path : str
        The file or directory to make.
    mode : int, default=0o777
        Octal permission to set on the newly created directory.
    """
    if not os.path.exists(path):
        os.makedirs(path, mode=mode, exist_ok=True)


def delete(path, recursive=False):
    """Delete file or directory.

    Parameters
    ----------
    path : str
        The file or directory to delete.
    recursive : bool, default=False
        Whether to recursively delete files and directories

    Returns
    -------
    bool
        returns True if the deletion was successful and False if no
        file or directory previously existed
    """
    if recursive:
        try:
            shutil.rmtree(path)
            return True
        except Exception:
            return False
    else:
        try:
            os.remove(path)
            return True
        except Exception:
            return False


def get_md5sum(path, chunksize=128 * 1024 * 1024):
    """
    Get the md5sum.

    Parameters
    ----------
    path : str
        File
    chunksize : int, optional
        Chunk size, by default 128*1024*1024
    """
    assert exists(path), f"{path} does not exists"
    md5 = hashlib.md5()
    with reader(path, mode="rb") as fread:
        while True:
            content = fread.read(chunksize)
            if not content:
                break
            md5.update(content)
    return md5.hexdigest()
