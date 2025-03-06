"""
Helper functions used by other package modules.

"""

import hashlib
import json
import zipfile

from pathlib import Path
from typing import Any, Union


def read_json(json_path: Union[str, Path]) -> dict[str, Any]:
    """
    Reads a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary representing the JSON structure.
    """
    assert Path(json_path).exists(), f"JSON file {json_path} not found."
    with open(json_path) as fp:
        data: dict = json.load(fp)

    return data


def save_json(
    data: dict, json_path: Union[str, Path], sort_keys=False, indent=4
) -> None:
    """
    Saves a JSON object to a file.

    Args:
        data (dict): JSON object to be saved.
        json_path (str): Path to the JSON file.
        sort_keys (bool, optional): Whether to sort the JSON keys. Defaults to False.
        indent (int, optional): Indentation level for pretty-formatting the JSON file. Defaults to None.

    Returns:
        None
    """
    Path(json_path).parent.mkdir(exist_ok=True)
    with open(json_path, "w") as fp:
        json.dump(data, fp, sort_keys=sort_keys, indent=indent)


def extract_archive(file_path: Union[str, Path]) -> list[str]:
    """
    Extracts a ZIP archive into a directory with the same name as the file's base name.

    Args:
        file_path (str): Path to the ZIP file.

    Returns:
        list: A list of extracted file paths.
    """
    dst_dir = file_path.parent.joinpath(file_path.stem)
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        print(f"Extracting file {file_path}")
        zip_ref.extractall(dst_dir)

    extracted_files = [str(f) for f in Path(dst_dir).rglob("*.*")]
    return extracted_files


def sha256(file_path) -> str:
    """
    Calculates the SHA-256 hash of a file.

    Reference:
        https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html

    Args:
        file_path (str): Path to the file.

    Returns:
        str: A string representing the SHA-256 hash hex digest.
    """
    if not Path(file_path).is_file():
        return ""

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
