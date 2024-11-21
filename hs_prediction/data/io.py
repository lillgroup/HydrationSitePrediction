from typing import Any
import json
import pickle
import yaml


def load_json(filepath: str) -> Any:
    with open(filepath, "r") as fi:
        d = json.load(fi)
    return d


def save_json(filepath: str, d: Any) -> None:
    with open(filepath, "w") as fo:
        json.dump(d, fo)


def load_pickle(filepath: str) -> Any:
    """Load pickle file from filepath"""
    with open(filepath, "rb") as fo:
        d = pickle.load(fo)
    return d


def save_pickle(filepath: str, d: Any) -> None:
    """Save pickle file to filepath"""
    with open(filepath, "wb") as fo:
        pickle.dump(d, fo)


def load_yaml(filepath: str) -> Any:
    with open(filepath, "r") as fi:
        d = yaml.safe_load(fi)
    return d


def save_yaml(filepath: str, d: Any) -> None:
    with open(filepath, "w") as fo:
        yaml.dump(d, fo)


