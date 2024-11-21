from multiprocessing import Pool
from typing import Tuple, Callable, List
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

FEATURES = {
    "ATOM_TYPE": [
        "N",
        "C",
        "CA",
        "O",
        "CB",
        "CG",
        "CD",
        "CD1",
        "CD2",
        "CG2",
        "CG1",
        "CZ",
        "CE1",
        "OE1",
        "OD1",
        "CE2",
        "CE",
        "OG",
        "OE2",
        "NE2",
        "NZ",
        "OG1",
        "OD2",
        "NH1",
        "NH2",
        "NE",
        "ND2",
        "OH",
        "ND1",
        "SD",
        "SG",
        "CH2",
        "CE3",
        "NE1",
        "CZ2",
        "CZ3",
        "OXT",
        "other",
    ],
    "RESIDUE": [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "other",
    ],
}


def flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, tuple) or isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def pool_wrapper(
    fn: Callable, iterable: List, num_workers: int = 1, desc: str = "Processing"
) -> List:
    """Multiprocessing wrapper with progress bar and filtering of results"""
    if num_workers == 1:
        results = []
        for key in tqdm(iterable, desc=desc):
            res = fn(key)
            if res is not None:
                results.append(res)
    else:
        pool = Pool(processes=num_workers)
        results = tqdm(pool.imap(fn, iterable), desc=desc, total=len(iterable))
        results = [result for result in results if result is not None]
        pool.close()
    return results


def create_distance_edges(
    coords: NDArray, r: float
) -> Tuple[NDArray, NDArray, NDArray]:
    """Calculates distances between points and returns edges within radius r"""
    # Calculate distances between points
    rel_coords = coords[:, None] - coords[None, :]
    dis = np.linalg.norm(rel_coords, axis=-1)
    # Create edges
    edge_index = np.argwhere(dis < r).T
    src, dst = edge_index
    edge_vec = rel_coords[src, dst]
    edge_dis = dis[src, dst]
    return edge_index, edge_vec, edge_dis


def safe_index(list_, e):
    """Return index of element e in list l. If e is not present, return the last index"""
    try:
        return list_.index(e)
    except:
        return len(list_) - 1


def one_hot(idx, n):
    """Create one-hot vector of length n with 1 at idx"""
    v = np.zeros(n)
    v[idx] = 1
    return v


def get_feature(key, val, safe=True):
    """Create one-hot vector for specific feature"""
    feat = FEATURES[key]
    if safe:
        idx = safe_index(feat, val)
    else:
        idx = feat.index(val)
    return one_hot(idx, len(feat))


def rbf(values, min_val=0.0, max_val=40.0, n_kernels=64, gamma=None):
    if gamma is None:
        gamma = n_kernels**2 / (2 * (max_val - min_val) ** 2)
    values = np.clip(values, min_val, max_val)
    mus = np.linspace(min_val, max_val, n_kernels)
    return np.exp(-gamma * np.square(values.reshape(-1, 1) - mus))
