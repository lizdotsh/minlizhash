import pickle

import numpy as np
import numpy.typing as npt


def progress_bar(i, mx, bar_length=20) -> str:
    """Generates an ASCII progress bar."""
    percent = float(i) / mx
    hashes = "#" * int(round(percent * bar_length))
    spaces = " " * (bar_length - len(hashes))
    return f"[{hashes}{spaces}] {i} of {mx}"


def cosine_similarity(a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]) -> float:
    """Compute the cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def jaccard_similarity(a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]) -> float:
    """Compute the Jaccard similarity between two vectors"""
    return np.intersect1d(a, b).size / np.union1d(a, b).size


def load_pickle(filename):
    """Load a pickle file"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def simple_shingles_generator(text: str, k: int = 3) -> npt.NDArray[np.str_]:
    """Generate shingles of length k from the text
    Made before moving over to using BPE; figured I should keep it around even though it probably won't work
    """
    return np.fromiter(
        (text[i : i + k] for i in range(len(text) - k + 1)), dtype="<U" + str(k)
    )
