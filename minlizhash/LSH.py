# Everything LSH related lives here

from dataclasses import dataclass
from typing import Callable, List, Set, DefaultDict
import numpy as np
import pickle
from collections import defaultdict


@dataclass
class LSHIndex:
    def __init__(self, lsh_dict: dict[int, list[int]], seed: int, num_bands: int, num_seeds: int):
        self.lsh_dict = lsh_dict
        self.seed = seed
        self.num_bands = num_bands
        self.buckets = [defaultdict(list) for _ in range(self.num_bands)]
        self.rows_per_band = num_seeds // num_bands

    def add(self, doc_id: int, buckets: list[int]):
        for bucket in buckets:
            self.lsh_dict[bucket].append(doc_id)
    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    #    pickle.dump(self, open(filename, "wb"))

def lsh_buckets_vectorized(
    signatures: np.ndarray, num_bands: int, seed: int = 42
) -> DefaultDict[int, Set[int]]:
    """
    Organize documents into LSH buckets using vectorized operations.

    :param signatures: Signature matrix (documents x hash functions).
    :param num_bands: Number of bands for LSH.
    :param seed: Seed value for hash function.
    :return: A dictionary mapping each bucket hash to a set of document IDs.
    """
    num_docs, num_hashes = signatures.shape
    assert (
        num_hashes % num_bands == 0
    ), "Number of hashes must be divisible by num_bands"

    rows_per_band = num_hashes // num_bands
    buckets: DefaultDict[int, Set[int]] = defaultdict(set)

    # Precompute all hashes for each band and each document
    all_band_hashes = np.zeros((num_docs, num_bands), dtype=np.uint64)

    for band in range(num_bands):
        # Efficiently compute the hashes for each band across all documents
        start_idx = band * rows_per_band
        band_signatures = signatures[:, start_idx : start_idx + rows_per_band]
        # hash_numpy_vec
        for doc_id, band_signature in enumerate(band_signatures):
            # Use an efficient hash function such as xxhash
            all_band_hashes[doc_id, band] = xxh32(
                np.append(band_signature, seed + band)
            ).intdigest()

    # Populate the buckets
    for doc_id in range(num_docs):
        for band in range(num_bands):
            buckets[all_band_hashes[doc_id, band]].add(doc_id)

    return buckets

