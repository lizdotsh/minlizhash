# Everything LSH related lives here

import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, List, Protocol, Set

import numpy as np
import numpy.typing as npt

from .hasher import Hasher
from .types import Document, DocumentSignature

LSH_Dictionary = DefaultDict[int, list[int]]

# Save masks to avoid recomputing
# Naive bitcount implementation without popcount

m1 = np.uint64(0x5555555555555555)
m2 = np.uint64(0x3333333333333333)
m3 = np.uint64(0x0F0F0F0F0F0F0F0F)
m4 = np.uint64(0x0101010101010101)

mask = np.uint64(-1)
# TODO - precompute type specific hashes
s55 = np.uint64(m1 & mask)  # Add more digits for 128bit support
s33 = np.uint64(m2 & mask)
s0F = np.uint64(m3 & mask)
s01 = np.uint64(m4 & mask)
num_bytes_64 = 8


class LSH(Protocol):
    def add(self, document: Document):
        """Add item to the LSH index."""
        ...

    def save(self, filename: str):
        """Save the LSH index to a file."""
        ...


class LSHIndex(LSH):
    def __init__(
        self, lsh_dict: LSH_Dictionary, seed: int, num_bands: int, num_seeds: int
    ):
        self.lsh_dict = lsh_dict
        self._seed = seed
        self._num_bands = num_bands
        self.num_seeds = num_seeds
        self.buckets = [defaultdict(list) for _ in range(self.num_bands)]
        self.rows_per_band = self.num_seeds // num_bands

    def add(self, document: Document):
        """Add item to the LSH index."""
        if len(document.signature) != self.num_seeds:
            raise ValueError(f"MinHash signature length must be {self.hash_size}")

        for bucket in self.buckets:
            self.lsh_dict[bucket].append(document.id)
        for i in range(self.num_bands):
            start_idx = i * self.rows_per_band
            end_idx = (i + 1) * self.rows_per_band
            band = document.signature[start_idx:end_idx]
            hashed_band = self._hash_band(band)
            self.buckets[i][hashed_band].append(document.id)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


class LSHIdx(LSH):
    def __init__(self, seed: int, num_bands: int, hasher: Hasher):
        # Make sure the number of bands is // 32:
        if num_bands % 32 != 0:
            raise ValueError("Number of bands must be divisible by 32")
        self._seed = seed
        self._num_bands = num_bands
        self._rng = np.random.default_rng(seed)
        self._num_permutations = len(hasher)
        self.buckets = [defaultdict(list) for _ in range(self.num_bands)]
        self.rows_per_band = self.num_seeds // num_bands
        self._projections = self._projections()

    def _projections(self):
        """Use Normal: https://stackoverflow.com/questions/59954810/generate-random-points-on-10-dimensional-unit-sphere"""
        projections = self._rng.normal(size=(self._permutations, self._num_bands))
        return projections / np.linalg.norm(projections, axis=1, keepdims=True)

    def _index(self, query: npt.NDArray[npt.uint64]):
        """Index np array of queries (shape (num_queries, num_permutations))"""
        hashes = np.packbits(query.dot(self._projections) >= 0, bitorder="little").view(
            np.uint32
        )
        self._index_hashes = np.vstack(self._index_hashes, hashes)

    @staticmethod
    def _bitcount_64(arr: npt.NPArray[np.uint64]) -> npt.NDArray[np.uint64]:
        """Count bits per element in a 64 bit array.

        Stolen from: https://github.com/softwaredoug/np-sims/blob/main/np_sims/hamming.py
        """
        # Inplace, (ie -=) probably fastesr
        arr -= (arr >> 1) & s55
        arr = (arr & s33) + ((arr >> 2) & s33)

        # Using inplace operators where possible
        arr += arr >> 4
        arr &= s0F
        arr *= s01
        arr >>= 8 * (num_bytes_64 - 1)

        return arr

    #         return arr.sum(axis=1)

    @staticmethod
    def _hamming_distance(self, a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]):
        """Compute Hamming distance between two arrays of 64 bit integers."""
        return self._bitcount_64(a ^ b)
