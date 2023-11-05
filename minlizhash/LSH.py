# Everything LSH related lives here

import pickle
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, DefaultDict, Dict, List, Set, Tuple

import numpy as np
import numpy.typing as npt
from xxhash import xxh32_intdigest

from .hash import Hasher
from .types import LSH, Document, DocumentSignature
from .utils import jaccard_similarity

LSH_Dictionary = List[DefaultDict[int, list[int]]]

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


class LSHIndex(LSH):
    def __init__(
        self,
        num_bands: int,
        num_permutations: int,
        seed: int = 0,
        hash_function: Callable[[bytes, int], int] = xxh32_intdigest,
    ):
        """
        Creates an LSH index using banding (https://en.wikipedia.org/wiki/Locality-sensitive_hashing).
        Very simple algorithm. Just takes the much longer MinHash signature and splits it into bands, then hashes each band.

        Documents that share at least one hash are considered candidates (more likely to be identical) and then manually checked or have their hashes checked later.

        If you already have a hasher object, you can use the static method from_hasher to create an LSHIndex object directly.

        Args:
            num_bands: Number of bands to use. Must be divisible by the number of permutations.
            num_permutations: Number of permutations to use. Must be divisible by the number of bands.
            seed: Seed to use for the hash function.
            hash_function: Hash function to use. Must be a function that takes a bytes object and a seed and returns an integer.

        Returns:
            LSHIndex: LSHIndex object


        """
        if num_permutations % num_bands != 0:
            raise ValueError(
                f"Number of bands must be divisible by the number of permutations: {num_permutations}"
            )
        self.seed = seed
        self._num_bands = num_bands
        self._num_permutations = num_permutations
        self.buckets: LSH_Dictionary = [
            defaultdict(list) for _ in range(self._num_bands)
        ]
        self._rows_per_band = self._num_permutations // num_bands
        self._hash_function = hash_function

    def _get_bands(self, signature: DocumentSignature) -> npt.NDArray[np.uint64]:
        """Split the signature into bands and hash them"""

        for i in range(self._num_bands):
            start_idx = i * self._rows_per_band
            end_idx = (i + 1) * self._rows_per_band
            yield self._hash_function(signature[start_idx:end_idx].tobytes(), self.seed)

    def _signature_to_hashed_bands(
        self, signature: DocumentSignature
    ) -> npt.NDArray[np.uint64]:
        """Split the signature into bands and hash them"""
        return np.array(
            [
                self._hash_function(band.tobytes(), self.seed)
                for band in np.hsplit(signature, self._num_bands)
            ],
            dtype=np.uint64,
        )

    def _add_id_sig(self, id: int, signature: DocumentSignature):
        if len(signature) != self._num_permutations:
            raise ValueError(
                f"MinHash signature length must be {self._num_permutations}"
            )

        for i, band in enumerate(self._get_bands(signature)):
            if id not in self.buckets[i][band]:
                self.buckets[i][band].append(id)
            # self.buckets[i][band].append()

    def add(self, document: Document):
        """Add item to the LSH index."""
        self._add_id_sig(document["id"], document["signature"])

    def add_batch_tuple(self, TupleList: List[Tuple[int, DocumentSignature]]):
        """Add batch of items to the LSH index."""
        for tup in TupleList:
            self._add_id_sig(tup[0], tup[1])

    def add_batch_documentlist(self, DocumentList: List[Document]):
        """Add batch of items to the LSH index."""
        for doc in DocumentList:
            self._add_id_sig(doc["id"], doc["signature"])

    def get_all_candidates(self) -> set[Tuple[int, int]]:
        """Get all candidate pairs from the LSH index."""
        candidates = set()
        for band in self.buckets:
            for bucket in band.values():
                for a, b in combinations(bucket, 2):
                    candidates.add((a, b))
        return candidates

    def save(self, filename: str):
        """Save the LSH index to a pickle."""
        data = {
            "num_bands": self._num_bands,
            "num_permutations": self._num_permutations,
            "seed": self.seed,
            "buckets": self.buckets,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def merge_buckets(self, lsh_dict: LSH_Dictionary):
        """Merge a dictionary into the current LSH index.
        If seed, bands, and permutations are the same, this should work."""
        for i, band in enumerate(lsh_dict):
            for hash, ids in band.items():
                self.buckets[i][hash].extend(ids)
                # make unique
                self.buckets[i][hash] = list(set(self.buckets[i][hash]))

    @staticmethod
    def load(filename: str):
        """Load the LSH index from a pickle"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        lsh = LSHIndex(data["num_bands"], data["num_permutations"], data["seed"])
        lsh.buckets = data["buckets"]
        return lsh

    @staticmethod
    def from_hasher(hasher: Hasher, num_bands: int):
        """Create a LSHIndex_Banding from a Hasher object."""
        return LSHIndex(
            num_bands,
            len(hasher),
            hasher.rng_seed,
            hash_function=hasher.document_signer.hash_function,
        )


class LSHIndex_Projection(LSH):
    """
    Only partly implimented. Do not use.
    Impliments LSH with Random Projection (https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection).


    """

    def __init__(self, seed: int, num_bands: int, hasher: Hasher):
        # Make sure the number of bands is // 32:
        if num_bands % 32 != 0:
            raise ValueError("Number of bands must be divisible by 32")
        self._seed = seed
        self._num_bands = num_bands
        self._rng = np.random.default_rng(seed)
        self._num_permutations = len(hasher)
        self.buckets: List[DefaultDict[Any, List[Any]]] = [
            defaultdict(list) for _ in range(self._num_bands)
        ]
        self.rows_per_band = self._num_permutations // num_bands
        self._projection_array = self._projections()
        self._index_hashes: npt.NDArray[np.uint32] = np.empty(
            (0, self._num_bands), dtype=np.uint32
        )

    def _projections(self):
        """Use Normal: https://stackoverflow.com/questions/59954810/generate-random-points-on-10-dimensional-unit-sphere"""
        projections = self._rng.normal(size=(self._num_permutations, self._num_bands))
        return projections / np.linalg.norm(projections, axis=1, keepdims=True)

    def _index(self, query: npt.NDArray[np.uint64]):
        """Index np array of queries (shape (num_queries, num_permutations))"""
        hashes = np.packbits(query.dot(self._projections) >= 0, bitorder="little").view(
            np.uint32
        )
        self._index_hashes = np.vstack(self._index_hashes, hashes)

    @staticmethod
    def _bitcount_64(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
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


def check_candidatelist(
    candidatelist: List[Tuple[int, int]],
    documentlist: List[Document] | Dict[Any, Document],
    exact=False,
) -> List[Tuple[int, int, float]]:
    """Check the candidates and return the actual Jaccard similarity.
    ID _MUST_ be the index in the documentlist.
    Alternatively, pass in a dictionary with the ID as the key.

    Args:
        documentlist: List of documents
        candidatelist: List of candidate pairs
        exact: If True, compute the exact Jaccard similarity. If False, use the MinHash signature
        filter_if_below (float): If the Jaccard similarity is below this value, filter it of the resulting list.
    Returns:
        List of tuples (a, b, jaccard_similarity)
    """
    key = "signature" if exact else "tokens"
    return [
        (
            a,
            b,
            jaccard_similarity(documentlist[a][key], documentlist[b][key]),
        )
        for a, b in candidatelist
    ]


def filter_checked_candidates(
    checked_candidates: List[Tuple[int, int, float]],
    filter_if_below: float = 0.0,
    leave_one: bool = False,
) -> set[int | str]:
    """
    Input a list of tuples (a, b, jaccard_similarity) and filter out the ones below the threshold.
    returns a set of document IDs where a duplicate exists.

    If delete_both is True, then if a duplicate is found, both documents are deleted from the set.
    Otherwise, only the second document is deleted. (leaving 1 copy of the duplicate)

    """
    deleteset: Set[Any] = set()
    if leave_one:
        deleteset = deleteset.union(
            {
                a
                for a, b, jaccard_similarity in checked_candidates
                if jaccard_similarity >= filter_if_below
            }
        )
    return deleteset.union(
        {
            b
            for a, b, jaccard_similarity in checked_candidates
            if jaccard_similarity >= filter_if_below
        }
    )
