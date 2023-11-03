# This is not efficient at all. am going to change so it calculates all the hashes at once in one vectorized function call. WIP
from collections import defaultdict
from typing import DefaultDict, List, Set

import numpy as np
import numpy.typing as npt
from xxhash import xxh32

from . import utils
from .hasher import Hasher, gen_signature_matrix, hash_document

# from nptyping import NDArray, Structure, Shape, String

# load in mylist.pkl
# with open("my_list.pkl", "rb") as f:
# mylist = pickle.load(f)
# mylist = pickle.load(open("mylist.pkl", "rb"))
#


# I know there is probably a more vectorized way to do this, esp with the bytes, but whatever


def gen_min_hash_for_tokens(tokens: npt.NDArray[np.int32], seed: np.int32) -> np.int64:
    """
    @param tokens: int32 x length of document
    @param seed: seed to use for hash function
    @return: int64 x 1
    """
    hashes = np.empty(tokens.shape[0], dtype=np.int64)
    for i in range(tokens.shape[0]):
        hashes[i] = hash_document(tokens[i].tobytes(), seed)
    return np.min(hashes)


def get_min_for_each_seed(
    tokens: npt.NDArray[np.int32], seeds: npt.NDArray[np.int32]
) -> npt.NDArray[np.int64]:
    """
    @param tokens: int32 x length of document
    @param seeds: int32 x num_seeds
    @return: int64 x num_seeds
    """
    hashes = np.empty(seeds.shape[0], dtype=np.int64)
    for i in range(seeds.shape[0]):
        hashes[i] = gen_min_hash_for_tokens(tokens, seeds[i])
    return hashes


#  return res


def compute_signature(
    tokens: npt.NDArray[np.uint64], hash_functions
) -> npt.NDArray[np.uint64]:
    """Compute the signature for a set of ngrams"""
    return np.array([np.min(h(tokens)) for h in hash_functions], dtype=np.uint64)


# Unused



# get_signature_matrix([enc.encode_ordinary(t) for t in _test_strings], hashfns)


def get_signature_matrix(
    tokenlist: List[npt.NDArray[np.uint64]], hash_functions
) -> npt.NDArray[np.uint64]:
    """[DEPRICATED] Compute the signature matrix for a list of tokenlists"""
    sig_matrix = np.zeros((len(tokenlist), len(hash_functions)), dtype=np.uint64)
    for i, tokens in enumerate(tokenlist):
        sig_matrix[i] = compute_signature(tokens, hash_functions)
    return sig_matrix


# some of this was just gpt4, going to rewrite it better tomorrow
# (to be clear, very, very little over this code generally is gpt4 lol)
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


def hash_band(band: np.ndarray, seed: int = 42) -> int:
    """Hashes a single band (portion of the signature) for LSH."""
    return xxh32(band.tobytes(), seed=int(seed)).intdigest()


def gen_sig_mat_for_each(
    tokenlist: List[npt.NDArray[np.int32]], hasher: Hasher
) -> List[npt.NDArray[np.uint64]]:
    # Preallocate signature matrix array
    sig_matrix = np.empty((len(tokenlist), hasher.seeds.shape[0]), dtype=np.uint64)

    for i, tokens in enumerate(tokenlist):
        sig_matrix[i] = gen_signature_matrix(tokens, hasher)

        if i % 25 == 0:
            print(utils.progress_bar(i, len(tokenlist)))

    return sig_matrix


def gen_sig_mat_for_each_rev(
    tokenlist: List[npt.NDArray[np.int32]], seeds: npt.NDArray[np.int32]
) -> List[npt.NDArray[np.uint64]]:
    # Preallocate signature matrix array
    sig_matrix = np.empty((len(tokenlist), seeds.shape[0]), dtype=np.uint64)

    for i, tokens in enumerate(tokenlist):
        sig_matrix[i] = get_min_for_each_seed(tokens, seeds)

        if i % 25 == 0:
            print(utils.progress_bar(i, len(tokenlist)))

    return sig_matrix
