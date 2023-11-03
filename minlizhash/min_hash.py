# This is not efficient at all. am going to change so it calculates all the hashes at once in one vectorized function call. WIP
from collections import defaultdict
from typing import DefaultDict, List, Set, Callable

import numpy as np
import numpy.typing as npt
from xxhash import xxh32

from . import utils
from .hasher import Hasher, gen_signature_matrix, hash_document
from .types import Document, TokenArray
# from nptyping import NDArray, Structure, Shape, String

# load in mylist.pkl
# with open("my_list.pkl", "rb") as f:
# mylist = pickle.load(f)
# mylist = pickle.load(open("mylist.pkl", "rb"))
#


# I know there is probably a more vectorized way to do this, esp with the bytes, but whatever


def create_document(tokens: TokenArray, id: int) -> Document:
    return {
        "id": id,
        "tokens": np.unique(tokens),
        "signature": None
    }

def create_document_from_raw(raw: str, id: int, tokenizer: Callable[[str], list[int]]) -> Document:
    return {
        "id": id,
        "raw": raw,
        "tokens": np.unique(tokenizer(raw)),
        "signature": None
    }


def sign_document(document: Document, hasher: Hasher) -> Document:
    document["signature"] = hasher.gen_signature(document["tokens"])
    return document


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
