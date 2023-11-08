import struct
from typing import Callable

import numpy as np
import numpy.typing as npt
from numba import njit

from .types import DocumentSignature, DocumentSigner, PermutationSeeds, TokenArray


@njit
def int32_to_bytes(x: np.int32) -> tuple:
    return (x >> 24 & 0xFF, x >> 16 & 0xFF, x >> 8 & 0xFF, x & 0xFF)


@njit
def fnv1a32(num: np.int32, seed: np.int32) -> np.uint64:
    """
    Returns the 32 bit FNV-1a hash value for the given number.
    """

    hval = 0x811C9DC5 ^ seed
    for byte in int32_to_bytes(num):
        hval = hval ^ byte
        hval = (hval * 0x01000193) % 2**32
    return np.uint64(hval)


@njit
def fnv1a32_bytes(data: bytes, seed: np.int32) -> np.uint64:
    """
    Returns the 32 bit FNV-1a hash value for the given number.
    """

    hval = 0x811C9DC5 ^ seed
    for byte in data:
        hval = hval ^ byte
        hval = (hval * 0x01000193) % 2**32
    return np.uint64(hval)


@njit
def minhash_jit(tokens: npt.NDArray[np.int32], hash_seed: np.int32) -> np.uint64:
    empty: npt.NDArray[np.uint64] = np.empty((tokens.shape[0],), dtype=np.uint64)
    for i in range(tokens.shape[0]):
        empty[i] = fnv1a32(tokens[i], hash_seed)
    return empty.min()


@njit
def sign_tokens_jit(
    tokens: TokenArray,
    seeds: PermutationSeeds,
) -> npt.NDArray[np.uint64]:
    arr = np.empty((seeds.shape[0],), dtype=np.uint64)
    for i in range(seeds.shape[0]):
        arr[i] = minhash_jit(tokens, seeds[i])
    return arr


class DocumentSignerJIT(DocumentSigner):
    """Signs the document token by token, vectorizing across the seeds for each iteration.
    hash function selection exclusively applies to what is used by external callers (in this case, LSH).
    minhash itself works with home grown fnv1a32 implementation.
    """

    def __init__(self, hash_function_lsh: Callable[[bytes, int], int] = fnv1a32_bytes):
        self.hash_function = hash_function_lsh

    def __call__(self, tokens, seeds) -> DocumentSignature:
        return sign_tokens_jit(tokens.astype(np.uint64), seeds[:, 0])
