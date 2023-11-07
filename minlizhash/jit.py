import struct
from typing import Callable

import numpy as np
import numpy.typing as npt
from numba import jit, njit

from .types import DocumentSignature, DocumentSigner, PermutationSeeds, TokenArray

FNV_32_PRIME = 0x01000193
FNV_64_PRIME = 0x100000001B3

FNV0_32_INIT = 0
FNV0_64_INIT = 0
FNV1_32_INIT = 0x811C9DC5
FNV1_32A_INIT = FNV1_32_INIT
FNV1_64_INIT = 0xCBF29CE484222325
FNV1_64A_INIT = FNV1_64_INIT


@njit
def int32_to_bytes(x):
    return (x >> 24 & 0xFF, x >> 16 & 0xFF, x >> 8 & 0xFF, x & 0xFF)


@njit
def int64_to_bytes(x):
    return (
        x >> 56 & 0xFF,
        x >> 48 & 0xFF,
        x >> 40 & 0xFF,
        x >> 32 & 0xFF,
        x >> 24 & 0xFF,
        x >> 16 & 0xFF,
        x >> 8 & 0xFF,
        x & 0xFF,
    )


@njit
def fnv1a32(data: np.int32, seed: np.int32):
    """
    Returns the 32 bit FNV-1a hash value for the given data.
    """
    # assert isinstance(data, bytes)

    hval = 0x811C9DC5 ^ seed
    for byte in int32_to_bytes(data):
        hval = hval ^ byte
        hval = (hval * 0x01000193) % 2**32
    return np.uint64(hval)


@njit
def fnv1a64(data: np.int32, seed: np.int32):
    """
    Returns the 64 bit FNV-1a hash value for the given data.
    """
    # assert isinstance(data, bytes)

    hval = 0xCBF29CE484222325 ^ seed
    for byte in int64_to_bytes(data):
        hval = hval ^ byte
        hval = (hval * 0x100000001B3) % 2**64
    return hval


@njit
def minhash_jit(
    tokens: npt.NDArray[np.int32],
    hash_seed: np.int32,  # ,
    # hash_fn: Callable[[bytes, int], int] = fnv1a,
) -> np.uint64:
    empty: npt.NDArray[np.uint64] = np.empty((tokens.shape[0],), dtype=np.uint64)
    for i in range(tokens.shape[0]):
        empty[i] = fnv1a32(tokens[i], hash_seed)
    return empty.min()


@njit
def minhash_jit_all(
    tokens: TokenArray,
    seeds: PermutationSeeds,
) -> npt.NDArray[np.uint64]:
    arr = np.empty((seeds.shape[0],), dtype=np.uint64)
    for i in range(seeds.shape[0]):
        arr[i] = minhash_jit(tokens, seeds[i])
    return arr


class DocumentSignerJIT(DocumentSigner):
    """Signs the document token by token, vectorizing across the seeds for each iteration"""

    def __init__(self, hash_function: Callable[[TokenArray, int], int] = minhash_jit):
        self.hash_function = hash_function

    def __call__(self, tokens, seeds) -> npt.NDArray[np.uint64]:
        return minhash_jit_all(tokens.astype(np.uint64), seeds[:, 0])
