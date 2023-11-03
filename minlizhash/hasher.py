# Anything to do with creation of the hasher object
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from xxhash import xxh32_intdigest

# from .min_hash import gen_signature_matrix


def hash_document(tokenbyte: bytes, seed: np.int32) -> np.int64:
    return xxh32_intdigest(tokenbyte, seed)


hash_document_with_seeds = np.vectorize(
    hash_document, excluded=["token"], otypes=[np.int64]
)

hash_tokens_with_seed = np.vectorize(
    hash_document, excluded=["seed"], otypes=[np.int64]
)


def gen_signature_matrix(
    tokens: npt.NDArray[np.int32], hasher
) -> npt.NDArray[np.int64]:
    """
    @param tokens: int32 x length of document
    @param hasher: hasher.Hasher type. Outputs an array of hashes for each seed
    @return: int64 x num_hashes
    """
    res = np.zeros((tokens.shape[0], len(hasher)), dtype=np.int32)
    for i in range(tokens.shape[0]):
        res[i] = hasher.gen_hashes(tokens[i])
    # get min for each column
    return np.min(res, axis=0)


@dataclass
class Hasher:
    """
    Partial function with seeds baked in. Gen_hashes returns a 1 x __len__ array of hashes for each seed
    """

    seeds: npt.NDArray[np.int32]
    gen_hashes: Callable[[np.int32], npt.NDArray[np.int64]]
    sign: Callable[
        [npt.NDArray[np.int32]], npt.NDArray[np.int64]
    ] = gen_signature_matrix

    def __len__(self) -> int:
        return self.seeds.shape[0]

    def save(self, filename: str) -> None:
        np.save(filename, self.seeds)


def _gen_random_seeds(length: int) -> npt.NDArray[np.int32]:
    return np.random.randint(0, 10000000, length)


def create_hasher(
    num_seeds: int,
    hash_fn_vec: Callable[
        [np.int32, npt.NDArray[np.int32]], npt.NDArray[np.int64]
    ] = hash_document_with_seeds,
) -> Hasher:
    """
    If custom seeds is set, ignores num_seeds and uses the custom seeds instead.
    @param num_seeds: number of seeds to generate
    @param hash_fn_vec: hash function to use
    @return: HashPartial object
    """
    seeds = _gen_random_seeds(num_seeds)
    return Hasher(seeds, partial(hash_fn_vec, seed=seeds))


def restore_hasher_from_seeds(
    seeds: npt.ArrayLike,
    hash_fn_vec: Callable[
        [np.int32, npt.NDArray[np.int32]], npt.NDArray[np.int64]
    ] = hash_document_with_seeds,
) -> Hasher:
    """
    use hasher.load_seeds to load seeds from file
    @param seeds: seeds to use, ArrayLike
    @param hash_fn_vec: hash function to use
    @return: HashPartial object
    """
    return Hasher(np.array(seeds), partial(hash_fn_vec, seed=seeds))


def restore_hasher_from_file(filename: str) -> Hasher:
    return Hasher(
        _load_seeds(filename),
        partial(hash_document_with_seeds, seed=_load_seeds(filename)),
    )


def _load_seeds(filename: str) -> npt.NDArray[np.int32]:
    return np.load(filename)
