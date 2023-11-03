# Anything to do with creation of the hasher object
from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
from xxhash import xxh32_intdigest

# from .min_hash import gen_signature_matrix


TokenArray = npt.NDArray[np.int32]
DocumentSignature = npt.NDArray[np.int64]
PermutationSeeds = npt.NDArray[np.int32]
#DocumentSigner = Callable[[TokenArray, PermutationSeeds], DocumentSignature]
class DocumentSigner(Protocol):
    hash_function: Callable[[bytes], np.uint64]
    def __call__(self, tokens: TokenArray, seeds: PermutationSeeds) -> DocumentSignature:
        ...
    

class Hasher:
    def __init__(self, seeds: PermutationSeeds, document_signer: DocumentSigner):
        self.seeds = np.ndarray(seeds.shape, dtype=np.int32)
        self.num_permutations = self.seeds.shape[0]
        self.document_signer = document_signer
    
    def gen_hashes(self, tokens: TokenArray) -> DocumentSignature:
        return self.document_signer(tokens, self.seeds)
    
    def save(self, filename: str) -> None:
        np.save(filename, self.seeds)
    
    def __len__(self) -> int:
        return self.num_permutations



class DocumentSignerMinAfter(DocumentSigner):
    """Signs the document token by token, vectorizing across the seeds for each iteration"""
    def __init__(self, hash_function: Callable[[bytes], np.uint64] = xxh32_intdigest):
        self.hash_function = hash_function
        self._hash_function_vec = np.vectorize(hash_function, excluded=['token'], otypes=[np.uint64])
    
    def __call__(self, tokens: TokenArray, seeds: PermutationSeeds) -> DocumentSignature:
        res = np.zeros((tokens.shape[0], len(seeds)), dtype=np.uint64)
        for i in range(tokens.shape[0]):
            res[i] = self._hash_function_vec(tokens[i])
        # get min for each column
        return np.min(res, axis=0)
    
    
class DocumentSignerMinBefore(DocumentSigner):
    """Signs the document seed by seed, vectorizing across the tokens and picking the min for each iteration"""
    def __init__(self, hash_function: Callable[[bytes], np.uint64] = xxh32_intdigest):
        self.hash_function = hash_function
       # self._hash_function_vec = np.vectorize(hash_function, excluded=['seed'], otypes=[np.uint64])
    def _hash_single_seed(self, tokens: TokenArray, seed: np.int32) -> np.uint64:
        return np.min([self.hash_function(token.tobytes(), seed) for token in tokens])

    def __call__(self, tokens: TokenArray, seeds: PermutationSeeds) -> DocumentSignature:
        return np.array([self._hash_single_seed(tokens, seed) for seed in seeds], dtype=np.uint64)
        
        
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
# class Hasher(Protocol):
#     num_permutations: int
#     seeds: npt.NDArray[np.int32]

#     def __len__(self) -> int:
#         ...
    
#     def gen_hashes(self, tokens: TokenArray) -> DocumentSignature:
#         ...
#     def save(self, filename: str) -> None:
#         ...

# class HasherMinAfter:
#     def __init__(self, seeds: npt.NDArray[np.int32], document_signer: DocumentSigner):
        
#         self.seeds = np.ndarray(seeds.shape, dtype=np.int32)
#         self.num_permutations = self.seeds.shape[0]
    
    
#     def gen_hashes(self, tokens: TokenArray) -> DocumentSignature:
#         return gen_signature_matrix(tokens, self.seeds)
    
#     def save(self, filename: str) -> None:
#         np.save(filename, self.seeds)
    
#     def __len__(self) -> int:
#         return self.num_permutations
    



def hash_document(tokenbyte: bytes, seed: np.int32) -> np.int64:
    return xxh32_intdigest(tokenbyte, seed)


hash_document_with_seeds = np.vectorize(
    hash_document, excluded=["token"], otypes=[np.int64]
)

hash_tokens_with_seed = np.vectorize(
    hash_document, excluded=["seed"], otypes=[np.int64]
)


def gen_signature_matrix(
    tokens: TokenArray, hasher: Hasher
) -> DocumentSignature:
    """
    @param tokens: int32 x length of document
    @param hasher: hasher.Hasher type. Outputs an array of hashes for each seed
    @return: int64 x num_hashes
    """
    


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

    Args:
        seeds: seeds to use, ArrayLike
        hash_fn_vec: hash function to use
        
    Returns: 
        Hasher: Hasher object
    """
    return Hasher(np.array(seeds), partial(hash_fn_vec, seed=seeds))


def restore_hasher_from_file(filename: str) -> Hasher:
    return Hasher(
        _load_seeds(filename),
        partial(hash_document_with_seeds, seed=_load_seeds(filename)),
    )


def _load_seeds(filename: str) -> npt.NDArray[np.int32]:
    return np.load(filename)
