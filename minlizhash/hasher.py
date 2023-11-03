# Anything to do with creation of the hasher object
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
from xxhash import xxh32_intdigest

# from .min_hash import gen_signature_matrix


TokenArray = npt.NDArray[np.int32]
DocumentSignature = npt.NDArray[np.int64]
PermutationSeeds = npt.NDArray[np.int32]
class DocumentSigner(Protocol):
    hash_function: Callable[[bytes], np.uint64]
    def __call__(self, tokens: TokenArray, seeds: PermutationSeeds) -> DocumentSignature:
        ...
    

class Hasher:
    """Creates a Hasher object, which can be used to generate signatures for documents
    Args:
        seeds: seeds to use, ArrayLike. Use load_seeds to load from file or gen_random_seeds to generate random seeds
        document_signer: DocumentSigner object, which is used to generate the signature for a document. 
            Use DocumentSignerMinAfter or DocumentSignerMinBefore.
    
    Returns:
        Hasher: Hasher object
    """
    def __init__(self, seeds: PermutationSeeds, document_signer: DocumentSigner):
        self.seeds = np.ndarray(seeds.shape, dtype=np.int32)
        self.num_permutations = self.seeds.shape[0]
        self.document_signer = document_signer
    
    def gen_signature(self, tokens: TokenArray) -> DocumentSignature:
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
        
        
    
def gen_random_seeds(num_permutations: int) -> npt.NDArray[np.int32]:
    return np.random.randint(0, 10000000, num_permutations)


def load_seeds(filename: str) -> npt.NDArray[np.int32]:
    return np.load(filename)
