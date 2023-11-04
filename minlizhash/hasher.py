# Anything to do with creation of the hasher object
from typing import Callable

import numpy as np
import numpy.typing as npt
from xxhash import xxh32_intdigest

from .types import (
    Document,
    DocumentSignature,
    DocumentSigner,
    PermutationSeeds,
    TokenArray,
)

# from .min_hash import gen_signature_matrix


class DocumentSignerMinAfter(DocumentSigner):
    """Signs the document token by token, vectorizing across the seeds for each iteration"""

    def __init__(self, hash_function: Callable[[bytes], np.uint64] = xxh32_intdigest):
        self.hash_function = hash_function
        self._hash_function_vec = np.vectorize(
            hash_function, excluded=["token"], otypes=[np.uint64]
        )

    def __call__(
        self, tokens: TokenArray, seeds: PermutationSeeds
    ) -> DocumentSignature:
        res = np.zeros((tokens.shape[0], len(seeds)), dtype=np.uint64)
        for i in range(tokens.shape[0]):
            res[i] = self._hash_function_vec(tokens[i], seeds)
        # get min for each column
        return np.min(res, axis=0)


class DocumentSignerMinBefore(DocumentSigner):
    """Signs the document seed by seed, vectorizing across the tokens and picking the min for each iteration"""

    def __init__(self, hash_function: Callable[[bytes], np.uint64] = xxh32_intdigest):
        self.hash_function = hash_function

    # self._hash_function_vec = np.vectorize(hash_function, excluded=['seed'], otypes=[np.uint64])
    def _hash_single_seed(self, tokens: TokenArray, seed: np.int32) -> np.uint64:
        return np.min([self.hash_function(token.tobytes(), seed) for token in tokens])

    def __call__(
        self, tokens: TokenArray, seeds: PermutationSeeds
    ) -> DocumentSignature:
        return np.array(
            [self._hash_single_seed(tokens, seed) for seed in seeds], dtype=np.uint64
        )


def gen_random_seeds_from_seed(num_permutations: int, seed: int) -> PermutationSeeds:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10000000, size=(num_permutations, 1), dtype=np.int32)


class Hasher:
    """Creates a Hasher object, which can be used to generate signatures for documents
    Args:
        seeds: seeds to use, ArrayLike. Use load_seeds to load from file or gen_random_seeds to generate random seeds
        document_signer: DocumentSigner object, which is used to generate the signature for a document.
            Use DocumentSignerMinAfter or DocumentSignerMinBefore.

    Returns
        Hasher: Hasher object
    """

    def __init__(
        self,
        num_permutations: int = 150,
        seed: int | None = None,
        document_signer: DocumentSigner = DocumentSignerMinBefore(
            hash_function=xxh32_intdigest
        ),
    ):
        self.num_permutations = num_permutations
        self.rng_seed = seed
        self.document_signer = document_signer
        self.seeds = Hasher._seeds_from_seed(self.rng_seed, num_permutations)

    def gen_signature(self, tokens: TokenArray) -> DocumentSignature:
        return self.document_signer(tokens, self.seeds)

    def sign_document(self, document: Document) -> Document:
        document["signature"] = self.gen_signature(document["tokens"])
        return document

    def save(self, filename: str) -> None:
        np.save(filename, self.seeds)

    @staticmethod
    def _seeds_from_seed(seed: int, num_permutations: int) -> PermutationSeeds:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 10000000, size=(num_permutations, 1), dtype=np.int32)

    def __len__(self) -> int:
        return self.num_permutations


def gen_random_seeds(num_permutations: int) -> PermutationSeeds:
    return np.random.randint(0, 10000000, num_permutations)


def load_seeds(filename: str) -> PermutationSeeds:
    return np.load(filename)
