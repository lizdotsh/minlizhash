# Anything to do with creation of the hasher object

from multiprocessing import Pool
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
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

    def __init__(self, hash_function: Callable[[bytes, int], int] = xxh32_intdigest):
        self.hash_function = hash_function
        self._hash_function_vec = np.vectorize(
            hash_function, excluded=["token"], otypes=[np.uint64]
        )

    def __call__(
        self, tokens: TokenArray, seeds: PermutationSeeds
    ) -> DocumentSignature:
        res: npt.NDArray[np.uint64] = np.zeros(
            (tokens.shape[0], len(seeds)), dtype=np.uint64
        )
        for i in range(tokens.shape[0]):
            res[i] = self._hash_function_vec(tokens[i], seeds)
        # get min for each column
        return np.min(res, axis=0)


class DocumentSignerMinBefore(DocumentSigner):
    """Signs the document seed by seed, vectorizing across the tokens and picking the min for each iteration.

    I found this to be faster than DocumentSignerMinAfter,
    but leaving both in (easy drop in replacement if better method found)"""

    def __init__(self, hash_function: Callable[[bytes, int], int] = xxh32_intdigest):
        self.hash_function = hash_function

    # self._hash_function_vec = np.vectorize(hash_function, excluded=['seed'], otypes=[np.uint64])
    def _hash_single_seed(self, tokens: TokenArray, seed: int) -> np.uint64:
        empty: npt.NDArray[np.uint64] = np.empty((tokens.shape[0],), dtype=np.uint64)
        for i in range(tokens.shape[0]):
            empty[i] = self.hash_function(tokens[i].tobytes(), seed)
        return empty.min()

    def __call__(
        self, tokens: TokenArray, seeds: PermutationSeeds
    ) -> DocumentSignature:
        return np.array(
            [self._hash_single_seed(tokens, int(seed)) for seed in seeds],
            dtype=np.uint64,
        )


def gen_random_seeds_from_seed(num_permutations: int, seed: int) -> PermutationSeeds:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10000000, size=(num_permutations, 1), dtype=np.int32)


class Hasher:
    def __init__(
        self,
        seed: int,
        num_permutations: int = 150,
        document_signer: DocumentSigner = DocumentSignerMinBefore(
            hash_function=xxh32_intdigest
        ),
    ):
        """Creates a Hasher object, which can be used to generate signatures for documents
        Args:
            seed: Seed used to generate the seeds for the permutations.
            num_permutations: Number of permutations to use.
            document_signer: DocumentSigner object, which is used to generate the signature for a document.
                Use DocumentSignerMinAfter or DocumentSignerMinBefore.

        Returns
            Hasher: Hasher object
        """
        self.num_permutations = num_permutations
        self.rng_seed = seed
        self.document_signer = document_signer
        self.seeds = Hasher._seeds_from_seed(self.rng_seed, num_permutations)

    def gen_signature(self, tokens: TokenArray) -> DocumentSignature:
        """
        Directly generates a signature for a an array of Tokens (any np array of integers) using the document_signer
        object passed to the Hasher object.

        Args:
            tokens: Array of tokens to generate a signature for

        Returns:
            DocumentSignature: Signature for the tokens (np array of uint64)

        """
        return self.document_signer(tokens, self.seeds)

    def _sign_documents_batch_mp(self, documents: List[Document]) -> List[Document]:
        with Pool() as pool:
            results = pool.starmap(
                Hasher.sign_document, [(self, document) for document in documents]
            )
        return results

    def sign_document(self, document: Document) -> Document:
        """takes a Document object and returns a new document object with the same id and tokens, but with a signature"""
        new_doc = document.copy()
        new_doc["signature"] = self.gen_signature(document["tokens"])
        return new_doc

    def sign_documents(
        self, documents: list[Document], inplace=False, mp=False, progress_bar=True
    ):
        """takes a list of Document objects and returns a new list of Document objects with the same ids and tokens, but with signatures"""
        if progress_bar:
            documents = tqdm(documents)
        if inplace:
            for doc in documents:
                self.sign_document_inplace(doc)
        if mp:
            return self._sign_documents_batch_mp(documents)
        else:
            return [self.sign_document(doc) for doc in documents]

    def sign_document_inplace(self, document: Document) -> None:
        """takes a Document object and adds a signature to it inplace"""
        document["signature"] = self.gen_signature(document["tokens"])

    def __len__(self) -> int:
        return self.num_permutations

    @staticmethod
    def _seeds_from_seed(seed: int, num_permutations: int) -> PermutationSeeds:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 10000000, size=(num_permutations, 1), dtype=np.int32)


def gen_random_seeds(num_permutations: int) -> PermutationSeeds:
    """Unused helper function"""
    return np.random.randint(0, 10000000, num_permutations)


def load_seeds(filename: str) -> PermutationSeeds:
    return np.load(filename)
