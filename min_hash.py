# This is not efficient at all. am going to change so it calculates all the hashes at once in one vectorized function call. WIP
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, DefaultDict, List, Set

import numpy as np
import numpy.typing as npt
import tiktoken
from xxhash import xxh32, xxh32_intdigest

import utils as utils

enc = tiktoken.get_encoding("cl100k_base")
# from nptyping import NDArray, Structure, Shape, String

# load in mylist.pkl
with open("my_list.pkl", "rb") as f:
    mylist = pickle.load(f)
# mylist = pickle.load(open("mylist.pkl", "rb"))
#


# I know there is probably a more vectorized way to do this, esp with the bytes, but whatever
def hash_document(tokenbyte: bytes, seed: np.int32) -> np.int64:
    return xxh32_intdigest(tokenbyte, seed)


hash_document_with_seeds = np.vectorize(
    hash_document, excluded=["token"], otypes=[np.int64]
)


class HashPartial_oop:
    """
    Partial function with seeds baked in. Gen_hashes returns a 1 x __len__ array of hashes for each seed
    """

    def __init__(
        self,
        num_seeds: int,
        hash_fn_vec: Callable[
            [np.int32, npt.NDArray[np.int32]], npt.NDArray[np.int64]
        ] = hash_document_with_seeds,
    ):
        self.seeds = np.random.randint(0, 10000000, num_seeds)
        self.gen_hashes = partial(hash_fn_vec, seed=self.seeds)

    def __len__(self) -> int:
        return self.seeds.shape[0]


@dataclass
class HashPartial:
    """
    Partial function with seeds baked in. Gen_hashes returns a 1 x __len__ array of hashes for each seed
    """

    seeds: npt.NDArray[np.int32]
    gen_hashes: Callable[[np.int32], npt.NDArray[np.int64]]

    def __len__(self) -> int:
        return self.seeds.shape[0]


def gen_random_seeds(length: int) -> npt.NDArray[np.int32]:
    return np.random.randint(0, 10000000, length)


def gen_hash_partial(
    num_seeds: int,
    hash_fn_vec: Callable[
        [np.int32, npt.NDArray[np.int32]], npt.NDArray[np.int64]
    ] = hash_document_with_seeds,
) -> HashPartial:
    """
    If custom seeds is set, ignores num_seeds and uses the custom seeds instead.
    @param num_seeds: number of seeds to generate
    @param hash_fn_vec: hash function to use
    @return: HashPartial object
    """
    seeds = gen_random_seeds(num_seeds)
    return HashPartial(seeds, partial(hash_fn_vec, seed=seeds))


def restore_hash_partial(
    seeds: npt.ArrayLike,
    hash_fn_vec: Callable[
        [np.int32, npt.NDArray[np.int32]], npt.NDArray[np.int64]
    ] = hash_document_with_seeds,
) -> HashPartial:
    """
    @param seeds: seeds to use, ArrayLike
    @param hash_fn_vec: hash function to use
    @return: HashPartial object
    """
    return HashPartial(np.array(seeds), partial(hash_fn_vec, seed=seeds))


def gen_signature_matrix(
    tokens: npt.NDArray[np.int32], hasher: HashPartial
) -> npt.NDArray[np.int64]:
    """
    @param tokens: int32 x length of document
    @param hasher: HashPartial type. Outputs an array of hashes for each seed
    @return: int64 x num_hashes
    """
    res = np.zeros((tokens.shape[0], len(hasher)), dtype=np.int32)
    for i in range(tokens.shape[0]):
        res[i] = hasher.gen_hashes(tokens[i])
    return res


def compute_signature(
    tokens: npt.NDArray[np.uint64], hash_functions: HashSet
) -> npt.NDArray[np.uint64]:
    """Compute the signature for a set of ngrams"""
    return np.array([np.min(h(tokens)) for h in hash_functions], dtype=np.uint64)


# Unused
def simple_shingles_generator(text: str, k: int = 3) -> npt.NDArray[np.str_]:
    """Generate shingles of length k from the text"""
    return np.fromiter(
        (text[i : i + k] for i in range(len(text) - k + 1)), dtype="<U" + str(k)
    )


def cosine_similarity(a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]) -> float:
    """Compute the cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def jaccard_similarity(a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]) -> float:
    """Compute the Jaccard similarity between two vectors"""
    return np.intersect1d(a, b).size / np.union1d(a, b).size


def _test_gen_hash_functions(test_strings):
    """Test the hash function generator"""
    hash_functions = gen_hash_functions(20, 10)
    sigs = []
    tokenlist = []
    for elm in test_strings:
        tokens = np.array(enc.encode_ordinary(elm), dtype=np.uint64)
        tokenlist.append(tokens)
        print(tokens)
        sigs.append(compute_signature(tokens, hash_functions))

    # Compare jaccard similarity between all pairs of documents
    for i in range(len(test_strings)):
        for j in range(i + 1, len(test_strings)):
            similarity = jaccard_similarity(tokenlist[i], tokenlist[j])
            print(
                f"Jaccard similarity between document {i} and document {j}: {similarity}"
            )
            similarity_sig = jaccard_similarity(sigs[i], sigs[j])
            print(f"Jaccard similarity between sig {i} and sig {j}: {similarity_sig}")


# get_signature_matrix([enc.encode_ordinary(t) for t in _test_strings], hashfns)


def get_signature_matrix(
    tokenlist: List[npt.NDArray[np.uint64]],
    hash_functions: HashSet,
) -> npt.NDArray[np.uint64]:
    """Compute the signature matrix for a list of tokenlists"""
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
    tokenlist: List[npt.NDArray[np.int32]], hasher: HashPartial
) -> List[npt.NDArray[np.uint64]]:
    """Compute the signature matrix for a list of tokenlists"""
    sig_matrix = []
    for i, tokens in enumerate(tokenlist):
        if i % 25 == 0:
            print(utils.progress_bar(i, len(tokenlist)))
        sig_matrix.append(gen_signature_matrix(tokens, hasher))
    return sig_matrix
