from functools import partial
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import tiktoken
from xxhash import xxh32

enc = tiktoken.get_encoding("cl100k_base")
# from nptyping import NDArray, Structure, Shape, String


# I know there is probably a more vectorized way to do this, esp with the bytes, but whatever
def hash_numpy(token: np.uint64, seed: np.uint64) -> np.uint64:
    return xxh32(bytes(token), int(seed)).intdigest()


hash_numpy_vec = np.vectorize(hash_numpy)

HashElm = Callable[[npt.NDArray[np.str_]], np.uint64]

HashSet = List[HashElm]
_test_strings = [
    "The quick brown fox jumps over the lazy dog. The dog was too lazy to react. The fox had a great time. In the distance, a bird chirped. The sun was setting, casting long shadows across the landscape. The fox looked around, its eyes gleaming in the fading light.",
    "The lazy dog decided to sleep all day. The quick brown fox was too busy jumping around to notice. The grass was green and lush, a perfect bed for a lazy dog. The dog yawned and stretched, then settled down for a nap. The fox continued to jump around, oblivious to the dog's slumber.",
    "The fox and the dog are good friends. They spend their days playing and jumping around. The forest was their playground, full of interesting smells and sights. They chased each other through the trees, their laughter echoing through the forest. At the end of the day, they would lie down side by side, content and tired.",
    "The fox and the dog are good friends. hey spend their days playing heir playground, full of interesting smells and sights. They chased each other throughtired.",
    "The fox and the are good friends. hey spend their days playing heir playground, full of interesting smells and sights. They chased each other throughtired.",
]


def hash_element(
    tokens: npt.NDArray[np.uint64],
    hash_gen_vec: Callable[[npt.NDArray, int], np.uint64],
    salt: np.uint64,
) -> HashElm:
    # pad the salt with zeros
    return hash_gen_vec(tokens, int(salt))


def gen_hash_functions(num_hashes: int, hash_element_fn=hash_element) -> HashSet:
    """Generate a list of hash functions"""
    salt = np.random.randint(0, 10000000, num_hashes)
    # salt_strings = np.char.zfill(salt.astype(str), salt_len)[-salt_len:]
    return [partial(hash_element_fn, hash_gen_vec=hash_numpy_vec, salt=s) for s in salt]


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


def _test_gen_hash_functions(test_strings=_test_strings):
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
