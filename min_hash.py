import numpy as np
import numpy.typing as npt
import farmhash
from dataclasses import dataclass
from typing import Callable, List
from functools import partial
#from nptyping import NDArray, Structure, Shape, String


HashElm = Callable[npt.NDArray[str], np.uint64]

HashSet = List[HashElm]



def hash_element(tokens: npt.NDArray[str], hash_gen_vec: Callable[npt.NDArray[str], np.uint64], salt: str) -> HashElm:
    # pad the salt with zeros
    return hash_gen_vec(np.char.add(tokens, salt))


def gen_hash_functions(num_hashes: int, salt_len: int = 20, hash_element_fn = hash_element) -> HashSet:
    """Generate a list of hash functions"""
    salt = np.random.randint(0, 10000000000, num_hashes)
    salt_strings = np.char.zfill(salt.astype(str), salt_len)[-salt_len:]
    return [partial(hash_element_fn, hash_gen_vec=np.vectorize(farmhash.hash64), salt=salt) for salt in salt_strings]




