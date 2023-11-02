import numpy as np
import farmhash
from dataclasses import dataclass
from typing import Callable, List
from functools import partial


# @dataclass
# class HashElement:
#     hash_gen: Callable[[str], np.uint64]
#     salt: str
#     def calc_hash(self, string: str) -> np.uint64:
#         return self.hash_gen(string + self.salt)
#salt_len: int



HashElm = Callable[[str], np.uint64]

HashSet = List[HashElm]
def hash_element(string, hash_gen: Callable[[str], np.uint64], salt: str) -> HashElm:
    # pad the salt with zeros
    return hash_gen(string + salt)

def gen_hash_functions(num_hashes: int, salt_len: int = 20) -> HashSet:
    salt = np.random.randint(0, 10000000000, num_hashes)
    salt_strings = np.char.zfill(salt.astype(str), salt_len)[-salt_len:]
    return [partial(farmhash.hash64, salt=salt) for salt in salt_strings]
    


# 1. Trying to recreate minhash from scratch
def calcHash(token: np.int64, )



def genSalt(length: int) -> np.int64:
