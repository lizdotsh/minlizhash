from multiprocessing import Pool
from typing import List

import numpy as np
import numpy.typing as npt

from min_hash import HashPartial, gen_signature_matrix


def gen_sig_mat_for_each_mp(
    tokenlist: List[npt.NDArray[np.int32]], hasher: HashPartial
) -> List[npt.NDArray[np.uint64]]:
    """Compute the signature matrix for a list of tokenlists"""
    with Pool() as p:
        sig_matrix = p.starmap(
            gen_signature_matrix, [(tokens, hasher) for tokens in tokenlist]
        )
    return sig_matrix
