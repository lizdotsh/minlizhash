from multiprocessing import Pool
from typing import List

import jsonlines
import numpy as np
import numpy.typing as npt

from .hasher import Hasher
from .types import Document


def sign_documents_batch_mp(documents: List[Document], hsr: Hasher):
    with Pool() as pool:
        results = pool.starmap(
            Hasher.sign_document, [(hsr, document) for document in documents]
        )
    return results
