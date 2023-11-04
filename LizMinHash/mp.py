from multiprocessing import Pool
from typing import List

import jsonlines
import numpy as np
import numpy.typing as npt

from .hasher import Hasher
from .types import Document

# def sign_documentlist_mp(document_list: Document, hasher: Hasher) -> List[Document]:
#     """Compute the signature matrix for a list of documents"""
#     with Pool() as p:
#         signature_matrix = p.starmap(hasher.sign_document, document_list)


# def mp_to_from_jsonlines(jsonl: )

# def mp_to_from_jsonlines(jsonl: )
