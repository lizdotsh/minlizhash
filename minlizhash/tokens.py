from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import tiktoken

from .hasher import Hasher

enc = tiktoken.get_encoding("cl100k_base")
def BPE(s):
    return enc.encode_ordinary(s)

@dataclass
class Document:
    id: int
    raw: str
    tokens: np.ndarray | None = None
    signature: np.ndarray | None = None


def create_document(
    raw: str | npt.NDArray[np.int32],
    id: int,
    preprocessor: Callable[[str], str] | None = None,
    postprocessor: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]] = None,
    hasher: Hasher | None = None,
    tokenizer: Callable[[str], list[int]] = BPE,
) -> Document:
    if preprocessor:
        raw = preprocessor(raw)
    if isinstance(raw, np.ndarray):
        tokens = raw
        raw = ""
    else:
        tokens = np.array(tokenizer(raw))
    if postprocessor:
        tokens = postprocessor(tokens)
    if hasher:
        signature = hasher(tokens)
        return Document(id, raw, tokens=tokens, signature=signature)
    return Document(id, raw, tokens=tokens)


def lst_of_strings_to_documents(
    lst: list[str | npt.NDArray[np.int32]],
    preprocessor: Callable[[str], str] | None = None,
    postprocessor: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]] = None,
    hasher: Hasher | None = None,
) -> list[Document]:
    return [
        create_document(
            raw,
            i,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            hasher=hasher,
        )
        for i, raw in enumerate(lst)
    ]


def sign_document_lst(documentlist: List[Document]):
    for doc in documentlist:
        doc.signature = doc.hasher.gen_hashes(doc.tokens)
