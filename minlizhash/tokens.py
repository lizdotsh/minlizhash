from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Document:
    id: int
    raw: str
    tokens: np.ndarray | None = None
    signature: np.ndarray | None = None


def create_document(
    raw: str,
    id: int,
    preprocessor: Callable[[str], str] | None = None,
    postprocessor: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]] = None,
    hasher: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int64]] | None = None,
) -> Document:
    if preprocessor:
        raw = preprocessor(raw)
    tokens = np.array(enc.encode_ordinary(raw))
    if postprocessor:
        tokens = postprocessor(tokens)
    if hasher:
        signature = hasher(tokens)
        return Document(id, raw, tokens=tokens, signature=signature)
    return Document(id, raw, tokens=tokens)
