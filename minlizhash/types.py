
from typing import Callable, Protocol, TypedDict
import numpy.typing as npt
import numpy as np

TokenArray = npt.NDArray[np.int32]
DocumentSignature = npt.NDArray[np.int64]
PermutationSeeds = npt.NDArray[np.int32]
class DocumentSigner(Protocol):
    hash_function: Callable[[bytes], np.uint64]
    def __call__(self, tokens: TokenArray, seeds: PermutationSeeds) -> DocumentSignature:
        ...

TextPreprocessor = Callable[[str], str]
TextTokenizer = Callable[[str], list[int]]

class Document(TypedDict):
    id: int
    tokens: TokenArray
    signature: DocumentSignature | None

