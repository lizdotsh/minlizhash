from typing import Callable, List, Protocol, TypedDict

import numpy as np
import numpy.typing as npt

TokenArray = npt.NDArray[np.int32]
DocumentSignature = npt.NDArray[np.uint64]
PermutationSeeds = npt.NDArray[np.int32]


class DocumentSigner(Protocol):
    hash_function: Callable[[bytes], np.uint64]

    def __call__(
        self, tokens: TokenArray, seeds: PermutationSeeds
    ) -> DocumentSignature:
        ...


TextPreprocessor = Callable[[str], str]
TextTokenizer = Callable[[str], list[int]]


class Document(TypedDict):
    id: int
    tokens: TokenArray
    signature: DocumentSignature | None


class IndexStorage(Protocol):
    def add(self, band, key: int, value: List[int]):
        """Add item to the LSH index."""
        ...

    def get(self, band, key: int) -> List[int]:
        """Get item from the LSH index."""
        ...

    def save(self, filename: str):
        """Save the LSH index to a file."""
        ...
