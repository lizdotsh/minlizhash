from typing import (
    Callable,
    DefaultDict,
    List,
    NotRequired,
    Protocol,
    Tuple,
    TypeAlias,
    TypedDict,
)

import numpy as np
import numpy.typing as npt

TokenArray: TypeAlias = npt.NDArray[np.int32]
DocumentSignature: TypeAlias = npt.NDArray[np.uint64]
PermutationSeeds: TypeAlias = npt.NDArray[np.int32]


class DocumentSigner(Protocol):
    hash_function: Callable[[bytes, int], int]

    def __call__(
        self, tokens: TokenArray, seeds: PermutationSeeds
    ) -> DocumentSignature:
        ...


TextPreprocessor: TypeAlias = Callable[[str], str]
TextTokenizer: TypeAlias = Callable[[str], npt.NDArray[np.int32]]


class Document(TypedDict):
    id: int
    tokens: TokenArray
    signature: NotRequired[DocumentSignature]


class LSH(Protocol):
    buckets: List[DefaultDict[int | str, List[int]]]

    def add(self, document: Document):
        """Add item to the LSH index."""
        ...

    def add_batch_tuple(self, TupleList: List[Tuple[int, DocumentSignature]]):
        """Add batch of items to the LSH index."""
        ...

    def add_batch_documentlist(self, DocumentList: List[Document]):
        """Add batch of items to the LSH index."""
        ...

    def get_all_candidates(self) -> set[Tuple[int, int]]:
        """Get all candidate pairs from the LSH index."""
        ...

    def save(self, filename: str):
        """Save the LSH index to a file."""
        ...


# class IndexStorage(Protocol):
#     def add(self, band, key: int, value: List[int]):
#         """Add item to the LSH index."""
#         ...

#     def get(self, band, key: int) -> List[int]:
#         """Get item from the LSH index."""
#         ...

#     def save(self, filename: str):
#         """Save the LSH index to a file."""
#         ...
