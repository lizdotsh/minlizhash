# This is not efficient at all. am going to change so it calculates all the hashes at once in one vectorized function call. WIP
from typing import Callable

import jsonlines
import numpy as np

from .hasher import Hasher
from .types import Document, List, TokenArray
from .utils import document_np_to_list

# from nptyping import NDArray, Structure, Shape, String

# load in mylist.pkl
# with open("my_list.pkl", "rb") as f:
# mylist = pickle.load(f)
# mylist = pickle.load(open("mylist.pkl", "rb"))
#


# I know there is probably a more vectorized way to do this, esp with the bytes, but whatever


def create_document(tokens: TokenArray, id: int) -> Document:
    return {"id": id, "tokens": np.unique(tokens), "signature": None}


def create_document_from_raw(
    raw: str, id: int, tokenizer: Callable[[str], list[int]]
) -> Document:
    return {
        "id": id,
        "raw": raw,
        "tokens": np.unique(tokenizer(raw)),
        "signature": None,
    }


def sign_document(document: Document, hasher: Hasher) -> Document:
    document["signature"] = hasher.gen_signature(document["tokens"])
    return document


def jsonl_to_documentlist(jsonl: str) -> List[Document]:
    """Convert a jsonl file to a list of documents"""
    with jsonlines.open(jsonl) as reader:
        return [
            {
                "id": doc["id"],
                "tokens": np.array(doc["tokens"]),
                "signature": np.array(doc["signature"])
                if doc["signature"] is not None
                else None,
            }
            for doc in reader
        ]


def documentlist_to_jsonl(path: str, documentlist: List[Document]) -> None:
    """Convert a list of documents to a jsonl file"""
    with jsonlines.open(path, mode="w") as writer:
        for document in documentlist:
            writer.write(document_np_to_list(document))
