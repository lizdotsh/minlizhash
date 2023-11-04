# This is not efficient at all. am going to change so it calculates all the hashes at once in one vectorized function call. WIP
from typing import Callable

import numpy as np

from .hasher import Hasher
from .types import Document, TokenArray

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
