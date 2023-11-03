from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import tiktoken

from .hasher import Hasher

import re
import unicodedata
from .types import TokenArray, TextPreprocessor, TextTokenizer

punctuation_pattern = re.compile(r'[^\w\s]')
whitespace_pattern = re.compile(r'\s+')
enc = tiktoken.get_encoding("gpt-2")
def BPE(s: str) -> TokenArray:
    return np.array(enc.encode_ordinary(s), dtype=np.int32)


def preprocessor(s: str) -> str:
    """
    WIP - BASIC
    Preprocesses a string for tokenization
    Turns to lowercase, removes punctuation, normalizes whitespace, normalizes unicode
    """
    s = s.lower()
    s = punctuation_pattern.sub('', s)
    # Normalize whitespace
    s = whitespace_pattern.sub(' ', s).strip()
    
    # Normalize unicode
    s = unicodedata.normalize('NFD', s)

    return "".join([c for c in s if not unicodedata.combining(c)])

def create_ngrams(raw: str,  preprocessor = preprocessor, tokenizer: TextTokenizer = BPE) -> TokenArray:
    """
    WIP - BASIC
    Creates ngrams from a string
    """
    return np.unique(tokenizer(preprocessor(raw)), dtype=np.int32)
        


