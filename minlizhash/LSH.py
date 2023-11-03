# Everything LSH related lives here

from dataclasses import dataclass
from typing import Callable, List, Set, DefaultDict
import numpy as np
import pickle
from collections import defaultdict
from tokens import Document


class LSHIndex:
    def __init__(self, lsh_dict: dict[int, list[int]], seed: int, num_bands: int, num_seeds: int):
        self.lsh_dict = lsh_dict
        self.seed = seed
        self.num_bands = num_bands
        self.num_seeds = num_seeds
        self.buckets = [defaultdict(list) for _ in range(self.num_bands)]
        self.rows_per_band = self.num_seeds // num_bands

    def add(self, document: Document):
        """Add item to the LSH index."""
        if len(document.signature) != self.num_seeds:
            raise ValueError(f"MinHash signature length must be {self.hash_size}")
    
        for bucket in self.buckets:
            self.lsh_dict[bucket].append(document.id)
        for i in range(self.num_bands):
            start_idx = i * self.rows_per_band
            end_idx = (i + 1) * self.rows_per_band
            band = document.signature[start_idx:end_idx]
            hashed_band = self._hash_band(band)
            self.buckets[i][hashed_band].append(document.id)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
  
