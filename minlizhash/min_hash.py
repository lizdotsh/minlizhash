# This is not efficient at all. am going to change so it calculates all the hashes at once in one vectorized function call. WIP
from typing import Callable, List, Union

import jsonlines
import numpy as np
from xxhash import xxh32_intdigest

from .hasher import DocumentSignerMinBefore, Hasher
from .LSH import LSHIndex, check_candidatelist, filter_checked_candidates
from .types import LSH, Document, TokenArray
from .utils import document_np_to_list


def create_document(tokens: TokenArray, id: int) -> Document:
    return {"id": id, "tokens": np.unique(tokens)}


def create_document_from_raw(
    raw: str, id: int, tokenizer: Callable[[str], list[int]]
) -> Document:
    return {
        "id": id,
        "tokens": np.unique(tokenizer(raw)),
    }


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


def filter_documentlist(
    documentlist: List[Document],
    seed: int,
    num_permutations: int = 128,
    num_bands: int = 32,
    mp=False,
    check_candidates_exactly: bool = False,
    filter_below: float = 0.0,
    leave_one_for_each_duplicate: bool = False,
    existing_index: Union[None, LSH, str] = None,
    index_save_dir: str | None = None,
    save_matches_dir: str | None = None,
    hash_function: Callable[[bytes, int], int] = xxh32_intdigest,
) -> List[Document]:
    """Takes a list of documents and returns a new list of documents with signatures
    Args:
        documentlist: List of documents to be signed
        seed: Seed used to generate the seeds for the permutations.
        num_permutations: Number of permutations to use.
        num_bands: Number of bands to use for LSH. Default is 20
        mp: Whether to use multiprocessing. Default is False
        check_candidates_exactly: Whether to check candidates or use minhash signature. Default is False
        filter_below: Threshold Jaccard for filtering candidates. Default is 0.0. Higher will increase result set (more false positives)
        leave_one_for_each_duplicate: Whether to leave one remaining for each duplicate. Default is False
        existing_index: Existing index to use. Default is None. If string, loads index from file. If LSH, uses that index.
        index_save_dir: Directory to save index to. Default is None
        hash_function: Hash function to use. Default is xxh32_intdigest

    Returns:
        List of documents with signatures
    """
    hsr = Hasher(
        seed=seed,
        num_permutations=num_permutations,
        document_signer=DocumentSignerMinBefore(hash_function=hash_function),
    )
    processed_documents = hsr.sign_documents_batch(documentlist, inplace=False, mp=mp)

    index = LSHIndex.from_hasher(hsr, num_bands)
    if existing_index is not None:
        if isinstance(existing_index, str):
            index.merge_buckets(LSHIndex.load(existing_index).buckets)
        else:
            index.merge_buckets(existing_index.buckets)

    index.add_batch_documentlist(processed_documents)

    if index_save_dir is not None:
        index.save(index_save_dir)
    checked_candidatelist = check_candidatelist(
        index.get_all_candidates(),
        processed_documents,
        exact=check_candidates_exactly,
    )
    filtered_checked_candidates = filter_checked_candidates(
        checked_candidatelist, filter_below, leave_one=leave_one_for_each_duplicate
    )
    if save_matches_dir is not None:
        with jsonlines.open(save_matches_dir, mode="w") as writer:
            for tup in checked_candidatelist:
                writer.write({"id1": tup[0], "id2": tup[1], "jaccard": tup[2]})

    return [
        doc
        for doc in processed_documents
        if doc["id"] not in filtered_checked_candidates
    ]


def filter_jsonl(
    input_file: str,
    output_file: str,
    seed: int,
    num_permutations: int = 150,
    num_bands: int = 20,
    mp=False,
    check_candidates_exactly: bool = False,
    filter_below: float = 0.0,
    leave_one_for_each_duplicate: bool = False,
    existing_index: Union[None, LSH, str] = None,
    index_save_dir: str | None = None,
    save_matches_dir: str | None = None,
    hash_function: Callable[[bytes, int], int] = xxh32_intdigest,
) -> None:
    """Takes a jsonl file and returns a new jsonl file with signatures
    May update later to batch load or similar. If you need more customization, all the APIs are internally available.
    Args:
        input_file: Path to input jsonl file
        output_file: Path to output jsonl file
        seed: Seed used to generate the seeds for the permutations.
        num_permutations: Number of permutations to use.
        num_bands: Number of bands to use for LSH. Default is 20
        mp: Whether to use multiprocessing. Default is False
        check_candidates_exactly: Whether to check candidates or use minhash signature. Default is False
        filter_below: Threshold Jaccard for filtering candidates. Default is 0.0
        leave_one_for_each_duplicate: Whether to leave one remaining for each duplicate. Default is False
        existing_index: Existing index to use. Default is None. If string, loads index from file. If LSH, uses that index.
        index_save_dir: Directory to save index to. Default is None
        hash_function: Hash function to use. Default is xxh32_intdigest
    """
    documentlist: List[Document] = jsonl_to_documentlist(input_file)
    filtered_documentlist = filter_documentlist(
        documentlist,
        seed,
        num_permutations,
        num_bands,
        mp,
        check_candidates_exactly,
        filter_below,
        leave_one_for_each_duplicate,
        existing_index,
        index_save_dir,
        save_matches_dir,
        hash_function,
    )
    documentlist_to_jsonl(output_file, filtered_documentlist)
