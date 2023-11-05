# WIP

WIP implementation of the minhash algorithm (LSH coming soon) for comaring document similarity. 

Not meant to be performant or actually usd, created as learning exercise.



# Usage 


Simple usage example. 

```
# use filter_jsonl with very similar args to go directly from jsonl to filtered jsonl
filter_documentlist(
    documentlist: List[Document],
    seed: int,
    num_permutations: int,
    hash_function: (bytes, int) -> int = xxh32_intdigest,
    mp: bool = False,
    check_candidates_exactly: bool = False,
    filter_below: float = 0,
    leave_one_for_each_duplicate: bool = False,
    existing_index: LSH | str | None = None,
    index_save_dir: str | None = None,
    save_matches_dir: str | None = None
) -> List[Document]
Takes a list of documents and returns a new list of documents with signatures Args:
    documentlist: List of documents to be signed
    seed: Seed used to generate the seeds for the permutations.
    num_permutations: Number of permutations to use.
    hash_function: Hash function to use. Default is xxh32_intdigest
    mp: Whether to use multiprocessing. Default is False
    check_candidates_exactly: Whether to check candidates or use minhash signature. Default is False
    filter_below: Threshold Jaccard for filtering candidates. Default is 0.0
    leave_one_for_each_duplicate: Whether to leave one remaining for each duplicate. Default is False
    existing_index: Existing index to use. Default is None. If string, loads index from file. If LSH, uses that index.
    index_save_dir: Directory to save index to. Default is None

Returns:
    List of documents with signatures

```

```python
import minlizhash as mh

documentList: List[Document] =  mh.jsonl_to_documentlist("path/to/file.jsonl")

filtered_documentlist: List[Document] = mh.filter_documentlist(
    documentList,
    seed=123,
    num_permutations=100,
    mp=True,
    check_candidates_exactly=False,
    filter_below=0.5,
    leave_one_for_each_duplicate=True,
    existing_index=None,
    index_save_dir=None,
    save_matches_dir=None
    )

mh.documentlist_to_jsonl(filtered_documentlist, "path/to/file.jsonl")
```
 

1. Load jsonl with {"id": int, tokens: List[str]} format

```python
import minlizhash as mh
documentList: List[Document] =  mh.jsonl_to_documentlist("path/to/file.jsonl")
```

2. Create Hasher object
```python
hsr = mh.Hasher(seed=123, num_permutations=100)
```
3. Hash documents
```python
hashed_documents: List[Document] = hsr.hash_documents(documentList)
```
4. Create LSHIndex object and add documents
```python
index = mh.LSHIndex.from_hasher(hsr)
index.add_batch_documentlist(hashed_documents)
candidates: List[Tuple[int, int]] = index.get_all_candidates()
```
5. Veryify candidates with Jaccard similarity of the minhashes

```python

candidates_with_jaccard: List[Tuple[int, int, float]] = mh.check_candidatelist(
    candidates, 
    hashed_documents, # Document IDs must be either equal to index or hashed_documents must be dict of id -> Document
    exact = False # True will use exact jaccard, False will use minhash jaccard
    )
```
6. Filter candidates by jaccard similarity
```python

filtered_candidates: List[Tuple[int, int, float]] = mh.filter_candidates(
    candidates_with_jaccard, 
    min_jaccard=0.5
    )


    




```

Separated into three primary parts: 

# Types: 

### Document

Document object is just a TypedDict with the following fields: 

```python

document: Document = {
    "id": str,
    "text": str,
    "hashes": npt.NDArray[np.int32],
    "signature": npt.NDArray[np.int32], # OPTIONAL
}
```

The signature field is optional and is designed to be added after processing. 

### 

### Hasher

Hasher object is created with a seed, number of permutations to generate, and a DocumentSigner object (I suggest you use default DocumentSigner. Mostly added for modularity).

It is used to hash the Document 

```python
import minlizhash as mh

hsr = mh.Hasher(
    seed=123, 
    num_permutations=100,
    # OPTIONAL 
    signer=mh.DocumentSignerMinBefore( 
        hash_function = xxhash.xxh32_intdigest)
    )

documentlist: List[Document] = mh.jsonl_to_documentlist("path/to/file.jsonl")
hashed_documents = hsr.hash_documents_batched(documentlist)
```

### LSHIndex

Object that allows indexing of Documents in an efficient manner by use of banding the hashes. Can be loaded manually or by using the seed/hash/num_permutations from a Hasher object. 

```python
index = mh.LSHIndex.from_hasher(hsr)
index.add_batch_documentlist(hashed_documents)
index.save("path/to/file/index.pkl")

# Get all candidate pairs
candidates = index.get_all_candidates()
checked_candidates = check_candidatelist(candidates, hashed_documents, exact=False)
filtered_candidates: Set[int | str] = filter_candidates(checked_candidates, min_jaccard=0.5)

# filter documentlist 

documentlist_no_dup: List[Document] = [
    doc for doc in documentlist if doc["id"] not in filtered_candidates
]

mh.documentlist_to_jsonl(documentlist_no_dup, "path/to/file.jsonl")

```

