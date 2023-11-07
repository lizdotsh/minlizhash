# Liz's MinHash Implimentation

This is a simple implimentation of the MinHash LSH deduplication algorithm. See more: https://en.wikipedia.org/wiki/MinHash.
Not meant to be performant or actually usd, created as learning exercise.


## UPDATE - 2022-11-07:

Added a JIT minhash backend using numba. Is around ~70x faster than before if jit=True. Selection of hash function will only work for LSH if jit is enabled, as I had to write a custom FNV-1a implementation to get it to work with numba. You can use FNV-1a for everything by setting hash_function = mh.jit.fnv1a32_bytes.

## Basic Usage 


Simple usage example. 

```
# use filter_jsonl with very similar args to go directly from jsonl to filtered jsonl
filter_documentlist(
    documentlist: List[Document],
    seed: int,
    num_permutations: int = 128,
    num_bands: int = 32,
    mp: bool = False,
    progress_bar: bool = True,
    jit: bool = True,
    check_candidates_exactly: bool = False,
    filter_below: float = 0.0,
    leave_one_for_each_duplicate: bool = False,
    existing_index: Union[None, LSH, str] = None,
    index_save_dir: str | None = None,
    save_matches_dir: str | None = None,
    hash_function: Callable[[bytes, int], int] = xxh32_intdigest,
) -> List[Document]:
    """Takes a list of documents and returns a new list of documents with signatures.
    If JIT is chosen, hash funciton choice will only be used for LSH. Minhash uses a custom implementation w/ JIT.

    Args:
        documentlist: List of documents to be signed
        seed: Seed used to generate the seeds for the permutations.
        num_permutations: Number of permutations to use.
        num_bands: Number of bands to use for LSH. Default is 20
        mp: Whether to use multiprocessing. Default is False
        progress_bar: Whether to show progress bar. Default is True
        jit: Whether to use custom JIT compiled hashing function (uses numba). Default is True (recommended).
        check_candidates_exactly: Whether to check candidates or use minhash signature. Default is False
        filter_below: Threshold Jaccard for filtering candidates. Default is 0.0. Higher will increase result set (more false positives)
        leave_one_for_each_duplicate: Whether to leave one remaining for each duplicate. Default is False
        existing_index: Existing index to use. Default is None. If string, loads index from file. If LSH, uses that index.
        index_save_dir: Directory to save index to. Default is None
        hash_function: Hash function to use. Default is xxh32_intdigest

    Returns:
        List of documents with signatures
    """
```

```python
import minlizhash as mh

documentList: List[Document] =  mh.jsonl_to_documentlist("path/to/file.jsonl")

filtered_documentlist: List[Document] = mh.filter_documentlist(
    documentList,
    seed=123,
    num_permutations=100,
    mp=True,
    progress_bar=True,
    jit=True,
    check_candidates_exactly=False,
    filter_below=0.5,
    leave_one_for_each_duplicate=True,
    existing_index=None,
    index_save_dir=None,
    save_matches_dir=None
    )

mh.documentlist_to_jsonl(filtered_documentlist, "path/to/file.jsonl")
```
## Rough Explanation

MinHash is an approximation of the [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) between two sets. If you assume the output space of a hash function is uniformly distributed, and you hash each element of a set and take the minimum (or maximum, doesn't matter) hash value, that can be thought of as a uniform random variable. In other words, it's the same as randomly drawing one element from the set. 

This by itself isn't that useful, until you realize that hash functions are deterministic. Each random draw will be the same for each seed if given the same input. Even just with one, this is already clearly an approximation of the jaccard index, just with a very low sample size. 

MinHash takes this a step further by computing many hashes with different seeds for each input variable. If you, say, hash each element 150 times and taken the minimum for each, the law of large numbers says that the average of these 150 minimums will be very close to the true jaccard index. So you can use this set of 150 minimums as a signature for each document, and compare the signatures to find similar documents.

The problem comes with large scale, as each similarity search is O(N), requiring you to search across every other already sampled document. And that is just for a single document. If you have millions of sets to compare, this becomes very slow.

This is where [Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) comes in. MinHash LSH can be done in a few ways, and I currently implemented the more simple mechanism. The idea is that if you have a signature of 200 hashes, you go over it and re hash the concatenated hashes of, say, 10 hashes at a time. This leaves you with a smaller (but fuzzier) signature of 20 hashes. For each of these 20 hashes we have buckets (in this case, a simple list of 20 dictionaries) that we store bighash: [docID1, docID2, ...]. If two items are duplicates, they will likely have at least on hash in common. So we now just look through all the buckets and find the hashes with more than one document in them. We then check the documents in each bucket to see if they are actually similar (either with minhash or the original jaccard index). This is much faster than checking every document against every other document, and the tradeoff is that we will have some false positives, and it's not as good for things that aren't as similar. 

This is a special version of LSH in general, with the more generalized one consisting of generating a bunch of random hyperplanes (random normal / norm(random_normal)) in a high dimensional space (with d being the dimensionality of the input data, in this case how many minhash permutations). Then, when querying, you simply take the dot product of your query vectors against these projections and then bucket them based on the sign of the result. This effectively creates buckets in a very efficient way, and lowers the search space dramatically. I have only implemented up to here, but iirc then within the buckets you can directly compare them similar to before, often using things like the hamming distance (bitwise method) to make it especially fast. As far as I can understand, this is much better for more continuous variables you want to intersect, like normal vector embeddings. For hashes of documents the hashes themselves don't mean all that much, and the minhash method is more than sufficient (and likely faster/so close it doesn't matter). Take everything I said in second half of paragraph with grain of salt. 

For more info see chapter 3 of [Mining Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf).
## Disclaimer: 

I didn't know what literally any of this stuff until a couple days ago. Would love any feedback on the code or if I am misunderstanding anything / got the math wrong. Thanks! :)


 
## Notes

Even though I'm releasing this, its very much a WIP. For instance, I'm still not sure if I want to combine together the initial text processing and minhashing, and went back and fourth about having the Document typedict have other fields (original string, processed string, etc). Decided to keep it simple for now but expose the functions in minlizhash.tokens for use in other projects.

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
hashed_documents = hsr.hash_documents(documentlist)
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

## Other / Advanced / Unfinished

### DocumentSigner
There are two versions of the DocumentSigner. One that hashes the document before signing and one that signs the document before hashing. 


### minlizhash.tokens 
This module contains a couple basic functions for cleaning up strings and tokenizing them for use in the minhash algorithm.

### LSHIndex_Projection

This is NOT FINISHED. It will NOT work. I just wanted to get the basic workings down. I also have realized that the speedup of doing this projection is not actually especially useful for document similarity. It is mathematically interesting though.