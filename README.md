# QuantizedEuclideanCAM documentation

`QuantizedEuclideanCAM` is a NumPy/Torch CAM-style simulator for approximate weighted cosine search using squared Euclidean distance over quantized CAM states.

The target ranking is:

```text
maximize_i weight_i * cosine(query, vector_i)
```

Internally, each stored vector is normalized, folded together with its signed weight, padded, optionally quantized, and searched by CAM-style Euclidean distance.

Negative weights are supported by flipping the stored vector direction.

---

## Import

```python
from CAM_patched import QuantizedEuclideanCAM
```

or, if the file is renamed back to `CAM.py`:

```python
from CAM import QuantizedEuclideanCAM
```

---

## Basic construction from raw vectors and weights

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    max_subarray_cols=64,
    pad_per_subarray=False,
    num_pad_dims=8,
)
```

Expected inputs:

```text
vectors: shape [num_vectors, original_dim]
weights: shape [num_vectors]
```

Example:

```python
import numpy as np
from CAM_patched import QuantizedEuclideanCAM

vectors = np.random.randn(1000, 768).astype(np.float32)
weights = np.random.rand(1000).astype(np.float32)

cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    max_subarray_cols=64,
    pad_per_subarray=False,
    num_pad_dims=8,
)

query = np.random.randn(768).astype(np.float32)

result = cam.search(query, top_k=10)

print(result.indices)
print(result.int_distances)
print(result.float_distances)
print(result.similarities)
```

---

## Search result fields

`cam.search(...)` returns a `CAMSearchResult` object with these fields:

```text
indices              Indices of the selected stored rows.
int_distances        Integer CAM-state squared distances.
float_distances      Approximate float-space CAM distances.
ideal_scores         Float reference weighted cosine scores before quantization.
vectors              Stored normalized vectors for the selected rows.
normalized_weights   Normalized absolute weights for the selected rows.
similarities         Similarity estimate computed as 1 - 0.5 * float_distance.
```

Example:

```python
result = cam.search(query, top_k=5)

print("indices:", result.indices)
print("integer distances:", result.int_distances)
print("float distances:", result.float_distances)
print("similarities:", result.similarities)
print("ideal scores:", result.ideal_scores)
```

---

## Batch search

Use `search_batch` for many queries at once.

```python
queries = np.random.randn(32, 768).astype(np.float32)

result = cam.search_batch(queries, top_k=10)

print(result["indices"])          # shape [num_queries, top_k]
print(result["int_distances"])    # shape [num_queries, top_k]
print(result["float_distances"])  # shape [num_queries, top_k]
print(result["similarities"])     # shape [num_queries, top_k]
```

---

## Using precomputed / prequantized CAM vectors

The patched class supports loading already-quantized CAM rows directly.

Important: `prequantized_cam_vectors` must already be final CAM rows. They are not raw embedding vectors. They must already include:

```text
1. vector normalization
2. weight/sign folding
3. CAM padding
4. integer quantization
```

Example:

```python
import numpy as np
from CAM_patched import QuantizedEuclideanCAM

prequantized = np.load("prequantized_cam_vectors.npy")

cam = QuantizedEuclideanCAM(
    prequantized_cam_vectors=prequantized,

    # CAM state precision
    bits=4,

    # Original query/vector dimension before CAM padding
    prequantized_original_dim=768,

    # Query quantization rule:
    # q_state = round(q_float * query_quant_scale + query_quant_zero_point)
    query_quant_scale=7.5,
    query_quant_zero_point=7.5,

    # These must match how the prequantized rows were constructed
    max_subarray_cols=64,
    pad_per_subarray=False,
    num_pad_dims=8,
)

query = np.random.randn(768).astype(np.float32)

result = cam.search(query, top_k=10)

print(result.indices)
print(result.int_distances)
print(result.float_distances)
print(result.similarities)
```

When using `prequantized_cam_vectors`, these fields are meaningful:

```python
result.indices
result.int_distances
result.float_distances
result.similarities
```

These fields are placeholders or `NaN`, because the CAM object no longer knows the original raw vectors and weights:

```python
result.vectors
result.normalized_weights
result.ideal_scores
```

---

## Saving a CAM for later prequantized loading

Build the CAM normally first:

```python
cam_build = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    max_subarray_cols=64,
    pad_per_subarray=False,
    num_pad_dims=8,
    quant_range_mode="percentile",
    quant_percentile=99.9,
)
```

Save the integer CAM rows and the metadata needed to quantize future queries:

```python
np.savez(
    "cam_precomputed.npz",
    prequantized_cam_vectors=cam_build.cam_int_vectors,
    bits=np.array(cam_build.bits),
    original_dim=np.array(cam_build.original_dim),
    max_subarray_cols=np.array(cam_build.max_subarray_cols),
    pad_per_subarray=np.array(cam_build.pad_per_subarray),
    num_pad_dims=np.array(cam_build.num_pad_dims),

    # Convert min/step form into direct query quantization form:
    query_quant_scale=np.array(1.0 / cam_build.query_quant_step),
    query_quant_zero_point=np.array(
        -float(cam_build.query_quant_min) / float(cam_build.query_quant_step)
    ),
)
```

Reload later without raw vectors or weights:

```python
z = np.load("cam_precomputed.npz")

cam = QuantizedEuclideanCAM(
    prequantized_cam_vectors=z["prequantized_cam_vectors"],
    bits=int(z["bits"]),
    prequantized_original_dim=int(z["original_dim"]),
    max_subarray_cols=int(z["max_subarray_cols"]),
    pad_per_subarray=bool(z["pad_per_subarray"]),
    num_pad_dims=int(z["num_pad_dims"]),
    query_quant_scale=float(z["query_quant_scale"]),
    query_quant_zero_point=float(z["query_quant_zero_point"]),
)
```

---

## CAM metadata / info

Use `info()` to inspect the CAM object.

```python
from pprint import pprint

pprint(cam.info())
```

Useful fields include:

```text
num_vectors
original_dim
cam_dim
quantize_vectors
prequantized_cam_vectors
query_quant_scale
query_quant_zero_point
bits
qmax
max_subarray_cols
num_col_subarrays
pad_per_subarray
num_pad_dims
quant_range_mode
quant_min
quant_max
quant_step
query_quant_min
query_quant_max
query_quant_step
use_torch
torch_fast_distances
torch_distance_is_approximate
```

Example:

```python
info = cam.info()

print("num_vectors:", info["num_vectors"])
print("original_dim:", info["original_dim"])
print("cam_dim:", info["cam_dim"])
print("bits:", info["bits"])
print("prequantized:", info["prequantized_cam_vectors"])
print("query_quant_scale:", info["query_quant_scale"])
print("query_quant_zero_point:", info["query_quant_zero_point"])
```

---

## Quantization options

Main stored-vector quantization options:

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    quant_range_mode="percentile",
    quant_percentile=99.9,
    quant_symmetric=True,
)
```

Supported `quant_range_mode` values:

```text
"fixed"
"stored_minmax"
"calibrated_minmax"
"percentile"
"custom"
```

Common options:

```text
bits                 Number of quantized CAM-state bits.
quant_clip           Fixed symmetric clipping range when using fixed mode.
quant_percentile     Percentile range when using percentile mode.
quant_symmetric      Whether to use symmetric quantization range.
quant_min            Manual minimum for custom mode.
quant_max            Manual maximum for custom mode.
calibration_queries  Optional queries used to calibrate stored CAM range.
```

---

## Separate query quantization

By default, stored CAM rows and queries use the same quantization range.

To give queries their own range:

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    separate_query_quantization=True,
    query_quant_range_mode="percentile",
    query_quant_percentile=99.9,
)
```

You can also pass example queries:

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    query_quantization_examples=example_queries,
)
```

---

## Subarray settings

CAM dimensions are split into physical column subarrays.

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    max_subarray_cols=64,
    pad_per_subarray=False,
    num_pad_dims=8,
)
```

Options:

```text
max_subarray_cols    Maximum CAM columns per physical subarray.
pad_per_subarray     If False, padding is appended once at the end.
                     If True, each subarray gets its own padding dimensions.
num_pad_dims         Number of padding dimensions.
```

If `pad_per_subarray=True`, then:

```text
num_pad_dims must be smaller than max_subarray_cols
```

because each subarray needs room for real vector dimensions.

---

## Voting search

`vote_search` simulates subarray-local winner-takes-all voting.

```python
result = cam.vote_search(query, top_k=10)

print(result["indices"])
print(result["votes"])
print(result["int_distances"])
print(result["similarities"])
```

Each column subarray picks one local winning row. Final ranking is by vote count, with full CAM distance used as the tie-breaker.

Batch version:

```python
result = cam.vote_search_batch(queries, top_k=10)

print(result["indices"])
print(result["votes"])
```

---

## Voting internals / diagnostics

Use `show_vote_internals` to inspect the voting behavior in detail.

```python
internals = cam.show_vote_internals(
    query,
    top_k=5,
    local_top_k=5,
    include_all_vectors=False,
)

print(internals["top"]["indices"])
print(internals["top"]["votes"])
print(internals["subarrays"])
```

Useful returned fields:

```text
query
top
final_winners
subarray_winner_indices
subarrays
vote_counts_all_vectors
ranking_all_vectors
metadata
```

Set `include_all_vectors=True` to include full per-subarray matrices for every stored vector.

---

## Query quantization helpers

Quantize one query:

```python
q_cam = cam.quantize_query(query)

print(q_cam.shape)
print(q_cam.dtype)
```

Quantize a batch of queries:

```python
q_cam_batch = cam.quantize_queries(queries)

print(q_cam_batch.shape)
print(q_cam_batch.dtype)
```

Dequantize a query after quantization:

```python
q_dequantized = cam.dequantized_query(query)
```

Dequantize stored CAM rows:

```python
cam_rows_float = cam.dequantized_cam_vectors()
```

---

## Float reference helpers

When the CAM is constructed from raw vectors and weights, these helpers compare against the unquantized ideal target.

```python
scores = cam.ideal_scores(query)
```

Higher is better.

```python
scores = cam.ideal_scores_batch(queries)
```

Returns shape:

```text
[num_queries, num_vectors]
```

```python
distances = cam.ideal_distances(query)
```

Lower is better.

When the CAM is loaded from `prequantized_cam_vectors`, these methods return `NaN` values because the original raw vectors and weights are unavailable.

---

## Torch acceleration

Use `use_torch=True` to enable Torch-backed fast distance computation where available.

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    use_torch=True,
    torch_device="cuda",
)
```

Optional Torch settings:

```text
torch_device                  Example: "cuda", "cpu", or None.
torch_chunk_rows              Optional query chunk size for batch operations.
torch_fast_quantized_bits     Maximum bit width for Torch fast quantized path.
torch_distance_dtype          "auto", "float32", or "float64".
strict_integer_distances      Require exact integer distance accumulation.
```

For large bit widths or very large CAM dimensions, exact integer squared distances may not fit safely in fixed-width integer arithmetic. In those cases, the class may use a floating-point distance path unless `strict_integer_distances=True`.

Check the active behavior with:

```python
from pprint import pprint
pprint(cam.info())
```

Look at:

```text
torch_fast_distances
torch_returns_integer_distances
torch_distance_is_approximate
int64_exact_distances_safe
float64_exact_distances_safe
```

---

## Constructor reference

```python
cam = QuantizedEuclideanCAM(
    vectors=None,
    weights=None,

    bits=4,
    quantize_vectors=True,

    prequantized_cam_vectors=None,
    query_quant_scale=None,
    query_quant_zero_point=0.0,
    prequantized_original_dim=None,

    max_subarray_cols=64,
    pad_per_subarray=False,
    num_pad_dims=8,

    quant_range_mode="percentile",
    quant_clip=1.0,
    quant_percentile=99.9,
    quant_symmetric=True,
    quant_min=None,
    quant_max=None,
    calibration_queries=None,

    separate_query_quantization=False,
    query_quantization_examples=None,
    query_quant_range_mode=None,
    query_quant_clip=None,
    query_quant_percentile=None,
    query_quant_symmetric=None,
    query_quant_min=None,
    query_quant_max=None,

    eps=1e-8,

    use_torch=False,
    torch_device=None,
    torch_chunk_rows=None,
    torch_fast_quantized_bits=32,
    torch_distance_dtype="auto",
    strict_integer_distances=False,
)
```

Use either:

```text
vectors + weights
```

or:

```text
prequantized_cam_vectors + query_quant_scale
```

Do not pass both.

---

## Public methods

```python
cam.search(query, top_k=5)
```

Exact CAM search for one query.

```python
cam.search_batch(queries, top_k=5, query_chunk_size=None)
```

Exact CAM search for many queries.

```python
cam.broadcast(query, top_k=5)
```

Alias for `search`.

```python
cam.threshold_search(query, threshold)
```

Return indices whose CAM similarity is greater than or equal to `threshold`.

```python
cam.vote_search(query, top_k=5)
```

Subarray winner-takes-all voting search for one query.

```python
cam.vote_search_batch(queries, top_k=5, query_chunk_size=None)
```

Subarray winner-takes-all voting search for many queries.

```python
cam.weighted_vote_search_batch(queries, top_k=3, points=[3,2,1])

```
Give many points, winner is the one with the most

```python
cam.show_vote_internals(
    query,
    top_k=5,
    local_top_k=5,
    include_all_vectors=True,
)
```

Detailed diagnostic output for voting search.

```python
cam.quantize_query(query)
```

Normalize, augment, and optionally quantize one query.

```python
cam.quantize_queries(queries)
```

Normalize, augment, and optionally quantize many queries.

```python
cam.ideal_scores(query)
```

Float reference weighted cosine scores before quantization.

```python
cam.ideal_scores_batch(queries)
```

Float reference weighted cosine scores for many queries.

```python
cam.ideal_distances(query)
```

Float Euclidean distances after CAM augmentation but before quantization.

```python
cam.dequantized_cam_vectors()
```

Return quantized CAM rows mapped back to approximate float values.

```python
cam.dequantized_query(query)
```

Return one query’s quantized CAM states mapped back to approximate float values.

```python
cam.info()
```

Return CAM, quantization, Torch, and subarray metadata.

---

## Common patterns

### Top-k nearest CAM rows

```python
result = cam.search(query, top_k=10)
top_indices = result.indices
```

### Top-k for many queries

```python
result = cam.search_batch(queries, top_k=10)
top_indices = result["indices"]
```

### Inspect CAM configuration

```python
from pprint import pprint
pprint(cam.info())
```

### Load precomputed CAM rows

```python
z = np.load("cam_precomputed.npz")

cam = QuantizedEuclideanCAM(
    prequantized_cam_vectors=z["prequantized_cam_vectors"],
    bits=int(z["bits"]),
    prequantized_original_dim=int(z["original_dim"]),
    max_subarray_cols=int(z["max_subarray_cols"]),
    pad_per_subarray=bool(z["pad_per_subarray"]),
    num_pad_dims=int(z["num_pad_dims"]),
    query_quant_scale=float(z["query_quant_scale"]),
    query_quant_zero_point=float(z["query_quant_zero_point"]),
)
```

### Use CUDA if Torch is available

```python
cam = QuantizedEuclideanCAM(
    vectors,
    weights,
    bits=4,
    use_torch=True,
    torch_device="cuda",
)
```

---

## Important caveats

1. `prequantized_cam_vectors` are final CAM rows, not raw vectors.

2. When using `prequantized_cam_vectors`, the original vectors, weights, and ideal unquantized scores are not recoverable.

3. Query dimension must equal `original_dim`.

4. Stored vectors and queries cannot be zero vectors because the class normalizes them.

5. If `pad_per_subarray=True`, the metadata used at reload time must exactly match the metadata used when the prequantized rows were created.

6. `similarities` are derived from CAM float distances as:

```text
similarity = 1 - 0.5 * float_distance
```

7. For quantized CAMs, search is approximate relative to the original weighted cosine objective.

