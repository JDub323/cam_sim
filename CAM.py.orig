from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import warnings
import numpy as np


QuantRangeMode = Literal[
    "fixed",
    "stored_minmax",
    "calibrated_minmax",
    "percentile",
    "custom",
]

FastDistanceDType = Literal["auto", "float32", "float64"]


@dataclass(frozen=True)
class CAMSearchResult:
    indices: np.ndarray
    int_distances: np.ndarray
    float_distances: np.ndarray
    ideal_scores: np.ndarray
    vectors: np.ndarray
    normalized_weights: np.ndarray
    similarities: Optional[np.ndarray] = None


class QuantizedEuclideanCAM:
    """
    Quantized Euclidean CAM simulator for weighted cosine search.

    The target ranking is:

        maximize_i weight_i * cosine(query, vector_i)

    The CAM itself only computes squared Euclidean distance, so each stored
    vector is transformed before quantization:

        x_i     <- normalize(x_i)
        sign_i  = sign(weight_i)
        x_i     <- sign_i * x_i
        alpha_i = abs(weight_i) / max(abs(weight))
        body_i  = alpha_i * x_i
        pad_i   = sqrt(1 - alpha_i^2)

    The CAM stores:

        [body_i, pad_i / sqrt(num_pad_dims), ..., pad_i / sqrt(num_pad_dims)]

    The query becomes:

        [normalize(q), 0, ..., 0]

    Before quantization, Euclidean nearest neighbor is equivalent to weighted
    cosine ranking. After quantization, it is an approximation.

    Negative weights are represented by flipping the corresponding normalized
    vector direction and using the absolute weight magnitude in the CAM.

    Set quantize_vectors=False to skip quantization entirely and compute CAM
    distances directly on the float CAM vectors. By default, queries use the
    same quantization range as stored rows; pass query_quantization_examples or
    separate_query_quantization=True to give queries their own range. Set
    use_torch=True to use a Torch-backed matrix-vector distance kernel for the
    common <=8-bit or float search path. In CAM_fast.py, use_torch=True can
    also accelerate wider quantized states up to torch_fast_quantized_bits
    with a floating-point distance kernel. That wide path is fast but
    approximate unless the state-space distance is small enough to be exactly
    represented.
    """

    def __init__(
        self,
        vectors: Optional[Sequence[Sequence[float]]] = None,
        weights: Optional[Sequence[float]] = None,
        *,
        bits: int = 4,
        quantize_vectors: bool = True,
        prequantized_cam_vectors: Optional[Sequence[Sequence[int]]] = None,
        query_quant_scale: Optional[float] = None,
        query_quant_zero_point: float = 0.0,
        prequantized_original_dim: Optional[int] = None,
        max_subarray_cols: int = 64,
        pad_per_subarray: bool = False,
        num_pad_dims: int = 8,
        quant_range_mode: QuantRangeMode = "percentile",
        quant_clip: float = 1.0,
        quant_percentile: float = 99.9,
        quant_symmetric: bool = True,
        quant_min: Optional[float] = None,
        quant_max: Optional[float] = None,
        calibration_queries: Optional[Sequence[Sequence[float]]] = None,
        separate_query_quantization: bool = False,
        query_quantization_examples: Optional[Sequence[Sequence[float]]] = None,
        query_quant_range_mode: Optional[QuantRangeMode] = None,
        query_quant_clip: Optional[float] = None,
        query_quant_percentile: Optional[float] = None,
        query_quant_symmetric: Optional[bool] = None,
        query_quant_min: Optional[float] = None,
        query_quant_max: Optional[float] = None,
        eps: float = 1e-8,
        use_torch: bool = False,
        torch_device: Optional[str] = None,
        torch_chunk_rows: Optional[int] = None,
        torch_fast_quantized_bits: int = 32,
        torch_distance_dtype: FastDistanceDType = "auto",
        strict_integer_distances: bool = False,
    ) -> None:
        if quantize_vectors and bits < 1:
            raise ValueError("bits must be >= 1 when quantize_vectors=True")

        if quantize_vectors and bits > 32:
            raise ValueError("CAM_fast supports quantized states up to 32 bits")

        if max_subarray_cols < 1:
            raise ValueError("max_subarray_cols must be >= 1")

        if num_pad_dims < 1:
            raise ValueError("num_pad_dims must be >= 1")

        if pad_per_subarray and num_pad_dims >= max_subarray_cols:
            raise ValueError(
                "When pad_per_subarray=True, num_pad_dims must be smaller than "
                "max_subarray_cols so each subarray has room for real vector data."
            )

        if quantize_vectors and not (0.0 < quant_percentile <= 100.0):
            raise ValueError("quant_percentile must be in (0, 100]")

        if query_quant_percentile is not None and not (
            0.0 < float(query_quant_percentile) <= 100.0
        ):
            raise ValueError("query_quant_percentile must be in (0, 100]")

        if torch_chunk_rows is not None and int(torch_chunk_rows) < 1:
            raise ValueError("torch_chunk_rows must be >= 1 when provided")

        if int(torch_fast_quantized_bits) < 0 or int(torch_fast_quantized_bits) > 32:
            raise ValueError("torch_fast_quantized_bits must be in [0, 32]")

        if torch_distance_dtype not in {"auto", "float32", "float64"}:
            raise ValueError(
                'torch_distance_dtype must be one of "auto", "float32", or "float64"'
            )

        has_prequantized = prequantized_cam_vectors is not None
        if has_prequantized and (vectors is not None or weights is not None):
            raise ValueError(
                "Pass either vectors/weights or prequantized_cam_vectors, not both. "
                "prequantized_cam_vectors are final CAM rows with weights and "
                "padding already baked in."
            )
        if not has_prequantized and (vectors is None or weights is None):
            raise ValueError(
                "vectors and weights are required unless prequantized_cam_vectors "
                "is provided"
            )
        if has_prequantized and not quantize_vectors:
            raise ValueError("prequantized_cam_vectors requires quantize_vectors=True")
        if has_prequantized and query_quant_scale is None:
            raise ValueError(
                "query_quant_scale is required when prequantized_cam_vectors is provided"
            )
        if not has_prequantized and prequantized_original_dim is not None:
            raise ValueError(
                "prequantized_original_dim is only used with prequantized_cam_vectors"
            )
        if query_quant_scale is not None and float(query_quant_scale) <= 0.0:
            raise ValueError("query_quant_scale must be positive")
        if prequantized_original_dim is not None and int(prequantized_original_dim) < 1:
            raise ValueError("prequantized_original_dim must be >= 1 when provided")

        self.quantize_vectors = bool(quantize_vectors)
        self.prequantized_cam_vectors = bool(has_prequantized)
        self.query_quant_scale = (
            None if query_quant_scale is None else float(query_quant_scale)
        )
        self.query_quant_zero_point = float(query_quant_zero_point)
        self.bits = int(bits)
        self.qmax = (1 << self.bits) - 1 if self.quantize_vectors else None
        self.max_subarray_cols = int(max_subarray_cols)
        self.pad_per_subarray = bool(pad_per_subarray)
        self.num_pad_dims = int(num_pad_dims)
        self.quant_range_mode = quant_range_mode
        self.quant_clip = float(quant_clip)
        self.quant_percentile = float(quant_percentile)
        self.quant_symmetric = bool(quant_symmetric)
        self.separate_query_quantization = bool(
            separate_query_quantization
            or query_quantization_examples is not None
            or query_quant_range_mode is not None
            or query_quant_clip is not None
            or query_quant_percentile is not None
            or query_quant_symmetric is not None
            or query_quant_min is not None
            or query_quant_max is not None
        )
        self.query_quant_range_mode = (
            quant_range_mode
            if query_quant_range_mode is None
            else query_quant_range_mode
        )
        self.query_quant_clip = (
            self.quant_clip if query_quant_clip is None else float(query_quant_clip)
        )
        self.query_quant_percentile = (
            self.quant_percentile
            if query_quant_percentile is None
            else float(query_quant_percentile)
        )
        self.query_quant_symmetric = (
            self.quant_symmetric
            if query_quant_symmetric is None
            else bool(query_quant_symmetric)
        )
        self.eps = float(eps)
        self.use_torch = bool(use_torch)
        self.torch_device = torch_device
        self.torch_chunk_rows = (
            None if torch_chunk_rows is None else int(torch_chunk_rows)
        )
        self.torch_fast_quantized_bits = int(torch_fast_quantized_bits)
        self.torch_distance_dtype = torch_distance_dtype
        self.strict_integer_distances = bool(strict_integer_distances)
        self._torch = None
        self._torch_distance_matrix = None
        self._torch_distance_matrix_t = None
        self._torch_distance_norm_sq = None
        self._torch_can_use_fast_distances = False
        self._torch_distance_torch_dtype = None
        self._torch_distance_np_dtype = None
        self._torch_returns_integer_distances = False
        self._torch_distance_is_approximate = False

        if self.prequantized_cam_vectors:
            prequantized = np.asarray(prequantized_cam_vectors)
            if prequantized.ndim != 2:
                raise ValueError(
                    "prequantized_cam_vectors must have shape [num_vectors, cam_dim]"
                )
            if prequantized.shape[0] == 0 or prequantized.shape[1] == 0:
                raise ValueError("prequantized_cam_vectors must be non-empty")
            if not np.issubdtype(prequantized.dtype, np.integer):
                raise ValueError("prequantized_cam_vectors must contain integer states")

            self.num_vectors = int(prequantized.shape[0])
            self.cam_dim = int(prequantized.shape[1])
            if prequantized_original_dim is None:
                if self.pad_per_subarray:
                    raise ValueError(
                        "prequantized_original_dim is required with prequantized_cam_vectors "
                        "when pad_per_subarray=True"
                    )
                inferred_original_dim = self.cam_dim - self.num_pad_dims
            else:
                inferred_original_dim = int(prequantized_original_dim)
            if inferred_original_dim < 1:
                raise ValueError(
                    "prequantized_cam_vectors has too few columns for the requested "
                    "num_pad_dims"
                )
            self.original_dim = inferred_original_dim

            # The stored CAM rows already include direction, weight scaling, and
            # padding. Keep metadata placeholders only for API compatibility.
            self.original_vectors = np.full(
                (self.num_vectors, self.original_dim), np.nan, dtype=np.float32
            )
            self.raw_weights = np.ones(self.num_vectors, dtype=np.float32)
            self.weight_signs = np.ones(self.num_vectors, dtype=np.float32)
            self.vectors = np.full(
                (self.num_vectors, self.original_dim), np.nan, dtype=np.float32
            )
            self.abs_weights = np.ones(self.num_vectors, dtype=np.float32)
            self.normalized_weights = np.ones(self.num_vectors, dtype=np.float32)
            self.signed_normalized_weights = np.ones(
                self.num_vectors, dtype=np.float32
            )

            self.original_slices = self._make_original_slices()
            expected_cam_dim = self._build_augmented_query(
                np.zeros(self.original_dim, dtype=np.float32)
            ).shape[0]
            if expected_cam_dim != self.cam_dim:
                raise ValueError(
                    "prequantized_cam_vectors has cam_dim "
                    f"{self.cam_dim}, but prequantized_original_dim={self.original_dim}, "
                    f"num_pad_dims={self.num_pad_dims}, "
                    f"pad_per_subarray={self.pad_per_subarray}, and "
                    f"max_subarray_cols={self.max_subarray_cols} imply "
                    f"cam_dim {expected_cam_dim}"
                )
            self.cam_col_slices = self._make_cam_col_slices()
            self.cam_int_vectors = prequantized.copy()
            self.cam_float_vectors = np.empty(
                (self.num_vectors, self.cam_dim), dtype=np.float32
            )
        else:
            x = self._as_float32_numpy(vectors)
            w = self._as_float32_numpy(weights)

            if x.ndim != 2:
                raise ValueError("vectors must have shape [num_vectors, vector_dim]")

            if x.shape[1] == 0:
                raise ValueError("vectors must have at least one dimension")

            if w.ndim != 1 or w.shape[0] != x.shape[0]:
                raise ValueError("weights must have shape [num_vectors]")

            if not np.all(np.isfinite(w)):
                raise ValueError("weights must be finite")

            self.num_vectors, self.original_dim = x.shape

            # Negative weights can be represented as positive magnitudes applied to
            # the opposite direction: w * cos(q, x) == abs(w) * cos(q, sign(w) * x).
            self.original_vectors = self._normalize_rows(x)
            self.raw_weights = w.astype(np.float32, copy=True)
            self.weight_signs = np.where(self.raw_weights < 0.0, -1.0, 1.0).astype(
                np.float32
            )
            self.vectors = (self.original_vectors * self.weight_signs[:, None]).astype(
                np.float32
            )
            self.abs_weights = np.abs(self.raw_weights).astype(np.float32)
            self.normalized_weights = self._normalize_weights(self.abs_weights)
            self.signed_normalized_weights = (
                self.normalized_weights * self.weight_signs
            ).astype(np.float32)

            self.original_slices = self._make_original_slices()

            self.cam_float_vectors = self._build_augmented_vectors()
            self.cam_float_vectors = self._normalize_rows(self.cam_float_vectors)

            self.cam_dim = self.cam_float_vectors.shape[1]
            self.cam_col_slices = self._make_cam_col_slices()
        self._int64_exact_distances_safe = self._integer_distances_fit_int64()
        self._float64_exact_distances_safe = self._integer_distances_fit_float64()

        if (
            self.quantize_vectors
            and self.strict_integer_distances
            and not self._int64_exact_distances_safe
        ):
            raise ValueError(
                "strict_integer_distances=True is not safe for this bits/cam_dim; "
                "32-bit quantized squared distances generally overflow fixed-width "
                "integer accumulators. Disable strict_integer_distances to use the "
                "fast approximate floating distance path."
            )

        self.calibration_queries = (
            None
            if (not self.quantize_vectors or calibration_queries is None)
            else self._as_float32_numpy(calibration_queries)
        )
        self.query_quantization_examples = (
            None
            if (not self.quantize_vectors or query_quantization_examples is None)
            else self._as_float32_numpy(query_quantization_examples)
        )

        self._validate_query_examples(
            self.calibration_queries,
            name="calibration_queries",
        )
        self._validate_query_examples(
            self.query_quantization_examples,
            name="query_quantization_examples",
        )

        if self.quantize_vectors:
            if self.prequantized_cam_vectors:
                # query_quant_scale is the multiplier used to quantize:
                # q_state = round(q_float * query_quant_scale + zero_point).
                # Therefore dequantization is:
                # q_float ~= (q_state - zero_point) / query_quant_scale.
                self.quant_step = np.float64(1.0 / float(self.query_quant_scale))
                self.quant_min = np.float32(
                    -float(self.query_quant_zero_point) * float(self.quant_step)
                )
                self.quant_max = np.float32(
                    float(self.quant_min) + float(self.qmax) * float(self.quant_step)
                )
                self.query_quant_min = self.quant_min
                self.query_quant_max = self.quant_max
                self.query_quant_step = self.quant_step
                self.separate_query_quantization = False
                self.cam_float_vectors = self._dequantize_stored(
                    self.cam_int_vectors
                ).astype(np.float32, copy=False)
            else:
                self.quant_min, self.quant_max = self._choose_quant_range(
                    mode=quant_range_mode,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    values=self._stored_quant_range_values(),
                    calibrated_values=self._stored_calibrated_quant_range_values(),
                    quant_clip_value=self.quant_clip,
                    quant_percentile_value=self.quant_percentile,
                    quant_symmetric_value=self.quant_symmetric,
                    context="stored CAM",
                )

                if self.quant_max <= self.quant_min:
                    raise ValueError("Invalid stored CAM quantization range")

                self.quant_step = np.float64(
                    (float(self.quant_max) - float(self.quant_min)) / self.qmax
                )

                if self.separate_query_quantization:
                    self.query_quant_min, self.query_quant_max = self._choose_quant_range(
                        mode=self.query_quant_range_mode,
                        quant_min=query_quant_min,
                        quant_max=query_quant_max,
                        values=self._query_quant_range_values(),
                        calibrated_values=None,
                        quant_clip_value=self.query_quant_clip,
                        quant_percentile_value=self.query_quant_percentile,
                        quant_symmetric_value=self.query_quant_symmetric,
                        context="query",
                    )

                    if self.query_quant_max <= self.query_quant_min:
                        raise ValueError("Invalid query quantization range")

                    self.query_quant_step = np.float64(
                        (float(self.query_quant_max) - float(self.query_quant_min))
                        / self.qmax
                    )
                else:
                    self.query_quant_min = self.quant_min
                    self.query_quant_max = self.quant_max
                    self.query_quant_step = self.quant_step

                self.cam_int_vectors = self._quantize_stored(self.cam_float_vectors)
        else:
            self.quant_min = None
            self.quant_max = None
            self.quant_step = None
            self.query_quant_min = None
            self.query_quant_max = None
            self.query_quant_step = None
            self.cam_int_vectors = self.cam_float_vectors.astype(np.float32, copy=True)

        self._setup_torch_cache()

    def search(self, query: Sequence[float], *, top_k: int = 5) -> CAMSearchResult:
        """
        Exact CAM search.

        Computes summed squared Euclidean distance between the query and every
        CAM row. If quantize_vectors=True, both query and stored CAM rows are
        quantized first. If quantize_vectors=False, distances are computed
        directly on float CAM rows.
        """
        q_cam = self.quantize_query(query)
        cam_distances = self._all_cam_distances(q_cam)

        top_k = max(1, min(int(top_k), self.num_vectors))
        idx = np.argpartition(cam_distances, top_k - 1)[:top_k]
        idx = idx[np.argsort(cam_distances[idx])]
        int_distances = cam_distances[idx]
        float_distances = self._cam_distances_to_float_distances(
            int_distances, q_cam=q_cam, row_indices=idx
        )
        similarities = (1.0 - 0.5 * float_distances).astype(np.float32)

        return CAMSearchResult(
            indices=idx,
            int_distances=int_distances,
            float_distances=float_distances,
            ideal_scores=self.ideal_scores(query)[idx],
            vectors=self.vectors[idx],
            normalized_weights=self.normalized_weights[idx],
            similarities=similarities,
        )

    def search_batch(
        self,
        queries: Sequence[Sequence[float]],
        *,
        top_k: int = 5,
        query_chunk_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Exact CAM search for a batch of queries.

        This is the batched equivalent of calling search(query, top_k=top_k)
        in a Python loop, but it quantizes all queries together and computes the
        query-by-row distance matrix with one batched matrix multiply whenever
        the Torch fast path is available.

        Returns arrays with leading dimension num_queries. For example,
        result["indices"] has shape [num_queries, top_k].
        """
        q_cam = self.quantize_queries(queries)
        cam_distances = self._all_cam_distances_batch(
            q_cam, query_chunk_size=query_chunk_size
        )

        top_k = max(1, min(int(top_k), self.num_vectors))
        idx = np.argpartition(cam_distances, top_k - 1, axis=1)[:, :top_k]
        dist_top = np.take_along_axis(cam_distances, idx, axis=1)
        order = np.argsort(dist_top, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

        int_distances = np.take_along_axis(cam_distances, idx, axis=1)
        float_distances_top = self._cam_distances_to_float_distances(
            int_distances, q_cam=q_cam, row_indices=idx
        )
        ideal = self.ideal_scores_batch(queries)

        return {
            "indices": idx,
            "int_distances": int_distances,
            "float_distances": float_distances_top,
            "similarities": (1.0 - 0.5 * float_distances_top).astype(np.float32),
            "ideal_scores": np.take_along_axis(ideal, idx, axis=1),
            "vectors": self.vectors[idx],
            "normalized_weights": self.normalized_weights[idx],
        }

    def broadcast(self, query: Sequence[float], *, top_k: int = 5) -> CAMSearchResult:
        """
        Alias for search(). The name reflects CAM-style query broadcasting.
        """
        return self.search(query, top_k=top_k)

    def threshold_search(self, query: Sequence[float], threshold: float) -> np.ndarray:
        """
        Return indices whose CAM similarity is greater than or equal to threshold.
        """
        q_cam = self.quantize_query(query)
        cam_distances = self._all_cam_distances(q_cam)
        similarities = self._cam_distances_to_similarities(cam_distances, q_cam=q_cam)
        return np.flatnonzero(similarities >= float(threshold))

    def vote_search(self, query: Sequence[float], *, top_k: int = 5) -> Dict[str, Any]:
        """
        Subarray winner-takes-all voting.

        Each column subarray picks one local winning row. Final ranking is by
        vote count, with full CAM distance used as the tie-breaker.
        """
        q_cam = self.quantize_query(query)
        votes = np.zeros(self.num_vectors, dtype=np.int32)
        distance_dtype = self._numpy_distance_dtype()

        for col_slice in self.cam_col_slices:
            block = self.cam_int_vectors[:, col_slice].astype(distance_dtype, copy=False)
            q_block = q_cam[col_slice].astype(distance_dtype, copy=False)
            local_distances = np.sum((block - q_block) ** 2, axis=1)
            winner = int(np.argmin(local_distances))
            votes[winner] += 1

        exact_distances = self._all_cam_distances(q_cam)
        order = np.lexsort((exact_distances, -votes))

        top_k = max(1, min(int(top_k), self.num_vectors))
        idx = order[:top_k]
        int_distances = exact_distances[idx]
        float_distances = self._cam_distances_to_float_distances(
            int_distances, q_cam=q_cam, row_indices=idx
        )

        return {
            "indices": idx,
            "votes": votes[idx],
            "int_distances": int_distances,
            "float_distances": float_distances,
            "similarities": (1.0 - 0.5 * float_distances).astype(np.float32),
            "ideal_scores": self.ideal_scores(query)[idx],
            "vectors": self.vectors[idx],
            "normalized_weights": self.normalized_weights[idx],
        }

    def show_vote_internals(
        self,
        query: Sequence[float],
        *,
        top_k: int = 5,
        local_top_k: int = 5,
        include_all_vectors: bool = True,
    ) -> Dict[str, Any]:
        """
        Explain one CAM-style voting broadcast in detail.

        This is a diagnostic/teaching helper for the voting search path. It
        broadcasts one query to every CAM row, lets each physical column
        subarray vote for its nearest row, ranks rows by vote count with full
        CAM distance as the tie-breaker, and returns the internals needed to
        inspect both the voting behavior and quantization effects.

        Returned highlights include:

        * final top_k rows from the vote search;
        * per-subarray winners and per-subarray local top rows;
        * per-subarray vote strength for rows, where 1.0 is the local winner
          and 0.0 is the weakest row in that subarray for this query;
        * stored CAM state values, dequantized CAM values, pre-quantization
          float CAM values, and quantization errors for the query and winners.

        If include_all_vectors=True, full per-subarray distance and strength
        matrices with shape [num_subarrays, num_vectors] are included. Set it
        to False for a smaller result that only contains compact summaries and
        top_k winner details.
        """
        q = self._as_float32_numpy(query)

        if q.ndim != 1 or q.shape[0] != self.original_dim:
            raise ValueError(f"query must have shape [{self.original_dim}]")

        q_normalized = self._normalize_vector(q)
        q_cam_float = self._build_augmented_query(q_normalized)
        q_cam_float = self._normalize_vector(q_cam_float)
        q_cam = self._quantize_query_cam(q_cam_float) if self.quantize_vectors else q_cam_float

        if self.quantize_vectors:
            if self.quant_min is None or self.quant_step is None:
                raise RuntimeError("quantization parameters are missing")
            q_cam_values = self._dequantize_query(q_cam)
            cam_values = self.dequantized_cam_vectors()
            cam_state_values = self.cam_int_vectors.copy()
            query_cam_state = q_cam.copy()
        else:
            q_cam_values = q_cam.astype(np.float32, copy=True)
            cam_values = self.cam_int_vectors.astype(np.float32, copy=True)
            cam_state_values = self.cam_int_vectors.astype(np.float32, copy=True)
            query_cam_state = q_cam.astype(np.float32, copy=True)

        num_subarrays = len(self.cam_col_slices)
        top_k = max(1, min(int(top_k), self.num_vectors))
        local_top_k = max(1, min(int(local_top_k), self.num_vectors))

        votes = np.zeros(self.num_vectors, dtype=np.int32)
        distance_dtype = self._numpy_distance_dtype()
        local_distance_dtype = self._numpy_distance_dtype()
        local_int_distances = np.empty(
            (num_subarrays, self.num_vectors), dtype=local_distance_dtype
        )
        local_float_distances = np.empty(
            (num_subarrays, self.num_vectors), dtype=np.float32
        )
        local_vote_strengths = np.empty(
            (num_subarrays, self.num_vectors), dtype=np.float32
        )
        subarray_winner_indices = np.empty(num_subarrays, dtype=np.int64)
        subarrays = []

        for group_id, col_slice in enumerate(self.cam_col_slices):
            block = self.cam_int_vectors[:, col_slice].astype(
                distance_dtype, copy=False
            )
            q_block = q_cam[col_slice].astype(distance_dtype, copy=False)
            local_distances = np.sum((block - q_block) ** 2, axis=1)
            local_float = self._cam_distances_to_float_distances(
                local_distances, q_cam=q_cam, col_slice=col_slice
            )

            local_int_distances[group_id] = local_distances
            local_float_distances[group_id] = local_float

            min_local = float(np.min(local_float))
            max_local = float(np.max(local_float))
            if max_local > min_local:
                strength = (max_local - local_float) / (max_local - min_local)
            else:
                strength = np.ones_like(local_float, dtype=np.float32)
            local_vote_strengths[group_id] = strength.astype(np.float32, copy=False)

            winner = int(np.argmin(local_distances))
            subarray_winner_indices[group_id] = winner
            votes[winner] += 1

            local_idx = np.argpartition(local_distances, local_top_k - 1)[:local_top_k]
            local_idx = local_idx[np.argsort(local_distances[local_idx])]
            subarrays.append(
                {
                    "subarray_index": int(group_id),
                    "columns": (int(col_slice.start), int(col_slice.stop)),
                    "winner_index": winner,
                    "winner_local_int_distance": local_distances[winner].item(),
                    "winner_local_float_distance": float(local_float[winner]),
                    "winner_vote_strength": float(strength[winner]),
                    "local_top_indices": local_idx,
                    "local_top_int_distances": local_distances[local_idx],
                    "local_top_float_distances": local_float[local_idx],
                    "local_top_vote_strengths": strength[local_idx],
                }
            )

        exact_distances = self._all_cam_distances(q_cam)
        float_distances = self._cam_distances_to_float_distances(exact_distances, q_cam=q_cam)
        ideal_distances = self.ideal_distances(query)
        ideal_scores = self.ideal_scores(query)
        order = np.lexsort((exact_distances, -votes))
        top_indices = order[:top_k]

        top = {
            "indices": top_indices,
            "votes": votes[top_indices],
            "vote_fraction": (votes[top_indices] / float(num_subarrays)).astype(
                np.float32
            ),
            "int_distances": exact_distances[top_indices],
            "float_distances": float_distances[top_indices],
            "ideal_distances": ideal_distances[top_indices],
            "ideal_scores": ideal_scores[top_indices],
            "normalized_weights": self.normalized_weights[top_indices],
            "vectors": self.vectors[top_indices],
            "subarray_int_distances": local_int_distances[:, top_indices].T,
            "subarray_float_distances": local_float_distances[:, top_indices].T,
            "subarray_vote_strengths": local_vote_strengths[:, top_indices].T,
            "won_subarrays": (subarray_winner_indices[None, :] == top_indices[:, None]),
            "cam_state_values": cam_state_values[top_indices],
            "cam_values": cam_values[top_indices],
            "cam_float_values": self.cam_float_vectors[top_indices].copy(),
            "cam_quantization_error": (
                cam_values[top_indices] - self.cam_float_vectors[top_indices]
            ).astype(np.float32),
        }

        final_winners = []
        for rank, index in enumerate(top_indices):
            i = int(index)
            final_winners.append(
                {
                    "rank": int(rank + 1),
                    "index": i,
                    "votes": int(votes[i]),
                    "vote_fraction": float(votes[i] / float(num_subarrays)),
                    "int_distance": exact_distances[i].item(),
                    "float_distance": float(float_distances[i]),
                    "ideal_distance": float(ideal_distances[i]),
                    "ideal_score": float(ideal_scores[i]),
                    "normalized_weight": float(self.normalized_weights[i]),
                    "vector": self.vectors[i].copy(),
                    "won_subarrays": np.flatnonzero(
                        subarray_winner_indices == i
                    ).astype(np.int64),
                    "subarray_int_distances": local_int_distances[:, i].copy(),
                    "subarray_float_distances": local_float_distances[:, i].copy(),
                    "subarray_vote_strengths": local_vote_strengths[:, i].copy(),
                    "cam_state_values": cam_state_values[i].copy(),
                    "cam_values": cam_values[i].copy(),
                    "cam_float_values": self.cam_float_vectors[i].copy(),
                    "cam_quantization_error": (
                        cam_values[i] - self.cam_float_vectors[i]
                    ).astype(np.float32),
                }
            )

        result = {
            "query": {
                "normalized_original_values": q_normalized,
                "cam_state_values": query_cam_state,
                "cam_values": q_cam_values,
                "cam_float_values": q_cam_float,
                "cam_quantization_error": (q_cam_values - q_cam_float).astype(
                    np.float32
                ),
                "quant_min": None
                if self.query_quant_min is None
                else float(self.query_quant_min),
                "quant_max": None
                if self.query_quant_max is None
                else float(self.query_quant_max),
                "quant_step": None
                if self.query_quant_step is None
                else float(self.query_quant_step),
            },
            "top": top,
            "final_winners": final_winners,
            "subarray_winner_indices": subarray_winner_indices,
            "subarray_winner_cam_state_values": cam_state_values[
                subarray_winner_indices
            ],
            "subarray_winner_cam_values": cam_values[subarray_winner_indices],
            "subarray_winner_cam_float_values": self.cam_float_vectors[
                subarray_winner_indices
            ].copy(),
            "subarray_winner_cam_quantization_error": (
                cam_values[subarray_winner_indices]
                - self.cam_float_vectors[subarray_winner_indices]
            ).astype(np.float32),
            "subarrays": subarrays,
            "vote_counts_all_vectors": votes,
            "ranking_all_vectors": order,
            "metadata": {
                "num_vectors": int(self.num_vectors),
                "original_dim": int(self.original_dim),
                "cam_dim": int(self.cam_dim),
                "num_subarrays": int(num_subarrays),
                "top_k": int(top_k),
                "local_top_k": int(local_top_k),
                "quantize_vectors": bool(self.quantize_vectors),
                "bits": None if self.qmax is None else int(self.bits),
                "qmax": None if self.qmax is None else int(self.qmax),
                "quant_min": None
                if self.quant_min is None
                else float(self.quant_min),
                "quant_max": None
                if self.quant_max is None
                else float(self.quant_max),
                "quant_step": None
                if self.quant_step is None
                else float(self.quant_step),
                "separate_query_quantization": bool(
                    self.separate_query_quantization
                ),
                "query_quant_range_mode": self.query_quant_range_mode,
                "query_quant_min": None
                if self.query_quant_min is None
                else float(self.query_quant_min),
                "query_quant_max": None
                if self.query_quant_max is None
                else float(self.query_quant_max),
                "query_quant_step": None
                if self.query_quant_step is None
                else float(self.query_quant_step),
            },
        }

        if include_all_vectors:
            result["all_vectors"] = {
                "indices": np.arange(self.num_vectors, dtype=np.int64),
                "votes": votes.copy(),
                "vote_fraction": (votes / float(num_subarrays)).astype(np.float32),
                "subarray_int_distances": local_int_distances,
                "subarray_float_distances": local_float_distances,
                "subarray_vote_strengths": local_vote_strengths,
                "exact_int_distances": exact_distances,
                "exact_float_distances": float_distances,
                "ideal_distances": ideal_distances,
                "ideal_scores": ideal_scores,
            }

        return result

    def vote_search_batch(
        self,
        queries: Sequence[Sequence[float]],
        *,
        top_k: int = 5,
        query_chunk_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Batched subarray winner-takes-all voting.

        This is the optimized replacement for repeatedly calling
        vote_search(query, top_k=top_k) in a token-by-token Python loop. It
        performs one batched query quantization step and, when use_torch=True
        and bits <= 8, keeps the subarray voting and tie-break distance work on
        the Torch device.

        Returns arrays with leading dimension num_queries. For example,
        result["indices"] and result["votes"] have shape
        [num_queries, top_k].
        """
        q_cam = self.quantize_queries(queries)
        top_k = max(1, min(int(top_k), self.num_vectors))

        if self.use_torch and self._torch_can_use_fast_distances:
            return self._vote_search_batch_torch(
                q_cam, queries, top_k=top_k, query_chunk_size=query_chunk_size
            )

        if self.use_torch:
            self._warn_about_numpy_fallback()

        votes = self._subarray_votes_batch_numpy(q_cam)
        exact_distances = self._all_cam_distances_batch(
            q_cam, query_chunk_size=query_chunk_size
        )
        ideal = self.ideal_scores_batch(queries)

        idx = np.empty((q_cam.shape[0], top_k), dtype=np.int64)
        for row in range(q_cam.shape[0]):
            order = np.lexsort((exact_distances[row], -votes[row]))
            idx[row] = order[:top_k]

        int_distances = np.take_along_axis(exact_distances, idx, axis=1)
        float_distances = self._cam_distances_to_float_distances(
            int_distances, q_cam=q_cam, row_indices=idx
        )

        return {
            "indices": idx,
            "votes": np.take_along_axis(votes, idx, axis=1),
            "int_distances": int_distances,
            "float_distances": float_distances,
            "similarities": (1.0 - 0.5 * float_distances).astype(np.float32),
            "ideal_scores": np.take_along_axis(ideal, idx, axis=1),
            "vectors": self.vectors[idx],
            "normalized_weights": self.normalized_weights[idx],
        }

    def quantize_query(self, query: Sequence[float]) -> np.ndarray:
        """
        Normalize and augment one query. If quantize_vectors=True, also quantize it.
        """
        q = self._as_float32_numpy(query)

        if q.ndim != 1 or q.shape[0] != self.original_dim:
            raise ValueError(f"query must have shape [{self.original_dim}]")

        q = self._normalize_vector(q)
        q_aug = self._build_augmented_query(q)
        q_aug = self._normalize_vector(q_aug)

        if not self.quantize_vectors:
            return q_aug.astype(np.float32, copy=False)

        return self._quantize_query_cam(q_aug)

    def quantize_queries(self, queries: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Normalize, augment, and optionally quantize a batch of queries.

        Accepts either [num_queries, original_dim] or a single [original_dim]
        vector. A single vector is returned with shape [1, cam_dim].
        """
        q = self._as_float32_numpy(queries)

        if q.ndim == 1:
            q = q[None, :]

        if q.ndim != 2 or q.shape[1] != self.original_dim:
            raise ValueError(
                f"queries must have shape [num_queries, {self.original_dim}]"
            )

        q = self._normalize_rows(q)
        q_aug = self._build_augmented_queries(q)
        q_aug = self._normalize_rows(q_aug)

        if not self.quantize_vectors:
            return q_aug.astype(np.float32, copy=False)

        return self._quantize_query_cam(q_aug)

    def ideal_scores(self, query: Sequence[float]) -> np.ndarray:
        """
        Float reference scores before CAM quantization.

        Higher is better:

            normalized_abs_weight_i * cosine(query, sign(weight_i) * vector_i)

        This is equivalent to the signed target score:

            (weight_i / max(abs(weight))) * cosine(query, original_vector_i)
        """
        if self.prequantized_cam_vectors:
            return np.full(self.num_vectors, np.nan, dtype=np.float32)

        q = self._as_float32_numpy(query)

        if q.ndim != 1 or q.shape[0] != self.original_dim:
            raise ValueError(f"query must have shape [{self.original_dim}]")

        q = self._normalize_vector(q)

        return (self.normalized_weights * (self.vectors @ q)).astype(np.float32)

    def ideal_scores_batch(self, queries: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Float reference scores for a batch of queries before CAM quantization.

        Returns an array with shape [num_queries, num_vectors]. Higher is
        better.
        """
        q = self._as_float32_numpy(queries)

        if q.ndim == 1:
            q = q[None, :]

        if q.ndim != 2 or q.shape[1] != self.original_dim:
            raise ValueError(
                f"queries must have shape [num_queries, {self.original_dim}]"
            )

        if self.prequantized_cam_vectors:
            return np.full((q.shape[0], self.num_vectors), np.nan, dtype=np.float32)

        q = self._normalize_rows(q)
        scores = q @ self.vectors.T
        scores *= self.normalized_weights[None, :]
        return scores.astype(np.float32, copy=False)

    def ideal_distances(self, query: Sequence[float]) -> np.ndarray:
        """
        Float Euclidean distances after augmentation but before quantization.

        Lower is better.
        """
        if self.prequantized_cam_vectors:
            return np.full(self.num_vectors, np.nan, dtype=np.float32)

        q = self._as_float32_numpy(query)

        if q.ndim != 1 or q.shape[0] != self.original_dim:
            raise ValueError(f"query must have shape [{self.original_dim}]")

        q = self._normalize_vector(q)
        q_aug = self._build_augmented_query(q)
        q_aug = self._normalize_vector(q_aug)

        diff = self.cam_float_vectors - q_aug[None, :]
        return np.sum(diff * diff, axis=1).astype(np.float32)

    def dequantized_cam_vectors(self) -> np.ndarray:
        """
        Return quantized CAM rows mapped back to approximate float values.
        If quantize_vectors=False, return the stored float CAM rows.
        """
        if not self.quantize_vectors:
            return self.cam_int_vectors.astype(np.float32, copy=True)

        if self.quant_min is None or self.quant_step is None:
            raise RuntimeError("quantization parameters are missing")

        return self._dequantize_stored(self.cam_int_vectors)

    def dequantized_query(self, query: Sequence[float]) -> np.ndarray:
        """
        Return one query's quantized CAM states mapped back to approximate
        float values using the query quantization range.
        """
        q_cam = self.quantize_query(query)

        if not self.quantize_vectors:
            return q_cam.astype(np.float32, copy=True)

        return self._dequantize_query(q_cam)

    def info(self) -> Dict[str, Any]:
        """
        Return basic CAM, quantization, and subarray metadata.
        """
        return {
            "num_vectors": self.num_vectors,
            "original_dim": self.original_dim,
            "cam_dim": self.cam_dim,
            "quantize_vectors": self.quantize_vectors,
            "prequantized_cam_vectors": self.prequantized_cam_vectors,
            "query_quant_scale": self.query_quant_scale,
            "query_quant_zero_point": self.query_quant_zero_point,
            "bits": self.bits,
            "qmax": None if self.qmax is None else int(self.qmax),
            "max_subarray_cols": self.max_subarray_cols,
            "num_col_subarrays": len(self.cam_col_slices),
            "pad_per_subarray": self.pad_per_subarray,
            "num_pad_dims": self.num_pad_dims,
            "quant_range_mode": self.quant_range_mode,
            "quant_symmetric": self.quant_symmetric,
            "quant_min": None if self.quant_min is None else float(self.quant_min),
            "quant_max": None if self.quant_max is None else float(self.quant_max),
            "quant_step": None if self.quant_step is None else float(self.quant_step),
            "separate_query_quantization": self.separate_query_quantization,
            "query_quant_range_mode": self.query_quant_range_mode,
            "query_quant_symmetric": self.query_quant_symmetric,
            "query_quant_min": None
            if self.query_quant_min is None
            else float(self.query_quant_min),
            "query_quant_max": None
            if self.query_quant_max is None
            else float(self.query_quant_max),
            "query_quant_step": None
            if self.query_quant_step is None
            else float(self.query_quant_step),
            "num_query_quantization_examples": 0
            if self.query_quantization_examples is None
            else int(self.query_quantization_examples.shape[0]),
            "use_torch": self.use_torch,
            "torch_device": self.torch_device,
            "torch_chunk_rows": self.torch_chunk_rows,
            "torch_fast_quantized_bits": self.torch_fast_quantized_bits,
            "torch_distance_dtype": self.torch_distance_dtype,
            "torch_active_distance_dtype": None
            if self._torch_distance_torch_dtype is None
            else str(self._torch_distance_torch_dtype).replace("torch.", ""),
            "torch_fast_distances": self._torch_can_use_fast_distances,
            "torch_returns_integer_distances": self._torch_returns_integer_distances,
            "torch_distance_is_approximate": self._torch_distance_is_approximate,
            "strict_integer_distances": self.strict_integer_distances,
            "int64_exact_distances_safe": self._int64_exact_distances_safe,
            "float64_exact_distances_safe": self._float64_exact_distances_safe,
            "num_negative_weights": int(np.sum(self.raw_weights < 0.0)),
            "min_raw_weight": float(np.min(self.raw_weights)),
            "max_raw_weight": float(np.max(self.raw_weights)),
            "max_abs_raw_weight": float(np.max(self.abs_weights)),
            "min_normalized_weight": float(np.min(self.normalized_weights)),
            "max_normalized_weight": float(np.max(self.normalized_weights)),
            "min_signed_normalized_weight": float(
                np.min(self.signed_normalized_weights)
            ),
            "max_signed_normalized_weight": float(
                np.max(self.signed_normalized_weights)
            ),
        }

    @staticmethod
    def _as_float32_numpy(x: Any) -> np.ndarray:
        """
        Convert lists, NumPy arrays, or Torch tensors to float32 NumPy arrays.
        """
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(np.float32, copy=False)
        except ImportError:
            pass

        return np.asarray(x, dtype=np.float32)

    def _normalize_rows(self, x: np.ndarray) -> np.ndarray:
        """
        L2-normalize every row in a 2D array.
        """
        norms = np.linalg.norm(x, axis=1, keepdims=True)

        if np.any(norms < self.eps):
            raise ValueError("cannot normalize a zero or near-zero vector")

        return (x / norms).astype(np.float32)

    def _normalize_vector(self, x: np.ndarray) -> np.ndarray:
        """
        L2-normalize one vector.
        """
        norm = float(np.linalg.norm(x))

        if norm < self.eps:
            raise ValueError("cannot normalize a zero or near-zero vector")

        return (x / norm).astype(np.float32)

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize nonnegative weight magnitudes by max magnitude.
        """
        max_weight = float(np.max(weights))

        if max_weight < self.eps:
            return np.zeros_like(weights, dtype=np.float32)

        return (weights / max_weight).astype(np.float32)

    def _make_original_slices(self) -> list[slice]:
        """
        Split original vector dimensions into subarray-sized chunks.
        """
        if self.pad_per_subarray:
            body_cols = self.max_subarray_cols - self.num_pad_dims
        else:
            body_cols = self.max_subarray_cols

        return [
            slice(start, min(start + body_cols, self.original_dim))
            for start in range(0, self.original_dim, body_cols)
        ]

    def _make_cam_col_slices(self) -> list[slice]:
        """
        Split final CAM dimensions into physical column subarrays.
        """
        return [
            slice(start, min(start + self.max_subarray_cols, self.cam_dim))
            for start in range(0, self.cam_dim, self.max_subarray_cols)
        ]

    def _max_possible_state_distance(self) -> int:
        """
        Conservative upper bound on summed squared integer CAM-state distance.
        """
        if not self.quantize_vectors or self.qmax is None:
            return 0

        return int(self.cam_dim) * int(self.qmax) * int(self.qmax)

    def _integer_distances_fit_int64(self) -> bool:
        """
        Whether exact integer squared distances fit in signed int64 arithmetic.
        """
        if not self.quantize_vectors or self.qmax is None:
            return True

        int64_max = np.iinfo(np.int64).max
        per_cell = int(self.qmax) * int(self.qmax)
        return per_cell <= int64_max and self._max_possible_state_distance() <= int64_max

    def _integer_distances_fit_float64(self) -> bool:
        """
        Whether integer CAM-state distances are exactly representable in float64.
        """
        if not self.quantize_vectors or self.qmax is None:
            return True

        # IEEE-754 float64 exactly represents all integers up to 2**53.
        return self._max_possible_state_distance() <= (1 << 53)

    def _numpy_distance_dtype(self):
        """
        NumPy dtype used for fallback distance arithmetic.

        int64 is used whenever it is safe. For very wide quantized states,
        fixed-width exact integer accumulation would overflow, so the fallback
        uses float64 unless strict_integer_distances=True already rejected the
        configuration during initialization.
        """
        if not self.quantize_vectors:
            return np.float32

        if self._int64_exact_distances_safe:
            return np.int64

        return np.float64

    def _build_augmented_vectors(self) -> np.ndarray:
        """
        Build weighted, padded CAM vectors before quantization.
        """
        alpha = self.normalized_weights.astype(np.float32)

        if self.pad_per_subarray:
            pieces = []

            for s in self.original_slices:
                x_slice = self.vectors[:, s]
                body = alpha[:, None] * x_slice
                body_norm_sq = np.sum(body * body, axis=1)

                pad_sq = 1.0 - body_norm_sq
                pad_total = np.sqrt(np.maximum(pad_sq, 0.0)).astype(np.float32)
                pad_each = pad_total[:, None] / np.sqrt(float(self.num_pad_dims))

                pad = np.repeat(pad_each, self.num_pad_dims, axis=1)
                pieces.append(np.concatenate([body, pad], axis=1))

            return np.concatenate(pieces, axis=1).astype(np.float32)

        body = alpha[:, None] * self.vectors

        pad_sq = 1.0 - alpha * alpha
        pad_total = np.sqrt(np.maximum(pad_sq, 0.0)).astype(np.float32)
        pad_each = pad_total[:, None] / np.sqrt(float(self.num_pad_dims))
        pad = np.repeat(pad_each, self.num_pad_dims, axis=1)

        return np.concatenate([body, pad], axis=1).astype(np.float32)

    def _build_augmented_query(self, q: np.ndarray) -> np.ndarray:
        """
        Append zero padding dimensions to the query.
        """
        zeros = np.zeros(self.num_pad_dims, dtype=np.float32)

        if self.pad_per_subarray:
            pieces = []

            for s in self.original_slices:
                pieces.append(np.concatenate([q[s], zeros]))

            return np.concatenate(pieces).astype(np.float32)

        return np.concatenate([q, zeros]).astype(np.float32)

    def _build_augmented_queries(self, q: np.ndarray) -> np.ndarray:
        """
        Append zero padding dimensions to a batch of normalized queries.
        """
        zeros = np.zeros((q.shape[0], self.num_pad_dims), dtype=np.float32)

        if self.pad_per_subarray:
            pieces = []

            for s in self.original_slices:
                pieces.append(np.concatenate([q[:, s], zeros], axis=1))

            return np.concatenate(pieces, axis=1).astype(np.float32)

        return np.concatenate([q, zeros], axis=1).astype(np.float32)

    def _validate_query_examples(self, examples: Optional[np.ndarray], *, name: str) -> None:
        """
        Validate representative query/example arrays used for calibration.
        """
        if examples is None:
            return

        if examples.ndim != 2:
            raise ValueError(f"{name} must have shape [num_queries, vector_dim]")

        if examples.shape[1] != self.original_dim:
            raise ValueError(f"{name} must have dimension {self.original_dim}")

    def _augmented_queries_from_examples(
        self, examples: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Normalize and augment representative query vectors.
        """
        if examples is None:
            return None

        queries = self._normalize_rows(examples)
        q_aug = self._build_augmented_queries(queries)
        return self._normalize_rows(q_aug).astype(np.float32, copy=False)

    def _stored_quant_range_values(self) -> np.ndarray:
        """
        Values stored in CAM rows, before quantization.
        """
        return self.cam_float_vectors.reshape(-1).astype(np.float32, copy=False)

    def _stored_calibrated_quant_range_values(self) -> np.ndarray:
        """
        Stored CAM values plus optional calibration queries.
        """
        values = [self._stored_quant_range_values()]

        cal_q = self._augmented_queries_from_examples(self.calibration_queries)
        if cal_q is not None:
            values.append(cal_q.reshape(-1))

        return np.concatenate(values).astype(np.float32)

    def _query_quant_range_values(self) -> Optional[np.ndarray]:
        """
        Values used to choose the query quantization range.
        """
        examples = self._augmented_queries_from_examples(
            self.query_quantization_examples
        )
        if examples is None:
            return None

        return examples.reshape(-1).astype(np.float32, copy=False)

    def _choose_quant_range(
        self,
        *,
        mode: QuantRangeMode,
        quant_min: Optional[float],
        quant_max: Optional[float],
        values: Optional[np.ndarray],
        calibrated_values: Optional[np.ndarray],
        quant_clip_value: float,
        quant_percentile_value: float,
        quant_symmetric_value: bool,
        context: str,
    ) -> Tuple[np.float32, np.float32]:
        """
        Choose one quantization range for either stored CAM rows or queries.
        """
        if mode == "fixed":
            c = float(quant_clip_value)
            if c <= 0:
                raise ValueError(f"{context} quant_clip must be positive for fixed mode")
            lo, hi = -c, c

        elif mode == "stored_minmax":
            if values is None or values.size == 0:
                raise ValueError(
                    f"{context} examples are required for stored_minmax quantization"
                )
            lo = float(np.min(values))
            hi = float(np.max(values))

        elif mode == "calibrated_minmax":
            range_values = calibrated_values if calibrated_values is not None else values
            if range_values is None or range_values.size == 0:
                raise ValueError(
                    f"{context} examples are required for calibrated_minmax quantization"
                )
            lo = float(np.min(range_values))
            hi = float(np.max(range_values))

        elif mode == "percentile":
            range_values = calibrated_values if calibrated_values is not None else values
            if range_values is None or range_values.size == 0:
                raise ValueError(
                    f"{context} examples are required for percentile quantization"
                )
            c = float(np.percentile(np.abs(range_values), quant_percentile_value))
            c = max(c, self.eps)
            lo, hi = -c, c

        elif mode == "custom":
            if quant_min is None or quant_max is None:
                raise ValueError(f"{context} custom mode requires quant_min and quant_max")
            lo, hi = float(quant_min), float(quant_max)

        else:
            raise ValueError(f"Unknown quant_range_mode: {mode}")

        if quant_symmetric_value and mode not in {"fixed", "percentile"}:
            c = max(abs(lo), abs(hi), self.eps)
            lo, hi = -c, c

        return np.float32(lo), np.float32(hi)

    def _quantize_with_params(
        self,
        x: np.ndarray,
        *,
        quant_min: Optional[np.float32],
        quant_step: Optional[np.float32],
    ) -> np.ndarray:
        """
        Uniformly quantize with an explicit range.
        """
        if not self.quantize_vectors:
            return x.astype(np.float32, copy=True)

        if quant_min is None or quant_step is None or self.qmax is None:
            raise RuntimeError("quantization parameters are missing")

        calc_dtype = np.float64 if self.bits > 16 else np.float32
        q = np.rint((x.astype(calc_dtype, copy=False) - float(quant_min)) / float(quant_step))
        q = np.clip(q, 0, self.qmax)

        if self.bits <= 8:
            return q.astype(np.uint8)

        if self.bits <= 16:
            return q.astype(np.uint16)

        return q.astype(np.uint32)

    def _quantize_stored(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize stored CAM rows using the stored row range.
        """
        return self._quantize_with_params(
            x, quant_min=self.quant_min, quant_step=self.quant_step
        )

    def _quantize_query_cam(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize CAM queries using either the stored row range or query range.
        """
        if self.prequantized_cam_vectors:
            q = np.rint(
                x.astype(np.float64, copy=False) * float(self.query_quant_scale)
                + float(self.query_quant_zero_point)
            )
            return self._cast_like_prequantized_cam(q)

        return self._quantize_with_params(
            x, quant_min=self.query_quant_min, quant_step=self.query_quant_step
        )

    def _cast_like_prequantized_cam(self, x: np.ndarray) -> np.ndarray:
        """Cast direct-mode query states to the same dtype as stored CAM states."""
        dtype = self.cam_int_vectors.dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            x = np.clip(x, info.min, info.max)
        return x.astype(dtype, copy=False)

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Backward-compatible alias for stored-row quantization.
        """
        return self._quantize_stored(x)

    def _dequantize_with_params(
        self,
        x: np.ndarray,
        *,
        quant_min: Optional[np.float32],
        quant_step: Optional[np.float32],
    ) -> np.ndarray:
        """
        Map quantized CAM states back to approximate float values.
        """
        if not self.quantize_vectors:
            return x.astype(np.float32, copy=True)

        if quant_min is None or quant_step is None:
            raise RuntimeError("quantization parameters are missing")

        calc_dtype = np.float64 if self.bits > 16 else np.float32
        return float(quant_min) + x.astype(calc_dtype, copy=False) * float(quant_step)

    def _dequantize_stored(self, x: np.ndarray) -> np.ndarray:
        """
        Dequantize stored CAM rows with the stored row range.
        """
        return self._dequantize_with_params(
            x, quant_min=self.quant_min, quant_step=self.quant_step
        )

    def _dequantize_query(self, x: np.ndarray) -> np.ndarray:
        """
        Dequantize CAM query states with the query range.
        """
        return self._dequantize_with_params(
            x, quant_min=self.query_quant_min, quant_step=self.query_quant_step
        )

    def _all_cam_distances(self, q_cam: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distance to every CAM row.
        """
        if self.use_torch and self._torch_can_use_fast_distances:
            return self._all_cam_distances_torch(q_cam)

        if self.use_torch:
            self._warn_about_numpy_fallback()

        return self._all_cam_distances_numpy(q_cam)

    def _warn_about_numpy_fallback(self):
        if self.use_torch:
            warnings.warn(
                "use_torch=True was requested, but the Torch fast distance path "
                "is disabled; falling back to NumPy batch distances. Check "
                "cam.info()['torch_fast_distances'] for details.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _all_cam_distances_batch(
        self, q_cam: np.ndarray, *, query_chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute squared Euclidean distances for every query/CAM-row pair.
        """
        if q_cam.ndim != 2 or q_cam.shape[1] != self.cam_dim:
            raise ValueError(f"q_cam must have shape [num_queries, {self.cam_dim}]")

        if self.use_torch and self._torch_can_use_fast_distances:
            return self._all_cam_distances_batch_torch(
                q_cam, query_chunk_size=query_chunk_size
            )

        if self.use_torch:
            self._warn_about_numpy_fallback()

        return self._all_cam_distances_batch_numpy(
            q_cam, query_chunk_size=query_chunk_size
        )

    def _all_cam_distances_batch_numpy(
        self, q_cam: np.ndarray, *, query_chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        NumPy batched squared Euclidean distances.
        """
        if query_chunk_size is None:
            return self._pairwise_distances_numpy(q_cam, self.cam_int_vectors)

        query_chunk_size = max(1, int(query_chunk_size))
        chunks = []
        for start in range(0, q_cam.shape[0], query_chunk_size):
            end = min(start + query_chunk_size, q_cam.shape[0])
            chunks.append(self._pairwise_distances_numpy(q_cam[start:end], self.cam_int_vectors))

        return np.concatenate(chunks, axis=0)

    def _pairwise_distances_numpy(
        self, q_matrix: np.ndarray, row_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Pairwise squared Euclidean distances using the identity
        ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q x^T.
        """
        if self.quantize_vectors:
            distance_dtype = self._numpy_distance_dtype()
            q = q_matrix.astype(distance_dtype, copy=False)
            rows = row_matrix.astype(distance_dtype, copy=False)
            distances = (
                np.sum(q * q, axis=1)[:, None]
                + np.sum(rows * rows, axis=1)[None, :]
                - 2 * (q @ rows.T)
            )
            if distance_dtype is np.int64:
                return np.maximum(distances, 0).astype(np.int64, copy=False)
            return np.maximum(distances, 0.0).astype(np.float64, copy=False)

        q = q_matrix.astype(np.float32, copy=False)
        rows = row_matrix.astype(np.float32, copy=False)
        distances = (
            np.sum(q * q, axis=1)[:, None]
            + np.sum(rows * rows, axis=1)[None, :]
            - 2.0 * (q @ rows.T)
        )
        return np.maximum(distances, 0.0).astype(np.float32, copy=False)

    def _subarray_votes_batch_numpy(self, q_cam: np.ndarray) -> np.ndarray:
        """
        Compute subarray winner votes for a batch of queries with NumPy.
        """
        votes = np.zeros((q_cam.shape[0], self.num_vectors), dtype=np.int32)
        row_ids = np.arange(q_cam.shape[0])

        for col_slice in self.cam_col_slices:
            local_distances = self._pairwise_distances_numpy(
                q_cam[:, col_slice], self.cam_int_vectors[:, col_slice]
            )
            winners = np.argmin(local_distances, axis=1)
            votes[row_ids, winners] += 1

        return votes

    def _all_cam_distances_numpy(self, q_cam: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distance to every CAM row with NumPy.
        """
        distance_dtype = self._numpy_distance_dtype()
        distances = np.zeros(self.num_vectors, dtype=distance_dtype)

        for col_slice in self.cam_col_slices:
            block = self.cam_int_vectors[:, col_slice].astype(distance_dtype, copy=False)
            q_block = q_cam[col_slice].astype(distance_dtype, copy=False)
            diff = block - q_block
            distances += np.sum(diff * diff, axis=1)

        return distances

    def _setup_torch_cache(self) -> None:
        """
        Create an optional Torch distance cache for large searches.

        CAM_fast can keep the fast Torch path enabled for quantized states up
        to torch_fast_quantized_bits, including 16-, 24-, and 32-bit states.
        For bits > 8 this is a floating-point state-distance kernel, not a
        fixed-width exact integer kernel. Use strict_integer_distances=True to
        force the exact int64-safe path and reject configurations that would
        overflow exact int64 accumulation.
        """
        if not self.use_torch:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "use_torch=True requires PyTorch to be installed"
            ) from exc

        self._torch = torch
        device = torch.device(
            self.torch_device
            if self.torch_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_device = str(device)

        if self.quantize_vectors:
            if self.bits > self.torch_fast_quantized_bits:
                self._torch_can_use_fast_distances = False
                return

            if self.strict_integer_distances and self.bits > 8:
                # Keep the historical exact-int fallback for wider states.
                self._torch_can_use_fast_distances = False
                return

        if self.torch_distance_dtype == "float32":
            torch_dtype = torch.float32
            np_dtype = np.float32
        elif self.torch_distance_dtype == "float64":
            torch_dtype = torch.float64
            np_dtype = np.float64
        else:
            if self.quantize_vectors and self.bits > 8:
                torch_dtype = torch.float64
                np_dtype = np.float64
            else:
                torch_dtype = torch.float32
                np_dtype = np.float32

        self._torch_distance_torch_dtype = torch_dtype
        self._torch_distance_np_dtype = np_dtype

        # Preserve the historical <=8-bit behavior of returning rounded integer
        # state distances. For wider states, only claim integer distances when
        # auto/float64 arithmetic can represent the full state-distance range.
        self._torch_returns_integer_distances = bool(
            self.quantize_vectors
            and (
                self.bits <= 8
                or (torch_dtype is torch.float64 and self._integer_distances_fit_float64())
            )
        )
        self._torch_distance_is_approximate = bool(
            self.quantize_vectors
            and (
                self.bits > 8
                or not self._integer_distances_fit_float64()
                or torch_dtype is torch.float32
            )
            and not (self.bits <= 8)
        )

        matrix_np = self.cam_int_vectors.astype(np_dtype, copy=False)
        self._torch_distance_matrix = torch.as_tensor(
            matrix_np, dtype=torch_dtype, device=device
        )
        self._torch_distance_matrix_t = self._torch_distance_matrix.t().contiguous()
        self._torch_distance_norm_sq = torch.sum(
            self._torch_distance_matrix * self._torch_distance_matrix, dim=1
        )
        self._torch_can_use_fast_distances = True

    def _torch_distances_to_output(self, distances_np: np.ndarray) -> np.ndarray:
        """
        Convert Torch floating state distances to public NumPy distances.
        """
        if self.quantize_vectors and self._torch_returns_integer_distances:
            return np.rint(distances_np).astype(np.int64)

        if self.quantize_vectors:
            return distances_np.astype(
                np.float64
                if self._torch_distance_np_dtype is np.float64
                else np.float32,
                copy=False,
            )

        return distances_np.astype(np.float32, copy=False)

    def _all_cam_distances_torch(self, q_cam: np.ndarray) -> np.ndarray:
        """
        Compute all CAM distances using a Torch matrix-vector kernel.
        """
        if (
            self._torch is None
            or self._torch_distance_matrix is None
            or self._torch_distance_norm_sq is None
        ):
            raise RuntimeError("Torch distance cache is missing")

        torch = self._torch
        q = torch.as_tensor(
            q_cam.astype(self._torch_distance_np_dtype, copy=False),
            dtype=self._torch_distance_torch_dtype,
            device=self._torch_distance_matrix.device,
        )
        q_norm_sq = torch.dot(q, q)

        if self.torch_chunk_rows is None:
            distances = (
                self._torch_distance_norm_sq
                + q_norm_sq
                - 2.0 * torch.mv(self._torch_distance_matrix, q)
            )
        else:
            chunks = []
            for start in range(0, self.num_vectors, self.torch_chunk_rows):
                end = min(start + self.torch_chunk_rows, self.num_vectors)
                block = self._torch_distance_matrix[start:end]
                block_norm_sq = self._torch_distance_norm_sq[start:end]
                chunks.append(block_norm_sq + q_norm_sq - 2.0 * torch.mv(block, q))
            distances = torch.cat(chunks, dim=0)

        distances = torch.clamp(distances, min=0.0)
        distances_np = distances.detach().cpu().numpy()
        return self._torch_distances_to_output(distances_np)

    def _all_cam_distances_batch_torch(
        self, q_cam: np.ndarray, *, query_chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Batched Torch distance kernel.
        """
        distances = self._all_cam_distances_batch_torch_tensor(
            q_cam, query_chunk_size=query_chunk_size
        )
        distances_np = distances.detach().cpu().numpy()
        return self._torch_distances_to_output(distances_np)

    def _all_cam_distances_batch_torch_tensor(
        self, q_cam: np.ndarray, *, query_chunk_size: Optional[int] = None
    ):
        """
        Return batched Torch distances without moving them back to CPU.
        """
        if (
            self._torch is None
            or self._torch_distance_matrix is None
            or self._torch_distance_matrix_t is None
            or self._torch_distance_norm_sq is None
        ):
            raise RuntimeError("Torch distance cache is missing")

        torch = self._torch
        q = torch.as_tensor(
            q_cam.astype(self._torch_distance_np_dtype, copy=False),
            dtype=self._torch_distance_torch_dtype,
            device=self._torch_distance_matrix.device,
        )

        if query_chunk_size is None:
            q_norm_sq = torch.sum(q * q, dim=1)
            distances = (
                q_norm_sq[:, None]
                + self._torch_distance_norm_sq[None, :]
                - 2.0 * torch.mm(q, self._torch_distance_matrix_t)
            )
            return torch.clamp(distances, min=0.0)

        query_chunk_size = max(1, int(query_chunk_size))
        chunks = []
        for start in range(0, q.shape[0], query_chunk_size):
            q_chunk = q[start : start + query_chunk_size]
            q_norm_sq = torch.sum(q_chunk * q_chunk, dim=1)
            distances = (
                q_norm_sq[:, None]
                + self._torch_distance_norm_sq[None, :]
                - 2.0 * torch.mm(q_chunk, self._torch_distance_matrix_t)
            )
            chunks.append(torch.clamp(distances, min=0.0))

        return torch.cat(chunks, dim=0)

    def _vote_search_batch_torch(
        self,
        q_cam: np.ndarray,
        original_queries: Sequence[Sequence[float]],
        *,
        top_k: int,
        query_chunk_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Torch-backed batched subarray voting plus exact-distance tie-break.
        """
        if self._torch is None or self._torch_distance_matrix is None:
            raise RuntimeError("Torch distance cache is missing")

        torch = self._torch
        q = torch.as_tensor(
            q_cam.astype(self._torch_distance_np_dtype, copy=False),
            dtype=self._torch_distance_torch_dtype,
            device=self._torch_distance_matrix.device,
        )

        votes = torch.zeros(
            (q.shape[0], self.num_vectors),
            dtype=torch.int32,
            device=self._torch_distance_matrix.device,
        )
        row_ids = torch.arange(q.shape[0], device=self._torch_distance_matrix.device)

        for col_slice in self.cam_col_slices:
            block = self._torch_distance_matrix[:, col_slice]
            q_block = q[:, col_slice]
            block_norm_sq = torch.sum(block * block, dim=1)
            q_norm_sq = torch.sum(q_block * q_block, dim=1)
            local_distances = (
                q_norm_sq[:, None]
                + block_norm_sq[None, :]
                - 2.0 * torch.mm(q_block, block.t().contiguous())
            )
            winners = torch.argmin(local_distances, dim=1)
            votes[row_ids, winners] += 1

        exact = self._all_cam_distances_batch_torch_tensor(
            q_cam, query_chunk_size=query_chunk_size
        )

        # Rank primarily by vote count descending and secondarily by exact
        # distance ascending, matching np.lexsort((distance, -votes)).
        max_distance = torch.max(exact).to(torch.float64) + 1.0
        scores = votes.to(torch.float64) * max_distance - exact.to(torch.float64)
        _, idx_t = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=True)

        votes_top = torch.gather(votes, 1, idx_t).detach().cpu().numpy().astype(np.int32)
        exact_top_t = torch.gather(exact, 1, idx_t)
        exact_top_np = exact_top_t.detach().cpu().numpy()
        idx = idx_t.detach().cpu().numpy()

        int_distances = self._torch_distances_to_output(exact_top_np)

        float_distances = self._cam_distances_to_float_distances(
            int_distances, q_cam=q_cam, row_indices=idx
        )
        ideal = self.ideal_scores_batch(original_queries)

        return {
            "indices": idx,
            "votes": votes_top,
            "int_distances": int_distances,
            "float_distances": float_distances,
            "ideal_scores": np.take_along_axis(ideal, idx, axis=1),
            "vectors": self.vectors[idx],
            "normalized_weights": self.normalized_weights[idx],
        }

    def _cam_distances_to_float_distances(
        self,
        cam_distances: np.ndarray,
        *,
        q_cam: Optional[np.ndarray] = None,
        col_slice: Optional[slice] = None,
        row_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert CAM state-space distances to approximate float squared distances.

        When stored rows and queries share one quantization range, this is the
        original cheap conversion, distance * quant_step**2. When queries use a
        separate quantization range, the integer state distance no longer has a
        single scale factor, so this computes the distance between dequantized
        stored row values and dequantized query values.
        """
        if not self.quantize_vectors:
            return cam_distances.astype(np.float32, copy=False)

        if not self.separate_query_quantization:
            if self.quant_step is None:
                raise RuntimeError("quant_step is missing for quantized CAM distances")
            return np.asarray(cam_distances, dtype=np.float64) * (float(self.quant_step) ** 2)

        if q_cam is None:
            raise RuntimeError(
                "q_cam is required to convert distances when query quantization is separate"
            )

        row_states = self.cam_int_vectors
        if col_slice is not None:
            row_states = row_states[:, col_slice]
            q_states = q_cam[..., col_slice]
        else:
            q_states = q_cam

        row_values_all = self._dequantize_stored(row_states)
        q_values_all = self._dequantize_query(q_states)

        if row_indices is not None:
            row_indices = np.asarray(row_indices)
            if q_values_all.ndim == 1:
                selected_rows = row_values_all[row_indices]
                diff = selected_rows - q_values_all[None, :]
                return np.sum(diff * diff, axis=-1).astype(np.float32)

            if row_indices.ndim != 2:
                raise ValueError("row_indices must be 2D for batched q_cam")

            out = np.empty(row_indices.shape, dtype=np.float32)
            for row in range(row_indices.shape[0]):
                selected_rows = row_values_all[row_indices[row]]
                diff = selected_rows - q_values_all[row][None, :]
                out[row] = np.sum(diff * diff, axis=-1)
            return out

        if q_values_all.ndim == 1:
            diff = row_values_all - q_values_all[None, :]
            return np.sum(diff * diff, axis=1).astype(np.float32)

        return self._pairwise_distances_float_numpy(q_values_all, row_values_all)

    def _pairwise_distances_float_numpy(
        self, q_matrix: np.ndarray, row_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Pairwise squared Euclidean distances for float matrices.
        """
        q = q_matrix.astype(np.float32, copy=False)
        rows = row_matrix.astype(np.float32, copy=False)
        distances = (
            np.sum(q * q, axis=1)[:, None]
            + np.sum(rows * rows, axis=1)[None, :]
            - 2.0 * (q @ rows.T)
        )
        return np.maximum(distances, 0.0).astype(np.float32, copy=False)

    def _cam_distances_to_similarities(
        self, cam_distances: np.ndarray, *, q_cam: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert raw CAM distances to CAM similarity scores.
        """
        float_distances = self._cam_distances_to_float_distances(
            cam_distances, q_cam=q_cam
        )
        return (1.0 - 0.5 * float_distances).astype(np.float32)


"""
Public API
==========

QuantizedEuclideanCAM(...)
    Builds a CAM from vectors and weights. Vectors are normalized,
    weights are normalized, weights are baked into the vectors, and padding
    dimensions are added. By default, final CAM rows are quantized; set
    quantize_vectors=False to keep CAM rows as float arrays.

search(query, top_k=5)
    Returns the top_k nearest CAM rows using exact summed squared Euclidean
    distance. Uses quantized distances by default, or float distances when
    quantize_vectors=False.

search_batch(queries, top_k=5, query_chunk_size=None)
    Batched equivalent of search(). Returns arrays with leading dimension
    num_queries and avoids token-by-token Python/CUDA overhead.

broadcast(query, top_k=5)
    Same as search(); named to match CAM query broadcasting terminology.

threshold_search(query, threshold)
    Computes CAM similarity for every stored vector and returns the indices
    whose similarity is greater than or equal to threshold. Similarity is
    computed from CAM squared Euclidean distance as:

        similarity = 1 - distance / 2

    If quantize_vectors=True, the threshold is applied to the quantized CAM
    approximation. If quantize_vectors=False, the threshold is applied to the
    float CAM distance.

vote_search(query, top_k=5)
    Simulates subarray-local winner-takes-all voting. Each subarray votes for
    one row, and rows are ranked by vote count.

show_vote_internals(query, top_k=5, local_top_k=5, include_all_vectors=True)
    Diagnostic helper for one voting broadcast. Returns final top rows,
    per-subarray vote strengths/distances, per-subarray winners, query CAM
    state values, winning-vector CAM state values, dequantized CAM values,
    pre-quantization float values, and quantization errors.

vote_search_batch(queries, top_k=5, query_chunk_size=None)
    Batched equivalent of vote_search(). This is the preferred API for token
    sequences because it avoids one Python/CUDA call per token.

quantize_query(query)
    Normalizes and augments a query. By default, also quantizes it into CAM
    integer states. If quantize_vectors=False, returns the float CAM query.

quantize_queries(queries)
    Batched version of quantize_query().

ideal_scores(query)
    Returns float weighted-cosine scores before CAM quantization. Higher is better.

ideal_distances(query)
    Returns float augmented Euclidean distances before quantization. Lower is better.

dequantized_cam_vectors()
    Converts stored integer CAM rows back into approximate float values.
    If quantize_vectors=False, returns the stored float CAM rows.

dequantized_query(query)
    Quantizes one query, then maps its CAM states back to approximate float
    values using the query quantization range. This is most useful when
    separate_query_quantization=True or query_quantization_examples is provided.

info()
    Returns CAM configuration metadata.


Initialization options
======================

bits
    Number of bits per CAM cell. Ignored when quantize_vectors=False.

quantize_vectors
    If True, quantize stored CAM vectors and queries into integer CAM states.
    If False, skip quantization entirely and keep CAM vectors and queries as
    float arrays. This makes the CAM behave like a simple NumPy-backed array.

max_subarray_cols
    Maximum number of CAM columns per subarray.

pad_per_subarray
    If False, padding dimensions are appended once to the full vector.
    If True, padding dimensions are appended to every subarray slice.

num_pad_dims
    Number of padding dimensions used to spread out the weight-padding term.
    Larger values reduce the chance that one large padding coordinate dominates
    quantization.

quant_range_mode
    Chooses the global quantization range. Ignored when quantize_vectors=False.

    "fixed":
        Use [-quant_clip, quant_clip].

    "stored_minmax":
        Use the min/max of stored CAM rows.

    "calibrated_minmax":
        Use the min/max of stored CAM rows and calibration queries.

    "percentile":
        Use a symmetric percentile range from stored rows and optional
        calibration queries.

    "custom":
        Use explicit quant_min and quant_max.

quant_clip
    Clip value for fixed mode. Ignored when quantize_vectors=False.

quant_percentile
    Percentile used for percentile mode. Ignored when quantize_vectors=False.

quant_symmetric
    If True, min/max modes are converted to a symmetric range [-c, c].
    Ignored when quantize_vectors=False.

quant_min, quant_max
    Explicit range used only when quant_range_mode="custom".
    Ignored when quantize_vectors=False.

calibration_queries
    Optional representative queries used to choose the stored CAM row
    quantization range when quant_range_mode is "calibrated_minmax" or
    "percentile". Ignored when quantize_vectors=False.

separate_query_quantization
    If True, quantize queries with their own range instead of reusing the stored
    CAM row range. Passing query_quantization_examples automatically enables
    this option. Search and voting still rank by integer CAM state distance;
    reported float_distances use dequantized stored-row and query values when
    the scales differ.

query_quantization_examples
    Optional representative examples of queries you expect to broadcast to the
    CAM. These are normalized, augmented, and used to choose the query-side
    quantization range. Required for separate query quantization with
    query_quant_range_mode="stored_minmax", "calibrated_minmax", or
    "percentile". Not required for "fixed" or "custom" query ranges.

query_quant_range_mode
    Query-side range mode. Defaults to quant_range_mode. Uses the same mode
    names as quant_range_mode, but applies them to query_quantization_examples
    instead of stored CAM rows.

query_quant_clip
    Query-side fixed-mode clip value. Defaults to quant_clip.

query_quant_percentile
    Query-side percentile. Defaults to quant_percentile.

query_quant_symmetric
    Query-side symmetric min/max behavior. Defaults to quant_symmetric.

query_quant_min, query_quant_max
    Explicit query-side range used only when query_quant_range_mode="custom".

use_torch
    If True, keep a Torch copy of the CAM rows and use matrix-vector or
    matrix-matrix distance kernels for search, batched search, threshold_search,
    and batched vote search. CAM_fast can also use this path for quantized
    states above 8 bits, up to torch_fast_quantized_bits.

torch_fast_quantized_bits
    Largest quantized bit width allowed to use the Torch fast path. Defaults
    to 32 in CAM_fast. Set this to 8 to recover the original conservative
    behavior.

torch_distance_dtype
    "auto", "float32", or "float64". In auto mode, CAM_fast uses float32
    for float and <=8-bit quantized distances, and float64 for wider
    quantized states. Wider quantized paths are fast approximate state-distance
    kernels unless the summed state distance is small enough to be represented
    exactly in float64.

strict_integer_distances
    If True, avoid approximate floating state-distance kernels for wider
    quantized states. Configurations whose exact summed squared distances would
    overflow signed int64 are rejected. Leave this False for the 32-bit fast
    path.

torch_device
    Optional Torch device string such as "cuda", "cuda:0", or "cpu". If
    omitted, CUDA is used when available; otherwise CPU is used. Ignored unless
    use_torch=True.

torch_chunk_rows
    Optional number of CAM rows to process per Torch chunk. Use this to reduce
    peak GPU memory during very large searches. Ignored unless use_torch=True.

eps
    Small numerical tolerance.
"""


# Example
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    vectors = rng.normal(size=(100, 768)).astype(np.float32)
    weights = rng.uniform(0.1, 5.0, size=100).astype(np.float32)
    queries = rng.normal(size=(256, 768)).astype(np.float32)

    cam = QuantizedEuclideanCAM(
        vectors,
        weights,
        bits=4,
        max_subarray_cols=64,
        pad_per_subarray=False,
        num_pad_dims=16,
        quant_range_mode="percentile",
        quant_percentile=99.9,
        calibration_queries=queries,
    )

    result = cam.search(queries[0], top_k=10)

    print("Quantized search")
    print(result.indices)
    print(result.int_distances)
    print(result.float_distances)
    print(result.ideal_scores)
    print(cam.info())

    threshold_indices = cam.threshold_search(queries[0], threshold=0.5)

    print("\nQuantized threshold search")
    print(threshold_indices)

    cam_float = QuantizedEuclideanCAM(
        vectors,
        weights,
        quantize_vectors=False,
        max_subarray_cols=64,
        pad_per_subarray=False,
        num_pad_dims=16,
    )

    float_result = cam_float.search(queries[0], top_k=10)

    print("\nFloat search, no quantization")
    print(float_result.indices)
    print(float_result.int_distances)
    print(float_result.float_distances)
    print(float_result.ideal_scores)
    print(cam_float.info())

    float_threshold_indices = cam_float.threshold_search(queries[0], threshold=0.5)

    print("\nFloat threshold search, no quantization")
    print(float_threshold_indices)

    cam_vote = QuantizedEuclideanCAM(
        vectors,
        weights,
        bits=4,
        max_subarray_cols=64,
        pad_per_subarray=True,
        num_pad_dims=4,
        quant_range_mode="percentile",
        quant_percentile=99.9,
        calibration_queries=queries,
    )

    cam_separate_query = QuantizedEuclideanCAM(
        vectors,
        weights,
        bits=4,
        max_subarray_cols=64,
        pad_per_subarray=True,
        num_pad_dims=4,
        quant_range_mode="percentile",
        quant_percentile=99.9,
        calibration_queries=queries,
        query_quantization_examples=queries,
        query_quant_range_mode="percentile",
        query_quant_percentile=99.9,
    )

    separate_query_result = cam_separate_query.search(queries[0], top_k=10)

    print("\nQuantized search with separate query scale")
    print(separate_query_result.indices)
    print(separate_query_result.int_distances)
    print(separate_query_result.float_distances)
    print(cam_separate_query.info())

    vote_result = cam_vote.vote_search(queries[0], top_k=10)

    print("\nQuantized vote search")
    print(vote_result["indices"])
    print(vote_result["votes"])
    print(vote_result["float_distances"])
    print(vote_result["ideal_scores"])

    internals = cam_vote.show_vote_internals(
        queries[0], top_k=5, local_top_k=5, include_all_vectors=False
    )

    print("\nVoting internals top 5")
    print(internals["top"]["indices"])
    print(internals["top"]["votes"])
    print(internals["top"]["subarray_vote_strengths"])
    print("Query CAM state values")
    print(internals["query"]["cam_state_values"])
    print("First winning vector CAM state values")
    print(internals["final_winners"][0]["cam_state_values"])
    print("First winning vector dequantized CAM values")
    print(internals["final_winners"][0]["cam_values"])
    print("First winning vector quantization error")
    print(internals["final_winners"][0]["cam_quantization_error"])
