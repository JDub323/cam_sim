from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np


QuantRangeMode = Literal[
    "fixed",
    "stored_minmax",
    "calibrated_minmax",
    "percentile",
    "custom",
]


@dataclass(frozen=True)
class CAMSearchResult:
    indices: np.ndarray
    int_distances: np.ndarray
    float_distances: np.ndarray
    ideal_scores: np.ndarray
    vectors: np.ndarray
    normalized_weights: np.ndarray


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
    distances directly on the float CAM vectors. Set use_torch=True to use a
    Torch-backed matrix-vector distance kernel for the common <=8-bit or float
    search path.
    """

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        weights: Sequence[float],
        *,
        bits: int = 4,
        quantize_vectors: bool = True,
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
        eps: float = 1e-8,
        use_torch: bool = False,
        torch_device: Optional[str] = None,
        torch_chunk_rows: Optional[int] = None,
    ) -> None:
        if quantize_vectors and bits < 1:
            raise ValueError("bits must be >= 1 when quantize_vectors=True")

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

        if torch_chunk_rows is not None and int(torch_chunk_rows) < 1:
            raise ValueError("torch_chunk_rows must be >= 1 when provided")

        self.quantize_vectors = bool(quantize_vectors)
        self.bits = int(bits)
        self.qmax = (1 << self.bits) - 1 if self.quantize_vectors else None
        self.max_subarray_cols = int(max_subarray_cols)
        self.pad_per_subarray = bool(pad_per_subarray)
        self.num_pad_dims = int(num_pad_dims)
        self.quant_range_mode = quant_range_mode
        self.quant_clip = float(quant_clip)
        self.quant_percentile = float(quant_percentile)
        self.quant_symmetric = bool(quant_symmetric)
        self.eps = float(eps)
        self.use_torch = bool(use_torch)
        self.torch_device = torch_device
        self.torch_chunk_rows = (
            None if torch_chunk_rows is None else int(torch_chunk_rows)
        )
        self._torch = None
        self._torch_distance_matrix = None
        self._torch_distance_matrix_t = None
        self._torch_distance_norm_sq = None
        self._torch_can_use_fast_distances = False

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

        self.calibration_queries = (
            None
            if (not self.quantize_vectors or calibration_queries is None)
            else self._as_float32_numpy(calibration_queries)
        )

        if self.calibration_queries is not None:
            if self.calibration_queries.ndim != 2:
                raise ValueError(
                    "calibration_queries must have shape [num_queries, vector_dim]"
                )
            if self.calibration_queries.shape[1] != self.original_dim:
                raise ValueError(
                    f"calibration_queries must have dimension {self.original_dim}"
                )

        if self.quantize_vectors:
            self.quant_min, self.quant_max = self._choose_quant_range(
                mode=quant_range_mode,
                quant_min=quant_min,
                quant_max=quant_max,
            )

            if self.quant_max <= self.quant_min:
                raise ValueError("Invalid quantization range")

            self.quant_step = np.float32((self.quant_max - self.quant_min) / self.qmax)
            self.cam_int_vectors = self._quantize(self.cam_float_vectors)
        else:
            self.quant_min = None
            self.quant_max = None
            self.quant_step = None
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
        float_distances = self._cam_distances_to_float_distances(cam_distances)

        top_k = max(1, min(int(top_k), self.num_vectors))
        idx = np.argpartition(cam_distances, top_k - 1)[:top_k]
        idx = idx[np.argsort(cam_distances[idx])]

        return CAMSearchResult(
            indices=idx,
            int_distances=cam_distances[idx],
            float_distances=float_distances[idx],
            ideal_scores=self.ideal_scores(query)[idx],
            vectors=self.vectors[idx],
            normalized_weights=self.normalized_weights[idx],
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
        float_distances = self._cam_distances_to_float_distances(cam_distances)

        top_k = max(1, min(int(top_k), self.num_vectors))
        idx = np.argpartition(cam_distances, top_k - 1, axis=1)[:, :top_k]
        dist_top = np.take_along_axis(cam_distances, idx, axis=1)
        order = np.argsort(dist_top, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

        int_distances = np.take_along_axis(cam_distances, idx, axis=1)
        float_distances_top = np.take_along_axis(float_distances, idx, axis=1)
        ideal = self.ideal_scores_batch(queries)

        return {
            "indices": idx,
            "int_distances": int_distances,
            "float_distances": float_distances_top,
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
        similarities = self._cam_distances_to_similarities(cam_distances)
        return np.flatnonzero(similarities >= float(threshold))

    def vote_search(self, query: Sequence[float], *, top_k: int = 5) -> Dict[str, Any]:
        """
        Subarray winner-takes-all voting.

        Each column subarray picks one local winning row. Final ranking is by
        vote count, with full CAM distance used as the tie-breaker.
        """
        q_cam = self.quantize_query(query)
        votes = np.zeros(self.num_vectors, dtype=np.int32)
        distance_dtype = np.int64 if self.quantize_vectors else np.float32

        for col_slice in self.cam_col_slices:
            block = self.cam_int_vectors[:, col_slice].astype(distance_dtype, copy=False)
            q_block = q_cam[col_slice].astype(distance_dtype, copy=False)
            local_distances = np.sum((block - q_block) ** 2, axis=1)
            winner = int(np.argmin(local_distances))
            votes[winner] += 1

        exact_distances = self._all_cam_distances(q_cam)
        float_distances = self._cam_distances_to_float_distances(exact_distances)
        order = np.lexsort((exact_distances, -votes))

        top_k = max(1, min(int(top_k), self.num_vectors))
        idx = order[:top_k]

        return {
            "indices": idx,
            "votes": votes[idx],
            "int_distances": exact_distances[idx],
            "float_distances": float_distances[idx],
            "ideal_scores": self.ideal_scores(query)[idx],
            "vectors": self.vectors[idx],
            "normalized_weights": self.normalized_weights[idx],
        }

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

        votes = self._subarray_votes_batch_numpy(q_cam)
        exact_distances = self._all_cam_distances_batch(
            q_cam, query_chunk_size=query_chunk_size
        )
        float_distances = self._cam_distances_to_float_distances(exact_distances)
        ideal = self.ideal_scores_batch(queries)

        idx = np.empty((q_cam.shape[0], top_k), dtype=np.int64)
        for row in range(q_cam.shape[0]):
            order = np.lexsort((exact_distances[row], -votes[row]))
            idx[row] = order[:top_k]

        return {
            "indices": idx,
            "votes": np.take_along_axis(votes, idx, axis=1),
            "int_distances": np.take_along_axis(exact_distances, idx, axis=1),
            "float_distances": np.take_along_axis(float_distances, idx, axis=1),
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

        return self._quantize(q_aug)

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

        return self._quantize(q_aug)

    def ideal_scores(self, query: Sequence[float]) -> np.ndarray:
        """
        Float reference scores before CAM quantization.

        Higher is better:

            normalized_abs_weight_i * cosine(query, sign(weight_i) * vector_i)

        This is equivalent to the signed target score:

            (weight_i / max(abs(weight))) * cosine(query, original_vector_i)
        """
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

        q = self._normalize_rows(q)
        scores = q @ self.vectors.T
        scores *= self.normalized_weights[None, :]
        return scores.astype(np.float32, copy=False)

    def ideal_distances(self, query: Sequence[float]) -> np.ndarray:
        """
        Float Euclidean distances after augmentation but before quantization.

        Lower is better.
        """
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

        return self.quant_min + self.cam_int_vectors.astype(np.float32) * self.quant_step

    def info(self) -> Dict[str, Any]:
        """
        Return basic CAM, quantization, and subarray metadata.
        """
        return {
            "num_vectors": self.num_vectors,
            "original_dim": self.original_dim,
            "cam_dim": self.cam_dim,
            "quantize_vectors": self.quantize_vectors,
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
            "use_torch": self.use_torch,
            "torch_device": self.torch_device,
            "torch_chunk_rows": self.torch_chunk_rows,
            "torch_fast_distances": self._torch_can_use_fast_distances,
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

    def _calibration_augmented_queries(self) -> Optional[np.ndarray]:
        """
        Build augmented calibration queries, if provided.
        """
        if self.calibration_queries is None:
            return None

        queries = self._normalize_rows(self.calibration_queries)
        augmented = []

        for q in queries:
            q_aug = self._build_augmented_query(q)
            q_aug = self._normalize_vector(q_aug)
            augmented.append(q_aug)

        return np.stack(augmented).astype(np.float32)

    def _choose_quant_range(
        self,
        *,
        mode: QuantRangeMode,
        quant_min: Optional[float],
        quant_max: Optional[float],
    ) -> Tuple[np.float32, np.float32]:
        """
        Choose one global quantization range shared by all CAM cells.
        """
        values = [self.cam_float_vectors.reshape(-1)]

        cal_q = self._calibration_augmented_queries()
        if cal_q is not None:
            values.append(cal_q.reshape(-1))

        all_values = np.concatenate(values).astype(np.float32)

        if mode == "fixed":
            c = float(self.quant_clip)
            if c <= 0:
                raise ValueError("quant_clip must be positive for fixed mode")
            lo, hi = -c, c

        elif mode == "stored_minmax":
            lo = float(np.min(self.cam_float_vectors))
            hi = float(np.max(self.cam_float_vectors))

        elif mode == "calibrated_minmax":
            lo = float(np.min(all_values))
            hi = float(np.max(all_values))

        elif mode == "percentile":
            c = float(np.percentile(np.abs(all_values), self.quant_percentile))
            c = max(c, self.eps)
            lo, hi = -c, c

        elif mode == "custom":
            if quant_min is None or quant_max is None:
                raise ValueError("custom mode requires quant_min and quant_max")
            lo, hi = float(quant_min), float(quant_max)

        else:
            raise ValueError(f"Unknown quant_range_mode: {mode}")

        if self.quant_symmetric and mode not in {"fixed", "percentile"}:
            c = max(abs(lo), abs(hi), self.eps)
            lo, hi = -c, c

        return np.float32(lo), np.float32(hi)

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Uniformly quantize using the chosen global range.
        """
        if not self.quantize_vectors:
            return x.astype(np.float32, copy=True)

        if self.quant_min is None or self.quant_step is None or self.qmax is None:
            raise RuntimeError("quantization parameters are missing")

        q = np.rint((x.astype(np.float32) - self.quant_min) / self.quant_step)
        q = np.clip(q, 0, self.qmax)

        if self.bits <= 8:
            return q.astype(np.uint8)

        if self.bits <= 16:
            return q.astype(np.uint16)

        return q.astype(np.uint32)

    def _all_cam_distances(self, q_cam: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distance to every CAM row.
        """
        if self.use_torch and self._torch_can_use_fast_distances:
            return self._all_cam_distances_torch(q_cam)

        return self._all_cam_distances_numpy(q_cam)

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
            q = q_matrix.astype(np.int64, copy=False)
            rows = row_matrix.astype(np.int64, copy=False)
            distances = (
                np.sum(q * q, axis=1)[:, None]
                + np.sum(rows * rows, axis=1)[None, :]
                - 2 * (q @ rows.T)
            )
            return np.maximum(distances, 0).astype(np.int64, copy=False)

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
        distance_dtype = np.int64 if self.quantize_vectors else np.float32
        distances = np.zeros(self.num_vectors, dtype=distance_dtype)

        for col_slice in self.cam_col_slices:
            block = self.cam_int_vectors[:, col_slice].astype(distance_dtype, copy=False)
            q_block = q_cam[col_slice].astype(distance_dtype, copy=False)
            diff = block - q_block
            distances += np.sum(diff * diff, axis=1)

        return distances

    def _setup_torch_cache(self) -> None:
        """
        Create an optional Torch distance cache for large matrix-vector searches.

        For quantized vectors, this cache is exact for the usual <=8-bit CAM
        states after rounding the float32 matrix-vector result back to integer
        distances. For wider integer states, NumPy's integer path is retained to
        avoid precision surprises.
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

        if self.quantize_vectors and self.bits > 8:
            # Keep the exact integer NumPy path for larger integer states.
            self._torch_can_use_fast_distances = False
            return

        matrix_np = self.cam_int_vectors.astype(np.float32, copy=False)
        self._torch_distance_matrix = torch.as_tensor(
            matrix_np, dtype=torch.float32, device=device
        )
        self._torch_distance_matrix_t = self._torch_distance_matrix.t().contiguous()
        self._torch_distance_norm_sq = torch.sum(
            self._torch_distance_matrix * self._torch_distance_matrix, dim=1
        )
        self._torch_can_use_fast_distances = True

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
            q_cam.astype(np.float32, copy=False),
            dtype=torch.float32,
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

        if self.quantize_vectors:
            return np.rint(distances_np).astype(np.int64)

        return distances_np.astype(np.float32, copy=False)

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

        if self.quantize_vectors:
            return np.rint(distances_np).astype(np.int64)

        return distances_np.astype(np.float32, copy=False)

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
            q_cam.astype(np.float32, copy=False),
            dtype=torch.float32,
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
            q_cam.astype(np.float32, copy=False),
            dtype=torch.float32,
            device=self._torch_distance_matrix.device,
        )

        votes = torch.zeros(
            (q.shape[0], self.num_vectors),
            dtype=torch.int16,
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

        if self.quantize_vectors:
            int_distances = np.rint(exact_top_np).astype(np.int64)
        else:
            int_distances = exact_top_np.astype(np.float32, copy=False)

        float_distances = self._cam_distances_to_float_distances(int_distances)
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

    def _cam_distances_to_float_distances(self, cam_distances: np.ndarray) -> np.ndarray:
        """
        Convert raw CAM distances to approximate float squared Euclidean distances.
        """
        if not self.quantize_vectors:
            return cam_distances.astype(np.float32, copy=False)

        if self.quant_step is None:
            raise RuntimeError("quant_step is missing for quantized CAM distances")

        return cam_distances.astype(np.float32) * (self.quant_step ** 2)

    def _cam_distances_to_similarities(self, cam_distances: np.ndarray) -> np.ndarray:
        """
        Convert raw CAM distances to CAM similarity scores.
        """
        float_distances = self._cam_distances_to_float_distances(cam_distances)
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
    Optional representative queries used to choose the quantization range.
    Ignored when quantize_vectors=False.

use_torch
    If True, keep a Torch float32 copy of the CAM rows and use a matrix-vector
    distance kernel for search and threshold_search. This can speed up large
    CAMs, especially on CUDA, at the cost of extra memory. For quantized CAMs,
    the Torch fast path is used for bits <= 8; wider integer states fall back
    to the exact NumPy integer path.

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

    vote_result = cam_vote.vote_search(queries[0], top_k=10)

    print("\nQuantized vote search")
    print(vote_result["indices"])
    print(vote_result["votes"])
    print(vote_result["float_distances"])
    print(vote_result["ideal_scores"])
