"""Comprehensive tests for rankops Python bindings (rankops Python bindings)."""

import pytest
import rankops
from typing import List, Tuple


# Test data fixtures
@pytest.fixture
def bm25_results() -> List[Tuple[str, float]]:
    """BM25 retrieval results."""
    return [("doc_A", 12.5), ("doc_B", 11.0), ("doc_C", 9.0)]


@pytest.fixture
def dense_results() -> List[Tuple[str, float]]:
    """Dense vector retrieval results."""
    return [("doc_B", 0.9), ("doc_D", 0.8), ("doc_A", 0.7)]


@pytest.fixture
def keyword_results() -> List[Tuple[str, float]]:
    """Keyword-based retrieval results."""
    return [("doc_C", 0.95), ("doc_E", 0.85), ("doc_A", 0.75)]


@pytest.fixture
def multi_lists(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    keyword_results: List[Tuple[str, float]],
) -> List[List[Tuple[str, float]]]:
    """Multiple retrieval lists for multi-list fusion."""
    return [bm25_results, dense_results, keyword_results]


# RRF Tests
class TestRRF:
    """Tests for Reciprocal Rank Fusion (RRF)."""

    def test_rrf_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic RRF fusion."""
        result = rankops.rrf(bm25_results, dense_results, k=60)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(isinstance(score, (int, float)) for _, score in result)

    def test_rrf_default_k(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test RRF with default k value."""
        result = rankops.rrf(bm25_results, dense_results)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_rrf_custom_k(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test RRF with custom k value."""
        result = rankops.rrf(bm25_results, dense_results, k=20)
        assert isinstance(result, list)

    def test_rrf_top_k(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test RRF with top_k parameter."""
        result = rankops.rrf(bm25_results, dense_results, k=60, top_k=2)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_rrf_k_zero_error(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test RRF with k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            rankops.rrf(bm25_results, dense_results, k=0)

    def test_rrf_empty_lists(self):
        """Test RRF with empty lists."""
        result = rankops.rrf([], [])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_rrf_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test RRF for multiple lists."""
        result = rankops.rrf_multi(multi_lists, k=60)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_rrf_multi_top_k(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test RRF multi with top_k."""
        result = rankops.rrf_multi(multi_lists, k=60, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_rrf_multi_empty(self):
        """Test RRF multi with empty list."""
        result = rankops.rrf_multi([])
        assert isinstance(result, list)
        assert len(result) == 0


# ISR Tests
class TestISR:
    """Tests for Inverse Square Rank (ISR)."""

    def test_isr_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic ISR fusion."""
        result = rankops.isr(bm25_results, dense_results, k=1)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_isr_default_k(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test ISR with default k."""
        result = rankops.isr(bm25_results, dense_results)
        assert isinstance(result, list)

    def test_isr_top_k(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test ISR with top_k."""
        result = rankops.isr(bm25_results, dense_results, k=1, top_k=2)
        assert len(result) <= 2

    def test_isr_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test ISR for multiple lists."""
        result = rankops.isr_multi(multi_lists, k=1)
        assert isinstance(result, list)


# CombSUM Tests
class TestCombSUM:
    """Tests for CombSUM fusion."""

    def test_combsum_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic CombSUM fusion."""
        result = rankops.combsum(bm25_results, dense_results)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_combsum_top_k(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test CombSUM with top_k."""
        result = rankops.combsum(bm25_results, dense_results, top_k=2)
        assert len(result) <= 2

    def test_combsum_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test CombSUM for multiple lists."""
        result = rankops.combsum_multi(multi_lists)
        assert isinstance(result, list)


# CombMNZ Tests
class TestCombMNZ:
    """Tests for CombMNZ fusion."""

    def test_combmnz_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic CombMNZ fusion."""
        result = rankops.combmnz(bm25_results, dense_results)
        assert isinstance(result, list)

    def test_combmnz_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test CombMNZ for multiple lists."""
        result = rankops.combmnz_multi(multi_lists)
        assert isinstance(result, list)


# Borda Tests
class TestBorda:
    """Tests for Borda count fusion."""

    def test_borda_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic Borda fusion."""
        result = rankops.borda(bm25_results, dense_results)
        assert isinstance(result, list)

    def test_borda_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test Borda for multiple lists."""
        result = rankops.borda_multi(multi_lists)
        assert isinstance(result, list)


# DBSF Tests
class TestDBSF:
    """Tests for DBSF fusion."""

    def test_dbsf_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic DBSF fusion."""
        result = rankops.dbsf(bm25_results, dense_results)
        assert isinstance(result, list)

    def test_dbsf_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test DBSF for multiple lists."""
        result = rankops.dbsf_multi(multi_lists)
        assert isinstance(result, list)


# Weighted Tests
class TestWeighted:
    """Tests for weighted fusion."""

    def test_weighted_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic weighted fusion."""
        result = rankops.weighted(
            bm25_results, dense_results, weight_a=0.7, weight_b=0.3, normalize=True
        )
        assert isinstance(result, list)

    def test_weighted_no_normalize(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test weighted fusion without normalization."""
        result = rankops.weighted(
            bm25_results, dense_results, weight_a=0.6, weight_b=0.4, normalize=False
        )
        assert isinstance(result, list)

    def test_weighted_zero_weights_error(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test weighted fusion with zero weights raises error."""
        with pytest.raises(ValueError, match="weights cannot both be zero"):
            rankops.weighted(bm25_results, dense_results, weight_a=0.0, weight_b=0.0, normalize=True)

    def test_weighted_infinite_weight_error(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test weighted fusion with infinite weight raises error."""
        with pytest.raises(ValueError, match="weights must be finite"):
            rankops.weighted(
                bm25_results, dense_results, weight_a=float("inf"), weight_b=0.5, normalize=True
            )


# Standardized Tests
class TestStandardized:
    """Tests for standardized fusion."""

    def test_standardized_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic standardized fusion."""
        result = rankops.standardized(bm25_results, dense_results)
        assert isinstance(result, list)

    def test_standardized_custom_clip(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test standardized fusion with custom clip range."""
        result = rankops.standardized(
            bm25_results, dense_results, clip_range=(-2.0, 2.0)
        )
        assert isinstance(result, list)

    def test_standardized_multi(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test standardized fusion for multiple lists."""
        result = rankops.standardized_multi(multi_lists)
        assert isinstance(result, list)


# Additive Multi-Task Tests
class TestAdditiveMultiTask:
    """Tests for additive multi-task fusion."""

    def test_additive_multi_task_basic(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test basic additive multi-task fusion."""
        result = rankops.additive_multi_task(
            bm25_results, dense_results, weights=(1.0, 1.0), normalization="minmax"
        )
        assert isinstance(result, list)

    def test_additive_multi_task_normalizations(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test additive multi-task with different normalizations."""
        for norm in ["zscore", "minmax", "sum", "rank", "none"]:
            result = rankops.additive_multi_task(
                bm25_results, dense_results, weights=(1.0, 1.0), normalization=norm
            )
            assert isinstance(result, list)

    def test_additive_multi_task_invalid_normalization(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ):
        """Test additive multi-task with invalid normalization raises error."""
        with pytest.raises(ValueError, match="normalization must be one of"):
            rankops.additive_multi_task(
                bm25_results, dense_results, weights=(1.0, 1.0), normalization="invalid"
            )


# Configuration Classes Tests
class TestConfigClasses:
    """Tests for configuration classes."""

    def test_rrf_config(self):
        """Test RrfConfigPy class."""
        config = rankops.RrfConfigPy(k=100)
        assert config.k == 100
        config_with_top_k = config.with_top_k(5)
        assert config_with_top_k.top_k == 5

    def test_fusion_config(self):
        """Test FusionConfigPy class."""
        config = rankops.FusionConfigPy()
        config_with_top_k = config.with_top_k(10)
        assert config_with_top_k.top_k == 10

    def test_weighted_config(self):
        """Test WeightedConfigPy class."""
        config = rankops.WeightedConfigPy(weight_a=0.7, weight_b=0.3)
        assert config.weight_a == 0.7
        assert config.weight_b == 0.3
        config_with_norm = config.with_normalize(True)
        assert config_with_norm.normalize is True

    def test_standardized_config(self):
        """Test StandardizedConfigPy class."""
        config = rankops.StandardizedConfigPy(clip_range=(-2.0, 2.0))
        assert config.clip_range == (-2.0, 2.0)

    def test_additive_multi_task_config(self):
        """Test AdditiveMultiTaskConfigPy class."""
        config = rankops.AdditiveMultiTaskConfigPy(weights=(1.0, 1.0))
        assert config.weights == (1.0, 1.0)
        config_with_norm = config.with_normalization("zscore")
        assert config_with_norm.normalization == "zscore"


# Explainability Tests
class TestExplainability:
    """Tests for explainability functions."""

    def test_rrf_explain(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test RRF explainability."""
        retriever_ids = [
            rankops.RetrieverIdPy("BM25"),
            rankops.RetrieverIdPy("Dense"),
            rankops.RetrieverIdPy("Keyword"),
        ]
        result = rankops.rrf_explain(multi_lists, retriever_ids, k=60)
        assert isinstance(result, list)
        if len(result) > 0:
            item = result[0]
            assert hasattr(item, "id")
            assert hasattr(item, "score")
            assert hasattr(item, "rank")
            assert hasattr(item, "explanation")

    def test_combsum_explain(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test CombSUM explainability."""
        retriever_ids = [
            rankops.RetrieverIdPy("BM25"),
            rankops.RetrieverIdPy("Dense"),
            rankops.RetrieverIdPy("Keyword"),
        ]
        result = rankops.combsum_explain(multi_lists, retriever_ids)
        assert isinstance(result, list)

    def test_combmnz_explain(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test CombMNZ explainability."""
        retriever_ids = [
            rankops.RetrieverIdPy("BM25"),
            rankops.RetrieverIdPy("Dense"),
            rankops.RetrieverIdPy("Keyword"),
        ]
        result = rankops.combmnz_explain(multi_lists, retriever_ids)
        assert isinstance(result, list)

    def test_dbsf_explain(
        self, multi_lists: List[List[Tuple[str, float]]]
    ):
        """Test DBSF explainability."""
        retriever_ids = [
            rankops.RetrieverIdPy("BM25"),
            rankops.RetrieverIdPy("Dense"),
            rankops.RetrieverIdPy("Keyword"),
        ]
        result = rankops.dbsf_explain(multi_lists, retriever_ids)
        assert isinstance(result, list)


# Edge Cases and Error Handling
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_input_type(self):
        """Test that invalid input types raise appropriate errors."""
        with pytest.raises((TypeError, ValueError)):
            rankops.rrf("not a list", [("doc", 1.0)])

    def test_malformed_tuples(self):
        """Test that malformed tuples raise errors."""
        with pytest.raises((TypeError, ValueError)):
            rankops.rrf([("doc",)], [("doc", 1.0)])

    def test_non_string_ids(self):
        """Test that non-string IDs are handled."""
        # Should work with numeric IDs converted to strings
        result = rankops.rrf([(1, 1.0), (2, 0.9)], [(2, 0.8), (3, 0.7)])
        assert isinstance(result, list)

    def test_negative_scores(self):
        """Test that negative scores are handled."""
        result = rankops.rrf([("doc", -1.0)], [("doc", -0.5)])
        assert isinstance(result, list)

    def test_very_large_k(self):
        """Test RRF with very large k value."""
        bm25 = [("doc_A", 12.5), ("doc_B", 11.0)]
        dense = [("doc_B", 0.9), ("doc_A", 0.8)]
        result = rankops.rrf(bm25, dense, k=10000)
        assert isinstance(result, list)

    def test_single_item_lists(self):
        """Test fusion with single-item lists."""
        result = rankops.rrf([("doc_A", 1.0)], [("doc_B", 0.9)])
        assert isinstance(result, list)
        assert len(result) >= 1

