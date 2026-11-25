"""Tests for all 15 benchmark dataset and scorer registrations.

These tests verify:
1. Each dataset class can be instantiated
2. Each dataset has correct dataset_id and dataset_name
3. Each scorer class can be constructed (with mock backend)
4. The CLI _build_dataset and _build_scorer factories work for all benchmarks
5. KNOWN_BENCHMARKS in config.py includes all 15 benchmarks
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Dataset instantiation tests
# ---------------------------------------------------------------------------


class TestDatasetInstantiation:
    """Verify each dataset class can be instantiated with correct attributes."""

    def test_supergpqa(self) -> None:
        from openjarvis.evals.datasets.supergpqa import SuperGPQADataset

        ds = SuperGPQADataset()
        assert ds.dataset_id == "supergpqa"
        assert ds.dataset_name == "SuperGPQA"

    def test_gpqa(self) -> None:
        from openjarvis.evals.datasets.gpqa import GPQADataset

        ds = GPQADataset()
        assert ds.dataset_id == "gpqa"
        assert ds.dataset_name == "GPQA"

    def test_mmlu_pro(self) -> None:
        from openjarvis.evals.datasets.mmlu_pro import MMLUProDataset

        ds = MMLUProDataset()
        assert ds.dataset_id == "mmlu-pro"
        assert ds.dataset_name == "MMLU-Pro"

    def test_math500(self) -> None:
        from openjarvis.evals.datasets.math500 import MATH500Dataset

        ds = MATH500Dataset()
        assert ds.dataset_id == "math500"
        assert ds.dataset_name == "MATH-500"

    def test_natural_reasoning(self) -> None:
        from openjarvis.evals.datasets.natural_reasoning import NaturalReasoningDataset

        ds = NaturalReasoningDataset()
        assert ds.dataset_id == "natural-reasoning"
        assert ds.dataset_name == "Natural Reasoning"

    def test_hle(self) -> None:
        from openjarvis.evals.datasets.hle import HLEDataset

        ds = HLEDataset()
        assert ds.dataset_id == "hle"
        assert ds.dataset_name == "HLE"

    def test_simpleqa(self) -> None:
        from openjarvis.evals.datasets.simpleqa import SimpleQADataset

        ds = SimpleQADataset()
        assert ds.dataset_id == "simpleqa"
        assert ds.dataset_name == "SimpleQA"

    def test_wildchat(self) -> None:
        from openjarvis.evals.datasets.wildchat import WildChatDataset

        ds = WildChatDataset()
        assert ds.dataset_id == "wildchat"
        assert ds.dataset_name == "WildChat"

    def test_ipw(self) -> None:
        from openjarvis.evals.datasets.ipw_mixed import IPWDataset

        ds = IPWDataset()
        assert ds.dataset_id == "ipw"
        assert ds.dataset_name == "IPW"

    def test_gaia(self) -> None:
        from openjarvis.evals.datasets.gaia import GAIADataset

        ds = GAIADataset()
        assert ds.dataset_id == "gaia"
        assert ds.dataset_name == "GAIA"

    def test_frames(self) -> None:
        from openjarvis.evals.datasets.frames import FRAMESDataset

        ds = FRAMESDataset()
        assert ds.dataset_id == "frames"
        assert ds.dataset_name == "FRAMES"

    def test_swebench(self) -> None:
        from openjarvis.evals.datasets.swebench import SWEBenchDataset

        ds = SWEBenchDataset()
        assert ds.dataset_id == "swebench"
        assert ds.dataset_name == "SWE-bench"

    def test_swefficiency(self) -> None:
        from openjarvis.evals.datasets.swefficiency import SWEfficiencyDataset

        ds = SWEfficiencyDataset()
        assert ds.dataset_id == "swefficiency"
        assert ds.dataset_name == "SWEfficiency"

    def test_terminalbench(self) -> None:
        from openjarvis.evals.datasets.terminalbench import TerminalBenchDataset

        ds = TerminalBenchDataset()
        assert ds.dataset_id == "terminalbench"
        assert ds.dataset_name == "TerminalBench"

    def test_terminalbench_native(self) -> None:
        from openjarvis.evals.datasets.terminalbench_native import (
            TerminalBenchNativeDataset,
        )

        ds = TerminalBenchNativeDataset()
        assert ds.dataset_id == "terminalbench-native"
        assert ds.dataset_name == "TerminalBench Native"

    def test_terminalbench_v2_1(self) -> None:
        from openjarvis.evals.datasets.terminalbench_v2_1 import (
            TerminalBenchV21Dataset,
        )

        ds = TerminalBenchV21Dataset()
        assert ds.dataset_id == "terminalbench-v2.1"
        assert ds.dataset_name == "TerminalBench V2.1"

    def test_livecodebench(self) -> None:
        from openjarvis.evals.datasets.livecodebench import LiveCodeBenchDataset

        ds = LiveCodeBenchDataset()
        assert ds.dataset_id == "livecodebench"
        assert ds.dataset_name == "LiveCodeBench"

    def test_deepresearch(self) -> None:
        from openjarvis.evals.datasets.liveresearch import LiveResearchBenchDataset

        ds = LiveResearchBenchDataset()
        assert ds.dataset_id == "liveresearch"
        assert ds.dataset_name == "DeepResearchBench"

    def test_toolcall15(self) -> None:
        from openjarvis.evals.datasets.toolcall15 import ToolCall15Dataset

        ds = ToolCall15Dataset()
        assert ds.dataset_id == "toolcall15"
        assert ds.dataset_name == "ToolCall-15"


# ---------------------------------------------------------------------------
# Scorer instantiation tests
# ---------------------------------------------------------------------------


def _mock_backend() -> MagicMock:
    """Create a mock inference backend for scorer construction."""
    backend = MagicMock()
    backend.generate.return_value = "A"
    return backend

class TestIPWDatasetLoading:

    def test_arrow_load_uses_indexed_access(self, monkeypatch, tmp_path) -> None:
        import random
        import datasets

        from openjarvis.evals.datasets import ipw_mixed

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Arrow dataset must not call to_list()")

        rows = [
            {
                "problem": f"Problem {i}",
                "answer": f"Answer {i}",
                "subject": "math",
            }
            for i in range(5)
        ]

        dataset = FakeDataset(rows)

        dataset_path = tmp_path / "mixed_1k_seed1_base"
        dataset_path.mkdir()

        monkeypatch.setattr(
            datasets,
            "load_from_disk",
            lambda path: dataset,
        )

        ds = ipw_mixed.IPWDataset(data_dir=str(tmp_path))

        expected_indices = list(range(len(rows)))
        random.Random(42).shuffle(expected_indices)
        expected_indices = expected_indices[:2]

        ds.load(seed=42, max_samples=2)

        assert dataset.accessed == expected_indices
        assert ds.size() == 2

        records = list(ds.iter_records())
        assert [record.record_id for record in records] == [
            "ipw-0",
            "ipw-1",
        ]


class TestMMLUProLoading:

    def test_load_uses_indexed_access(self, monkeypatch) -> None:
        import random

        from openjarvis.evals.datasets import mmlu_pro

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("load() must not call to_list()")

        rows = [
            {
                "question": f"Question {i}",
                "options": ["A", "B", "C"],
                "answer": "A",
                "category": "math",
            }
            for i in range(5)
        ]

        dataset = FakeDataset(rows)

        import datasets

        monkeypatch.setattr(
            datasets,
            "load_dataset",
            lambda *args, **kwargs: dataset,
        )

        ds = mmlu_pro.MMLUProDataset()

        expected_indices = list(range(len(rows)))
        random.Random(42).shuffle(expected_indices)
        expected_indices = expected_indices[:2]

        ds.load(seed=42, max_samples=2)

        assert dataset.accessed == expected_indices
        assert ds.size() == 2

        records = list(ds.iter_records())
        assert [r.record_id for r in records] == [
            "mmlu-pro-0",
            "mmlu-pro-1",
        ]

class TestScorerInstantiation:
    """Verify each scorer class can be constructed."""

    def test_supergpqa_scorer(self) -> None:
        from openjarvis.evals.scorers.supergpqa_mcq import SuperGPQAScorer

        s = SuperGPQAScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "supergpqa"

    def test_gpqa_scorer(self) -> None:
        from openjarvis.evals.scorers.gpqa_mcq import GPQAScorer

        s = GPQAScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "gpqa"

    def test_mmlu_pro_scorer(self) -> None:
        from openjarvis.evals.scorers.mmlu_pro_mcq import MMLUProScorer

        s = MMLUProScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "mmlu-pro"

    def test_reasoning_judge_scorer(self) -> None:
        from openjarvis.evals.scorers.reasoning_judge import ReasoningJudgeScorer

        s = ReasoningJudgeScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "reasoning_judge"

    def test_hle_scorer(self) -> None:
        from openjarvis.evals.scorers.hle_judge import HLEScorer

        s = HLEScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "hle"

    def test_simpleqa_scorer(self) -> None:
        from openjarvis.evals.scorers.simpleqa_judge import SimpleQAScorer

        s = SimpleQAScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "simpleqa"

    def test_wildchat_scorer(self) -> None:
        from openjarvis.evals.scorers.wildchat_judge import WildChatScorer

        s = WildChatScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "wildchat"

    def test_ipw_mixed_scorer(self) -> None:
        from openjarvis.evals.scorers.ipw_mixed import IPWMixedScorer

        s = IPWMixedScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "ipw"

    def test_gaia_scorer(self) -> None:
        from openjarvis.evals.scorers.gaia_exact import GAIAScorer

        s = GAIAScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "gaia"

    def test_frames_scorer(self) -> None:
        from openjarvis.evals.scorers.frames_judge import FRAMESScorer

        s = FRAMESScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "frames"

    def test_swebench_scorer(self) -> None:
        from openjarvis.evals.scorers.swebench_structural import SWEBenchScorer

        s = SWEBenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "swebench"

    def test_swefficiency_scorer(self) -> None:
        from openjarvis.evals.scorers.swefficiency_structural import (
            SWEfficiencyScorer,
        )

        s = SWEfficiencyScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "swefficiency"

    def test_terminalbench_scorer(self) -> None:
        from openjarvis.evals.scorers.terminalbench_judge import TerminalBenchScorer

        s = TerminalBenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "terminalbench"

    def test_terminalbench_native_scorer(self) -> None:
        from openjarvis.evals.scorers.terminalbench_native_structural import (
            TerminalBenchNativeScorer,
        )

        s = TerminalBenchNativeScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "terminalbench-native"

    def test_terminalbench_v2_1_scorer(self) -> None:
        from openjarvis.evals.scorers.terminalbench_v2_1 import (
            TerminalBenchV21Scorer,
        )

        s = TerminalBenchV21Scorer(_mock_backend(), "test-model")
        assert s.scorer_id == "terminalbench-v2.1"

    def test_livecodebench_scorer(self) -> None:
        from openjarvis.evals.scorers.livecodebench import LiveCodeBenchScorer

        s = LiveCodeBenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "livecodebench"

    def test_deepresearch_scorer(self) -> None:
        from openjarvis.evals.scorers.liveresearch import LiveResearchBenchScorer

        s = LiveResearchBenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "liveresearch"

    def test_toolcall15_scorer(self) -> None:
        from openjarvis.evals.scorers.toolcall15 import ToolCall15Scorer

        s = ToolCall15Scorer(_mock_backend(), "test-model")
        assert s.scorer_id == "toolcall15"


# ---------------------------------------------------------------------------
# CLI factory tests
# ---------------------------------------------------------------------------


ALL_BENCHMARKS = [
    "supergpqa",
    "gpqa",
    "mmlu-pro",
    "math500",
    "natural-reasoning",
    "hle",
    "simpleqa",
    "wildchat",
    "ipw",
    "gaia",
    "frames",
    "swebench",
    "swefficiency",
    "terminalbench",
    "terminalbench-native",
    "terminalbench-v2.1",
    "livecodebench",
    "liveresearch",
    "toolcall15",
]


class TestCLIFactories:
    """Verify CLI _build_dataset and _build_scorer work for all benchmarks."""

    @pytest.mark.parametrize("benchmark", ALL_BENCHMARKS)
    def test_build_dataset(self, benchmark: str) -> None:
        from openjarvis.evals.cli import _build_dataset

        ds = _build_dataset(benchmark)
        assert ds is not None
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")

    @pytest.mark.parametrize("benchmark", ALL_BENCHMARKS)
    def test_build_scorer(self, benchmark: str) -> None:
        from openjarvis.evals.cli import _build_scorer

        scorer = _build_scorer(benchmark, _mock_backend(), "test-model")
        assert scorer is not None
        assert hasattr(scorer, "score")

    def test_build_dataset_unknown(self) -> None:
        import click

        from openjarvis.evals.cli import _build_dataset

        with pytest.raises(click.ClickException, match="Unknown benchmark"):
            _build_dataset("nonexistent")

    def test_build_scorer_unknown(self) -> None:
        import click

        from openjarvis.evals.cli import _build_scorer

        with pytest.raises(click.ClickException, match="Unknown benchmark"):
            _build_scorer("nonexistent", _mock_backend(), "test-model")


# ---------------------------------------------------------------------------
# Config KNOWN_BENCHMARKS test
# ---------------------------------------------------------------------------


class TestConfigBenchmarks:
    """Verify KNOWN_BENCHMARKS includes all 15 benchmarks."""

    def test_all_benchmarks_known(self) -> None:
        from openjarvis.evals.core.config import KNOWN_BENCHMARKS

        for b in ALL_BENCHMARKS:
            assert b in KNOWN_BENCHMARKS, f"{b} missing from KNOWN_BENCHMARKS"

    def test_benchmarks_count(self) -> None:
        from openjarvis.evals.core.config import KNOWN_BENCHMARKS

        assert len(KNOWN_BENCHMARKS) == 32


# ---------------------------------------------------------------------------
# Structural scorer tests
# ---------------------------------------------------------------------------


class TestStructuralScorers:
    """Test structural scorers that don't need LLM calls."""

    def test_swebench_empty_response(self) -> None:
        from openjarvis.evals.core.types import EvalRecord
        from openjarvis.evals.scorers.swebench_structural import SWEBenchScorer

        scorer = SWEBenchScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="swe-1",
            problem="Fix bug",
            reference="patch",
            category="agentic",
        )
        is_correct, meta = scorer.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"

    def test_swebench_with_diff(self) -> None:
        from openjarvis.evals.core.types import EvalRecord
        from openjarvis.evals.scorers.swebench_structural import SWEBenchScorer

        scorer = SWEBenchScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="swe-2",
            problem="Fix bug",
            reference="patch",
            category="agentic",
        )
        answer = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        is_correct, meta = scorer.score(record, answer)
        assert is_correct is None  # indeterminate
        assert meta["reason"] == "requires_test_execution"
        assert meta["has_diff_markers"] is True

    def test_terminalbench_native_no_results(self) -> None:
        from openjarvis.evals.core.types import EvalRecord
        from openjarvis.evals.scorers.terminalbench_native_structural import (
            TerminalBenchNativeScorer,
        )

        scorer = TerminalBenchNativeScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="tb-1",
            problem="Run command",
            reference="",
            category="agentic",
        )
        is_correct, meta = scorer.score(record, "some output")
        assert is_correct is None
        assert meta["reason"] == "no_test_results"

    def test_terminalbench_native_resolved(self) -> None:
        from openjarvis.evals.core.types import EvalRecord
        from openjarvis.evals.scorers.terminalbench_native_structural import (
            TerminalBenchNativeScorer,
        )

        scorer = TerminalBenchNativeScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="tb-2",
            problem="Run command",
            reference="",
            category="agentic",
            metadata={"is_resolved": True},
        )
        is_correct, meta = scorer.score(record, "output")
        assert is_correct is True


class TestTerminalBenchDatasetLoading:
    def test_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.terminalbench import TerminalBenchDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized with to_list()")

        rows = [
            {"prompt": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(10)
        ]
        fake_dataset = FakeDataset(rows)

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:3]

        ds = TerminalBenchDataset()
        ds.load(max_samples=3, seed=seed)

        records = list(ds.iter_records())

        assert ds.size() == 3
        assert fake_dataset.accessed_indices == expected_indices
        assert [r.problem for r in records] == [
            f"Question {i}" for i in expected_indices
        ]
        assert [r.record_id for r in records] == [
            f"terminalbench-tb_{i}" for i in expected_indices
        ]

    def test_simpleqa_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.simpleqa import SimpleQADataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized with to_list()")

        rows = [
            {"problem": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(10)
        ]
        fake_dataset = FakeDataset(rows)

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:3]

        ds = SimpleQADataset()
        ds.load(max_samples=3, seed=seed)

        records = list(ds.iter_records())

        assert ds.size() == 3
        assert fake_dataset.accessed_indices == expected_indices
        assert [r.record_id for r in records] == [
            f"simpleqa-{i}" for i in expected_indices
        ]

    def test_math500_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.math500 import MATH500Dataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "problem": f"Problem {i}",
                "solution": f"Solution {i}",
                "subject": "Algebra",
                "level": "Level 1",
            }
            for i in range(10)
        ]

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:3]

        ds = MATH500Dataset()
        ds.load(max_samples=3, seed=seed)
        records = list(ds.iter_records())

        assert ds.size() == 3
        assert fake_dataset.accessed_indices == expected_indices
        assert [r.record_id for r in records] == [
            f"math500-{i}" for i in expected_indices
        ]

    def test_swebench_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.swebench import SWEBenchDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "instance_id": f"test-{i}",
                "repo": "owner/repo",
                "problem_statement": f"Fix issue {i}",
                "patch": f"patch-{i}",
                "base_commit": "abc123",
                "hints_text": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }
            for i in range(10)
        ]

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:3]

        ds = SWEBenchDataset()
        ds.load(max_samples=3, seed=seed)
        records = list(ds.iter_records())

        assert ds.size() == 3
        assert fake_dataset.accessed_indices == expected_indices
        assert [r.record_id for r in records] == [
            f"swebench-test-{i}" for i in expected_indices
        ]

    def test_swefficiency_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.swefficiency import SWEfficiencyDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "instance_id": f"test-{i}",
                "repo": "owner/repo",
                "problem_statement": f"Optimize function {i}",
                "workload": f"workload-{i}",
                "patch": f"patch-{i}",
                "speedup": 1.5,
            }
            for i in range(10)
        ]

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:3]

        ds = SWEfficiencyDataset()
        ds.load(max_samples=3, seed=seed)

        records = list(ds.iter_records())

        assert ds.size() == 3
        assert fake_dataset.accessed_indices == expected_indices
        assert [r.record_id for r in records] == [
            f"swefficiency-test-{i}" for i in expected_indices
        ]

    def test_gpqa_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.gpqa import GPQADataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "Question": f"Question {i}",
                "Correct Answer": f"Correct {i}",
                "Incorrect Answer 1": f"Wrong A {i}",
                "Incorrect Answer 2": f"Wrong B {i}",
                "Incorrect Answer 3": f"Wrong C {i}",
                "Subdomain": "Algebra",
            }
            for i in range(10)
        ]

        # Make one selected row invalid to verify filtering and original IDs.
        rows[1]["Question"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:5]

        ds = GPQADataset()
        ds.load(max_samples=5, seed=seed)

        records = list(ds.iter_records())
        valid_indices = [
            idx for idx in expected_indices if idx != 1
        ]

        assert fake_dataset.accessed_indices == expected_indices
        assert ds.size() == len(valid_indices)
        assert [r.record_id for r in records] == [
            f"gpqa-{i}" for i in valid_indices
        ]

    def test_supergpqa_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.supergpqa import SuperGPQADataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "question": f"Question {i}",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer_letter": "A",
                "answer": "Option A",
                "field": "Science",
            }
            for i in range(10)
        ]

        # Verify invalid rows are filtered without changing original indices.
        rows[1]["question"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:5]

        ds = SuperGPQADataset()
        ds.load(max_samples=5, seed=seed)

        records = list(ds.iter_records())
        valid_indices = [idx for idx in expected_indices if idx != 1]

        assert fake_dataset.accessed_indices == expected_indices
        assert ds.size() == len(valid_indices)
        assert [r.record_id for r in records] == [
            f"supergpqa-{i}" for i in valid_indices
        ]

    def test_wildchat_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.wildchat import WildChatDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = []
        for i in range(10):
            rows.append(
                {
                    "language": "English",
                    "conversation": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}"},
                    ],
                    "model": "test-model",
                }
            )

        # Invalid rows must be excluded before shuffle and sampling.
        rows[1]["language"] = "French"
        rows[3]["conversation"] = [
            {"role": "user", "content": "Multi-turn"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        rows[5]["conversation"][0]["content"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        valid_indices = [0, 2, 4, 6, 7, 8, 9]
        expected_indices = list(valid_indices)
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:4]

        ds = WildChatDataset()
        ds.load(max_samples=4, seed=seed)

        records = list(ds.iter_records())

        # First pass scans all rows; second pass retrieves selected rows.
        assert fake_dataset.accessed_indices == (
            list(range(len(rows))) + expected_indices
        )
        assert ds.size() == 4
        assert [r.record_id for r in records] == [
            f"wildchat-{i}" for i in range(4)
        ]
        assert [r.problem for r in records] == [
            f"Question {i}" for i in expected_indices
        ]
        assert [r.reference for r in records] == [
            f"Answer {i}" for i in expected_indices
        ]


    def test_frames_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.frames import FRAMESDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "Prompt": f"Question {i}",
                "Answer": f"Answer {i}",
                "reasoning_types": ["multi-hop"],
                "wiki_links": [f"Article {i}"],
            }
            for i in range(10)
        ]

        # Invalid row must be filtered without changing original indices.
        rows[1]["Prompt"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:5]

        ds = FRAMESDataset()
        ds.load(max_samples=5, seed=seed)

        records = list(ds.iter_records())
        valid_indices = [idx for idx in expected_indices if idx != 1]

        assert fake_dataset.accessed_indices == expected_indices
        assert ds.size() == len(valid_indices)

        assert [r.record_id for r in records] == [
            f"frames-{i}" for i in valid_indices
        ]
        assert [r.metadata["index"] for r in records] == valid_indices
        assert [r.reference for r in records] == [
            f"Answer {i}" for i in valid_indices
        ]


    def test_generalthoughts_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.generalthoughts import GeneralThoughtsDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "question": f"Question {i}",
                "model_answer": f"Answer {i}",
                "question_id": None,
                "task": "reasoning",
            }
            for i in range(20)
        ]

        # Invalid question is filtered after split and sampling.
        rows[1]["question"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        shuffled_indices = list(range(len(rows)))
        random.Random(seed).shuffle(shuffled_indices)
        expected_indices = shuffled_indices[:4]  # 20% train split

        ds = GeneralThoughtsDataset()
        ds.load(split="train", seed=seed)

        records = list(ds.iter_records())
        valid_indices = [i for i in expected_indices if i != 1]

        assert fake_dataset.accessed_indices == expected_indices
        assert ds.size() == len(valid_indices)
        assert [r.record_id for r in records] == [
            f"generalthoughts-{i}" for i in valid_indices
        ]
        assert [r.problem for r in records] == [
            f"Question {i}" for i in valid_indices
        ]
        assert [r.reference for r in records] == [
            f"Answer {i}" for i in valid_indices
        ]


    def test_natural_reasoning_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.natural_reasoning import (
            NaturalReasoningDataset,
        )

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "category": "science",
            }
            for i in range(10)
        ]
        rows[2]["question"] = ""
        rows[6]["answer"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        expected_indices = list(range(len(rows)))
        random.Random(seed).shuffle(expected_indices)
        expected_indices = expected_indices[:6]

        ds = NaturalReasoningDataset()
        ds.load(max_samples=6, seed=seed)

        records = list(ds.iter_records())
        valid_indices = [
            i for i in expected_indices if i not in (2, 6)
        ]

        assert fake_dataset.accessed_indices == expected_indices
        assert ds.size() == len(valid_indices)
        assert [r.record_id for r in records] == [
            f"natural-reasoning-{i}" for i in valid_indices
        ]
        assert [r.reference for r in records] == [
            f"Answer {i}" for i in valid_indices
        ]


    def test_hle_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.hle import HLEDataset

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError("Dataset must not be materialized")

        rows = [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "id": f"task-{i}",
            }
            for i in range(10)
        ]

        # Invalid and multimodal rows must be skipped.
        rows[1]["question"] = ""
        rows[3]["image"] = "image-data"
        rows[5]["answer"] = ""

        fake_dataset = FakeDataset(rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        shuffled_indices = list(range(len(rows)))
        random.Random(seed).shuffle(shuffled_indices)

        expected_indices = []
        for idx in shuffled_indices:
            if idx in (1, 3, 5):
                continue
            expected_indices.append(idx)
            if len(expected_indices) == 4:
                break

        ds = HLEDataset(text_only=True)
        ds.load(max_samples=4, seed=seed)

        records = list(ds.iter_records())

        assert fake_dataset.accessed_indices == shuffled_indices[
            : shuffled_indices.index(expected_indices[-1]) + 1
        ]
        assert ds.size() == 4
        assert [r.record_id for r in records] == [
            f"hle-{i}" for i in expected_indices
        ]
        assert [r.reference for r in records] == [
            f"Answer {i}" for i in expected_indices
        ]


class TestToolOrchestraDatasetLoading:
    def test_load_uses_indexed_access_without_materializing_dataset(
        self, monkeypatch
    ) -> None:
        import random
        import sys
        import types

        from openjarvis.evals.datasets.toolorchestra import (
            ToolOrchestraDataset,
        )

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.accessed_indices = []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.accessed_indices.append(index)
                return self.rows[index]

            def to_list(self):
                raise AssertionError(
                    "Dataset must not be materialized with to_list()"
                )

        rows = [
            {
                "id": f"tool-{i}",
                "user_scenario": {
                    "instructions": {
                        "task_instructions": f"Task {i}",
                    }
                },
                "evaluation_criteria": {
                    "actions": [f"action-{i}"],
                },
                "description": {"purpose": f"Purpose {i}"},
            }
            for i in range(10)
        ]

        fake_dataset = FakeDataset(rows)

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *args, **kwargs: fake_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        seed = 42
        shuffled_indices = list(range(len(rows)))
        random.Random(seed).shuffle(shuffled_indices)

        # ToolOrchestra uses the first 20% for train and the remainder
        # for test. max_samples is applied after splitting.
        expected_indices = shuffled_indices[2:5]

        ds = ToolOrchestraDataset()
        ds.load(split="test", seed=seed, max_samples=3)

        records = list(ds.iter_records())

        assert fake_dataset.accessed_indices == expected_indices
        assert ds.size() == 3

        assert [r.record_id for r in records] == [
            f"tool-{i}" for i in expected_indices
        ]

        assert [r.problem for r in records] == [
            f"Task {i}" for i in expected_indices
        ]

        assert [r.reference for r in records] == [
            str([f"action-{i}"]) for i in expected_indices
        ]

        assert all(
            r.metadata["source"] == "toolorchestra"
            for r in records
        )


