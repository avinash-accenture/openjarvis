
from typing import Any

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.datasets.livecodebench import (
    LiveCodeBenchDataset,
    _LazyTestCaseSource,
    _extract_prompt_examples,
)
from openjarvis.evals.scorers import livecodebench as scorer_module
from openjarvis.evals.scorers.livecodebench import LiveCodeBenchScorer


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.accesses: list[int] = []

    @property
    def column_names(self) -> list[str]:
        return list(self.rows[0].keys()) if self.rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accesses.append(index)
        return self.rows[index]

    def select_columns(self, columns: list[str]) -> "FakeDataset":
        selected_rows = [
            {key: row[key] for key in columns if key in row}
            for row in self.rows
        ]
        selected = FakeDataset(selected_rows)
        selected.accesses = self.accesses
        return selected

def _record(metadata: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        record_id="test-1",
        problem="Solve this problem",
        reference="",
        category="coding",
        metadata=metadata,
    )


def test_lazy_source_loads_all_structured_cases_without_caching() -> None:
    dataset = FakeDataset(
        [
            {
                "public_test_cases": [
                    {"input": "1", "output": "2"},
                    {"input": "3", "output": "4"},
                ],
                "hidden_test_cases": [
                    {"input": "5", "output": "6"},
                ],
            }
        ]
    )

    source = _LazyTestCaseSource(dataset, 0)

    inputs, outputs = source.load()

    assert inputs == ["1", "3", "5"]
    assert outputs == ["2", "4", "6"]
    assert dataset.accesses == [0]

    # A second call retrieves the row again rather than retaining arrays.
    inputs_again, outputs_again = source.load()

    assert inputs_again == inputs
    assert outputs_again == outputs
    assert dataset.accesses == [0, 0]

def test_lazy_source_includes_private_test_cases() -> None:
    dataset = FakeDataset(
        [
            {
                "public_test_cases": [
                    {"input": "1 2", "output": "3"},
                ],
                "private_test_cases": (
                    '[{"input": "5 6", "output": "11", "testtype": "stdin"},'
                    '{"input": "7 8", "output": "15", "testtype": "stdin"}]'
                ),
            }
        ]
    )

    inputs, outputs = _LazyTestCaseSource(dataset, 0).load()

    assert inputs == ["1 2", "5 6", "7 8"]
    assert outputs == ["3", "11", "15"]

def test_lazy_source_supports_json_encoded_cases() -> None:
    dataset = FakeDataset(
        [
            {
                "public_test_cases": [
                    '{"input": "hello", "output": "world"}',
                ]
            }
        ]
    )

    inputs, outputs = _LazyTestCaseSource(dataset, 0).load()

    assert inputs == ["hello"]
    assert outputs == ["world"]


def test_lazy_source_supports_flat_test_case_fields() -> None:
    dataset = FakeDataset(
        [
            {
                "input": '["1", "2"]',
                "output": '["3", "4"]',
            }
        ]
    )

    inputs, outputs = _LazyTestCaseSource(dataset, 0).load()

    assert inputs == ["1", "2"]
    assert outputs == ["3", "4"]


def test_dataset_records_keep_lazy_source_not_test_arrays(
    monkeypatch: Any,
) -> None:
    import datasets

    fake_dataset = FakeDataset(
        [
            {
                "question_id": "problem-1",
                "question_content": "Add two numbers",
                "public_test_cases": [
                    {"input": "1 2", "output": "3"},
                    {"input": "4 5", "output": "9"},
                ],
                "hidden_test_cases": [
                    {"input": "10 20", "output": "30"},
                ],
            }
        ]
    )

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    provider = LiveCodeBenchDataset()
    provider.load(split="custom", max_samples=1)

    assert provider.size() == 1
    record = next(provider.iter_records())

    assert record.record_id == "problem-1"
    assert "Input 1:" in record.problem
    assert "test_inputs" not in record.metadata
    assert "test_outputs" not in record.metadata

    source = record.metadata["_test_case_source"]
    assert isinstance(source, _LazyTestCaseSource)

    # The row was read once to build the prompt.
    assert fake_dataset.accesses == [0]

    # Full test cases are available only when requested.
    inputs, outputs = source.load()

    assert inputs == ["1 2", "4 5", "10 20"]
    assert outputs == ["3", "9", "30"]


def test_scorer_uses_lazy_source(monkeypatch: Any) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run(
        code: str,
        test_input: str,
        expected_output: str,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        calls.append((test_input, expected_output))
        return True, "exact_match"

    monkeypatch.setattr(scorer_module, "_run_single_test", fake_run)

    dataset = FakeDataset(
        [
            {
                "public_test_cases": [
                    {"input": "1", "output": "2"},
                    {"input": "3", "output": "4"},
                ],
                "hidden_test_cases": [
                    {"input": "5", "output": "6"},
                ],
            }
        ]
    )

    source = _LazyTestCaseSource(dataset, 0)
    record = _record({"_test_case_source": source})

    result, metadata = LiveCodeBenchScorer().score(
        record,
        "print(input())",
    )

    assert result is True
    assert metadata["tests_passed"] == 3
    assert metadata["tests_total"] == 3
    assert calls == [("1", "2"), ("3", "4"), ("5", "6")]
    assert dataset.accesses == [0]


def test_scorer_preserves_ordinary_list_compatibility(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run(
        code: str,
        test_input: str,
        expected_output: str,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        calls.append((test_input, expected_output))
        return True, "exact_match"

    monkeypatch.setattr(scorer_module, "_run_single_test", fake_run)

    record = _record(
        {
            "test_inputs": ["1", "2"],
            "test_outputs": ["2", "3"],
        }
    )

    result, metadata = LiveCodeBenchScorer().score(
        record,
        "print(input())",
    )

    assert result is True
    assert metadata["tests_total"] == 2
    assert calls == [("1", "2"), ("2", "3")]


def test_scorer_preserves_mismatch_truncation(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run(
        code: str,
        test_input: str,
        expected_output: str,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        calls.append((test_input, expected_output))
        return True, "exact_match"

    monkeypatch.setattr(scorer_module, "_run_single_test", fake_run)

    record = _record(
        {
            "test_inputs": ["1", "2", "3"],
            "test_outputs": ["2", "3"],
        }
    )

    result, metadata = LiveCodeBenchScorer().score(
        record,
        "print(input())",
    )

    assert result is True
    assert metadata["tests_total"] == 2
    assert calls == [("1", "2"), ("2", "3")]

def test_prompt_examples_extract_only_first_two_structured_cases() -> None:
    raw = {
        "public_test_cases": [
            {"input": "1", "output": "2"},
            {"input": "3", "output": "4"},
            {"input": "5", "output": "6"},
        ],
        "hidden_test_cases": [
            {"input": "7", "output": "8"},
        ],
    }

    inputs, outputs = _extract_prompt_examples(raw)

    assert inputs == ["1", "3"]
    assert outputs == ["2", "4"]


def test_prompt_examples_support_json_encoded_cases() -> None:
    raw = {
        "public_test_cases": [
            '{"input": "hello", "output": "world"}',
            '{"input": "foo", "output": "bar"}',
        ]
    }

    inputs, outputs = _extract_prompt_examples(raw)

    assert inputs == ["hello", "foo"]
    assert outputs == ["world", "bar"]


def test_prompt_examples_support_flat_fields() -> None:
    raw = {
        "input": '["1", "2", "3"]',
        "output": '["4", "5", "6"]',
    }

    inputs, outputs = _extract_prompt_examples(raw)

    assert inputs == ["1", "2"]
    assert outputs == ["4", "5"]


def test_prompt_examples_preserve_independent_fallback() -> None:
    raw = {
        "public_test_cases": [
            {"input": "structured-input", "output": ""},
        ],
        "input": '["flat-input"]',
        "output": '["flat-output"]',
    }

    inputs, outputs = _extract_prompt_examples(raw)

    assert inputs == ["structured-input"]
    assert outputs == [""]

def test_scorer_uses_iterator_without_loading_all_cases(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, str]] = []

    class IteratorOnlySource:
        def iter_cases(self):
            yield ("1", "2")
            yield ("3", "4")
            yield ("5", "6")

        def load(self):
            raise AssertionError("Scorer must not call load()")

    def fake_run(
        code: str,
        test_input: str,
        expected_output: str,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        calls.append((test_input, expected_output))
        return True, "exact_match"

    monkeypatch.setattr(scorer_module, "_run_single_test", fake_run)

    record = _record({"_test_case_source": IteratorOnlySource()})

    result, metadata = LiveCodeBenchScorer().score(
        record,
        "print(input())",
    )

    assert result is True
    assert metadata["tests_total"] == 3
    assert metadata["tests_passed"] == 3
    assert calls == [("1", "2"), ("3", "4"), ("5", "6")]


def test_scorer_retains_only_first_five_test_details(
    monkeypatch: Any,
) -> None:
    def fake_run(
        code: str,
        test_input: str,
        expected_output: str,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        return test_input == expected_output, "exact_match"

    monkeypatch.setattr(scorer_module, "_run_single_test", fake_run)

    class TenCaseSource:
        def iter_cases(self):
            for i in range(10):
                value = str(i)
                yield value, value

        def load(self):
            raise AssertionError("Scorer must not call load()")

    record = _record({"_test_case_source": TenCaseSource()})

    result, metadata = LiveCodeBenchScorer().score(
        record,
        "print(input())",
    )

    assert result is True
    assert metadata["tests_total"] == 10
    assert metadata["tests_passed"] == 10
    assert metadata["pass_rate"] == 1.0
    assert len(metadata["test_details"]) == 5
    assert metadata["test_details_truncated"] is True
    assert [item["test_index"] for item in metadata["test_details"]] == [
        0, 1, 2, 3, 4
    ]

def test_arrow_backed_source_reads_private_cases_without_materializing_row() -> None:
    import pyarrow as pa

    from openjarvis.evals.datasets.livecodebench import (
        _LazyTestCaseSource,
    )

    table = pa.Table.from_pylist(
        [
            {
                "public_test_cases": (
                    '[{"input":"1","output":"2"}]'
                ),
                "private_test_cases": (
                    '[{"input":"3","output":"4"},'
                    '{"input":"5","output":"6"}]'
                ),
            }
        ]
    )

    class ArrowDataset:
        data = table
        column_names = table.column_names

        def __getitem__(self, index: int) -> Any:
            raise AssertionError(
                "Arrow-backed iteration must not materialize the dataset row"
            )

    source = _LazyTestCaseSource(ArrowDataset(), 0)

    assert list(source.iter_cases()) == [
        ("1", "2"),
        ("3", "4"),
        ("5", "6"),
    ]

def test_json_array_parser_handles_escaped_quotes_and_nested_values() -> None:
    import json

    from openjarvis.evals.datasets.livecodebench import _iter_json_array

    cases = [
        {
            "input": 'print("hello, \\"world\\"")',
            "output": '{"message": "hello, world"}',
            "metadata": {"nested": [1, 2, {"value": "} ] ,"}]},
        },
        {
            "input": "line1\nline2",
            "output": "line2",
        },
    ]

    encoded = json.dumps(cases).encode("utf-8")
    parsed = list(_iter_json_array(encoded, chunk_size=7))

    assert parsed == cases


def test_json_array_parser_handles_empty_array() -> None:
    from openjarvis.evals.datasets.livecodebench import _iter_json_array

    assert list(_iter_json_array(b"[]")) == []


def test_json_array_parser_handles_large_item_across_chunks() -> None:
    import json

    from openjarvis.evals.datasets.livecodebench import _iter_json_array

    cases = [
        {"input": "x" * 200_000, "output": "y" * 100_000}
    ]

    encoded = json.dumps(cases).encode("utf-8")

    parsed = list(_iter_json_array(encoded, chunk_size=1024))

    assert parsed == cases


def test_dataset_split_selection_is_deterministic(
    monkeypatch: Any,
) -> None:
    import random

    import datasets

    rows = [
        {
            "question_id": f"problem-{i}",
            "question_content": f"Problem {i}",
        }
        for i in range(10)
    ]

    fake_dataset = FakeDataset(rows)

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    seed = 7
    shuffled_indices = list(range(len(rows)))
    random.Random(seed).shuffle(shuffled_indices)

    cut = int(len(shuffled_indices) * 0.2)

    expected_indices = {
        "train": shuffled_indices[:cut],
        "test": shuffled_indices[cut:],
        "all": shuffled_indices,
    }

    for split, indices in expected_indices.items():
        provider = LiveCodeBenchDataset()
        provider.load(split=split, seed=seed)

        actual_ids = [
            record.record_id
            for record in provider.iter_records()
        ]

        expected_ids = [
            f"problem-{i}"
            for i in indices
        ]

        assert actual_ids == expected_ids

    # max_samples must be applied after split selection.
    provider = LiveCodeBenchDataset()
    provider.load(split="test", seed=seed, max_samples=3)

    actual_ids = [
        record.record_id
        for record in provider.iter_records()
    ]

    expected_ids = [
        f"problem-{i}"
        for i in shuffled_indices[cut:cut + 3]
    ]

    assert actual_ids == expected_ids

def test_lazy_scorer_checks_empty_tests_before_extracting_code() -> None:
    class EmptyTestCaseSource:
        def iter_cases(self):
            return iter(())

    record = EvalRecord(
        record_id="empty-tests",
        problem="Test problem",
        reference="",
        category="coding",
        metadata={"_test_case_source": EmptyTestCaseSource()},
    )

    scorer = LiveCodeBenchScorer()
    score, metadata = scorer.score(record, "No code here")

    assert score is None
    assert metadata == {"reason": "no_test_cases"}


