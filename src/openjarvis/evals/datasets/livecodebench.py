"""LiveCodeBench dataset provider — competitive programming code generation.

Loads problems from the LiveCodeBench HuggingFace dataset
(livecodebench/code_generation_lite) for evaluating code generation capability.

Reference: https://livecodebench.github.io/
           https://github.com/LiveCodeBench/LiveCodeBench
"""

from __future__ import annotations

import json
import logging
import random
import struct
import re
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.types import EvalRecord

LOGGER = logging.getLogger(__name__)

_HF_DATASET = "livecodebench/code_generation_lite"
_HF_DATASET_FULL = "livecodebench/code_generation"
_DEFAULT_SPLIT = "test"

_JSON_STRING_SPECIAL = re.compile(rb'["\\]')
_JSON_STRUCTURE = re.compile(rb'["{}\[\]]')

_PROMPT_TEMPLATE = """Solve the following competitive programming problem. Write a complete Python solution that reads from stdin and writes to stdout.

## Problem

{problem_statement}

## Input Format

{input_format}

## Output Format

{output_format}

## Constraints

{constraints}

## Examples

{examples}

Write ONLY the Python code solution. Do not include any explanations."""


def _format_examples(example_inputs: list, example_outputs: list) -> str:
    """Format input/output examples for the prompt."""
    parts = []
    for i, (inp, out) in enumerate(zip(example_inputs, example_outputs), 1):
        parts.append(f"Input {i}:\n{inp}\n\nOutput {i}:\n{out}")
    return "\n\n".join(parts) if parts else "No examples provided."


def _parse_json_field(value: Any) -> Any:
    """Parse a field that might be a JSON string or already parsed."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value

def _extract_prompt_examples(
    raw: MutableMapping[str, object],
    limit: int = 2,
) -> Tuple[List[str], List[str]]:
    """Extract only a limited number of examples for the prompt."""
    test_inputs: List[str] = []
    test_outputs: List[str] = []

    public_cases = _parse_json_field(
        raw.get("public_test_cases", raw.get("sample_io", []))
    )
    hidden_cases = _parse_json_field(
        raw.get("hidden_test_cases", raw.get("test_cases", []))
    )

    for cases in (public_cases, hidden_cases):
        if not isinstance(cases, list):
            continue

        for case in cases:
            if isinstance(case, str):
                case = _parse_json_field(case)

            if not isinstance(case, dict):
                continue

            inp = str(case.get("input", "")).strip()
            out = str(case.get("output", "")).strip()

            if inp or out:
                test_inputs.append(inp)
                test_outputs.append(out)

                if len(test_inputs) >= limit:
                    return test_inputs, test_outputs

    # Preserve the original independent fallback behavior.
    flat_inputs = _parse_json_field(raw.get("input", raw.get("test_inputs", [])))
    flat_outputs = _parse_json_field(raw.get("output", raw.get("test_outputs", [])))

    if not test_inputs and isinstance(flat_inputs, list):
        test_inputs = [str(value).strip() for value in flat_inputs[:limit]]

    if not test_outputs and isinstance(flat_outputs, list):
        test_outputs = [str(value).strip() for value in flat_outputs[:limit]]

    return test_inputs[:limit], test_outputs[:limit]

def _extract_test_cases(
    raw: MutableMapping[str, object],
) -> Tuple[List[str], List[str]]:
    """Normalize all supported LiveCodeBench test-case formats."""
    test_inputs = _parse_json_field(raw.get("input", raw.get("test_inputs", [])))
    test_outputs = _parse_json_field(raw.get("output", raw.get("test_outputs", [])))
    public_test_cases = _parse_json_field(
        raw.get("public_test_cases", raw.get("sample_io", []))
    )
    private_test_cases = _parse_json_field(
        raw.get("private_test_cases", [])
    )
    hidden_test_cases = _parse_json_field(
        raw.get("hidden_test_cases", raw.get("test_cases", []))
    )

    all_test_inputs: List[str] = []
    all_test_outputs: List[str] = []

    for cases in [public_test_cases, private_test_cases, hidden_test_cases]:
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict):
                    inp = str(case.get("input", "")).strip()
                    out = str(case.get("output", "")).strip()
                    if inp or out:
                        all_test_inputs.append(inp)
                        all_test_outputs.append(out)
                elif isinstance(case, str):
                    parsed = _parse_json_field(case)
                    if isinstance(parsed, dict):
                        inp = str(parsed.get("input", "")).strip()
                        out = str(parsed.get("output", "")).strip()
                        if inp or out:
                            all_test_inputs.append(inp)
                            all_test_outputs.append(out)

    if not all_test_inputs and isinstance(test_inputs, list):
        for inp in test_inputs:
            all_test_inputs.append(str(inp).strip())

    if not all_test_outputs and isinstance(test_outputs, list):
        for out in test_outputs:
            all_test_outputs.append(str(out).strip())

    return all_test_inputs, all_test_outputs


def _arrow_string_buffer(dataset: Any, column_name: str, row_index: int) -> Any:
    """Return a zero-copy Arrow buffer slice for one string field."""
    import pyarrow as pa

    if column_name not in dataset.column_names:
        return None

    column = dataset.data.column(column_name)
    remaining = row_index

    for chunk in column.chunks:
        if remaining >= len(chunk):
            remaining -= len(chunk)
            continue

        if not chunk[remaining].is_valid:
            return None

        offsets_buffer, data_buffer = chunk.buffers()[1:3]
        absolute_index = chunk.offset + remaining
        offset_size = 8 if pa.types.is_large_string(chunk.type) else 4
        offset_format = "<q" if offset_size == 8 else "<i"

        start = struct.unpack_from(
            offset_format, offsets_buffer, absolute_index * offset_size
        )[0]
        end = struct.unpack_from(
            offset_format, offsets_buffer, (absolute_index + 1) * offset_size
        )[0]

        return data_buffer.slice(start, end - start)

    raise IndexError(f"Row {row_index} is outside column {column_name!r}")


def _iter_json_array(buffer: Any, chunk_size: int = 65536) -> Iterable[Any]:
    """Parse a JSON array incrementally, scanning structural bytes in bulk."""
    import pyarrow as pa

    reader = pa.BufferReader(buffer)
    item = bytearray()
    depth = 0
    in_array = False
    in_string = False
    escaped = False
    string_item = False

    quote = ord('"')
    backslash = ord("\\")
    opening = (ord("{"), ord("["))
    closing = (ord("}"), ord("]"))
    whitespace = b" \t\r\n,"

    while True:
        raw = reader.read(chunk_size)
        if not raw:
            break

        pos = 0
        size = len(raw)

        while pos < size:
            if not in_array:
                if raw[pos] in b" \t\r\n":
                    pos += 1
                    continue
                if raw[pos] != ord("["):
                    raise ValueError("Expected a JSON array")
                in_array = True
                pos += 1
                continue

            if not item:
                while pos < size and raw[pos] in whitespace:
                    pos += 1

                if pos >= size:
                    continue

                if raw[pos] == ord("]"):
                    return

                if raw[pos] not in (ord("{"), quote):
                    raise ValueError(
                        f"Unsupported JSON array item starting with byte {raw[pos]}"
                    )

                item.append(raw[pos])
                if raw[pos] == quote:
                    in_string = True
                    string_item = True
                else:
                    depth = 1
                pos += 1
                continue

            if in_string:
                if escaped:
                    item.append(raw[pos])
                    escaped = False
                    pos += 1
                    continue

                match = _JSON_STRING_SPECIAL.search(raw, pos)
                if match is None:
                    item.extend(raw[pos:])
                    pos = size
                    continue

                next_pos = match.start()
                item.extend(raw[pos : next_pos + 1])

                if raw[next_pos] == backslash:
                    escaped = True
                else:
                    in_string = False
                    if string_item:
                        yield json.loads(item)
                        item.clear()
                        string_item = False

                pos = next_pos + 1
                continue

            # Outside a JSON string, locate the next structural character.
            match = _JSON_STRUCTURE.search(raw, pos)

            if match is None:
                item.extend(raw[pos:])
                pos = size
                continue

            next_pos = match.start()
            item.extend(raw[pos : next_pos + 1])
            token = raw[next_pos]

            if token == quote:
                in_string = True
            elif token in opening:
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    yield json.loads(item)
                    item.clear()

            pos = next_pos + 1

    if not in_array:
        raise ValueError("Empty or invalid JSON array")
    if item:
        raise ValueError("Incomplete JSON array item")

class _LazyTestCaseSource:
    """Retrieve test cases lazily without materializing large Arrow fields."""

    def __init__(self, dataset: Any, row_index: int) -> None:
        self._dataset = dataset
        self._row_index = row_index

    def iter_cases(self) -> Iterable[Tuple[str, str]]:
        """Yield input/output pairs one test case at a time."""

        # Arrow-backed Hugging Face datasets can expose individual columns
        # without converting the entire row into a Python dictionary.
        if hasattr(self._dataset, "data") and hasattr(
            self._dataset.data, "column"
        ):
            column_names = self._dataset.column_names
            structured_columns = [
                name
                for name in (
                    "public_test_cases",
                    "sample_io",
                    "private_test_cases",
                    "hidden_test_cases",
                    "test_cases",
                )
                if name in column_names
            ]

            yielded = 0

            for column_name in structured_columns:
                if column_name == "private_test_cases":
                    buffer = _arrow_string_buffer(
                        self._dataset, column_name, self._row_index
                    )
                    if buffer is None:
                        continue
                    cases = _iter_json_array(buffer)
                else:
                    value = self._dataset.data.column(column_name)[
                        self._row_index
                    ].as_py()
                    value = _parse_json_field(value)
                    if not isinstance(value, list):
                        continue
                    cases = iter(value)

                for case in cases:
                    if isinstance(case, str):
                        case = _parse_json_field(case)

                    if not isinstance(case, dict):
                        continue

                    inp = str(case.get("input", "")).strip()
                    out = str(case.get("output", "")).strip()

                    if inp or out:
                        yielded += 1
                        yield inp, out

            if yielded:
                return

            # Preserve the legacy flat input/output fallback.
            flat_inputs = []
            flat_outputs = []

            for column_name in ("input", "test_inputs"):
                if column_name in column_names:
                    value = self._dataset.data.column(column_name)[
                        self._row_index
                    ].as_py()
                    flat_inputs = _parse_json_field(value)
                    break

            for column_name in ("output", "test_outputs"):
                if column_name in column_names:
                    value = self._dataset.data.column(column_name)[
                        self._row_index
                    ].as_py()
                    flat_outputs = _parse_json_field(value)
                    break

            if isinstance(flat_inputs, list) and isinstance(
                flat_outputs, list
            ):
                for inp, out in zip(flat_inputs, flat_outputs):
                    yield str(inp).strip(), str(out).strip()

            return

        # Compatibility path for ordinary Python and test datasets.
        raw = self._dataset[self._row_index]
        inputs, outputs = _extract_test_cases(raw)

        for inp, out in zip(inputs, outputs):
            yield inp, out

    def load(self) -> Tuple[List[str], List[str]]:
        """Backward-compatible API returning materialized test-case lists."""
        pairs = list(self.iter_cases())
        return (
            [inp for inp, _ in pairs],
            [out for _, out in pairs],
        )

class LiveCodeBenchDataset(DatasetProvider):
    """LiveCodeBench competitive programming benchmark.

    Loads problems from the LiveCodeBench HuggingFace dataset for
    evaluating code generation capability on fresh competitive
    programming problems from LeetCode, AtCoder, and CodeForces.
    """

    dataset_id = "livecodebench"
    dataset_name = "LiveCodeBench"

    def __init__(self) -> None:
        self._records: List[EvalRecord] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        from datasets import load_dataset

        use_split = (
            _DEFAULT_SPLIT
            if split in ("train", "test", "all") or split is None
            else split
        )

        # Try lite version first (smaller, faster download), fall back to full
        dataset = None
        for hf_path in [_HF_DATASET, _HF_DATASET_FULL]:
            try:
                dataset = load_dataset(hf_path, split=use_split)
                LOGGER.info("Loaded LiveCodeBench from %s (%s)", hf_path, use_split)
                break
            except Exception as exc:
                LOGGER.debug("Could not load %s: %s", hf_path, exc)

        if dataset is None:
            raise RuntimeError(
                "Failed to load LiveCodeBench dataset. "
                f"Tried: {_HF_DATASET}, {_HF_DATASET_FULL}. "
                "Ensure 'datasets' is installed and you have network access."
            )

        effective_seed = 42 if seed is None else seed

        # Keep the Hugging Face dataset columnar; avoid materializing all
        # problem statements and test cases as Python dictionaries.
        selected_indices = list(range(len(dataset)))

        if split in ("train", "test", "all"):
            random.Random(effective_seed).shuffle(selected_indices)
            cut = int(len(selected_indices) * 0.2)

            if split == "train":
                selected_indices = selected_indices[:cut]
            elif split == "test":
                selected_indices = selected_indices[cut:]

        elif seed is not None:
            random.Random(seed).shuffle(selected_indices)

        if max_samples is not None:
            selected_indices = selected_indices[:max_samples]

        prompt_columns = [
            "question_content",
            "problem_description",
            "question",
            "problem",
            "question_id",
            "task_id",
            "problem_id",
            "question_title",
            "title",
            "name",
            "difficulty",
            "question_difficulty",
            "platform",
            "source",
            "contest_source",
            "input_format",
            "output_format",
            "constraints",
            "public_test_cases",
            "sample_io",
            "input",
            "test_inputs",
            "output",
            "test_outputs",
            "starter_code",
            "code_stub",
            "time_limit",
            "memory_limit",
            "contest_id",
            "contest_date",
        ]

        prompt_dataset = dataset.select_columns(
            [
                column
                for column in prompt_columns
                if column in dataset.column_names
            ]
        )

        self._records = []
        for idx in selected_indices:
            raw = prompt_dataset[int(idx)]
            record = self._convert_row(
                raw,
                int(idx),
                _LazyTestCaseSource(dataset, int(idx)),
            )

            if record is not None:
                self._records.append(record)


        LOGGER.info("LiveCodeBench: loaded %d problems", len(self._records))

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def _convert_row(
        self,
        raw: MutableMapping[str, object],
        idx: int,
        test_case_source: _LazyTestCaseSource,
    ) -> Optional[EvalRecord]:
        # Extract problem statement — field names vary across dataset versions
        problem_text = str(
            raw.get("question_content")
            or raw.get("problem_description")
            or raw.get("question")
            or raw.get("problem")
            or ""
        ).strip()
        if not problem_text:
            LOGGER.debug("Skipping row %d: no problem text", idx)
            return None

        # Extract problem ID
        problem_id = str(
            raw.get("question_id")
            or raw.get("task_id")
            or raw.get("problem_id")
            or f"lcb-{idx}"
        )

        # Extract title
        title = str(
            raw.get("question_title") or raw.get("title") or raw.get("name") or ""
        ).strip()

        # Extract difficulty
        difficulty = str(
            raw.get("difficulty") or raw.get("question_difficulty") or ""
        ).strip()

        # Extract platform/source
        platform = str(
            raw.get("platform") or raw.get("source") or raw.get("contest_source") or ""
        ).strip()

        # Extract input/output format (may not always be separate fields)
        input_format = str(raw.get("input_format", "")).strip()
        output_format = str(raw.get("output_format", "")).strip()
        constraints = str(raw.get("constraints", "")).strip()
        # Extract only the examples needed for the prompt.
        # Full test cases are retrieved lazily when the scorer needs them.
        example_inputs, example_outputs = _extract_prompt_examples(raw)
        examples_str = _format_examples(example_inputs, example_outputs)

        # Build prompt
        problem = _PROMPT_TEMPLATE.format(
            problem_statement=problem_text,
            input_format=input_format or "(see problem statement)",
            output_format=output_format or "(see problem statement)",
            constraints=constraints or "(see problem statement)",
            examples=examples_str,
        )

        # Starter code (if available)
        starter_code = str(raw.get("starter_code", raw.get("code_stub", ""))).strip()

        # Metadata for scoring
        metadata: Dict[str, Any] = {
            "_test_case_source": test_case_source,
            "difficulty": difficulty,
            "platform": platform,
            "title": title,
        }

        if starter_code:
            metadata["starter_code"] = starter_code

        # Store raw data for any fields the scorer might need
        for key in ["time_limit", "memory_limit", "contest_id", "contest_date"]:
            val = raw.get(key)
            if val is not None:
                metadata[key] = val

        # Build subject from platform and difficulty
        subject = (
            f"{platform}/{difficulty}"
            if platform and difficulty
            else (platform or difficulty or "competitive-programming")
        )

        return EvalRecord(
            record_id=problem_id,
            problem=problem,
            reference="",  # No single reference answer; scored via test execution
            category="coding",
            subject=subject,
            metadata=metadata,
        )


__all__ = ["LiveCodeBenchDataset"]
