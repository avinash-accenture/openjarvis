"""WildChat dataset provider (allenai/WildChat-1M).

Filters to English single-turn conversations for chat quality evaluation.
"""

from __future__ import annotations

import random
from typing import Iterable, List, MutableMapping, Optional

from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.types import EvalRecord


class WildChatDataset(DatasetProvider):
    """WildChat conversation quality benchmark."""

    dataset_id = "wildchat"
    dataset_name = "WildChat"

    _hf_path = "allenai/WildChat-1M"
    _default_split = "train"

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

        use_split = split or self._default_split
        dataset = load_dataset(self._hf_path, split=use_split)

        # First pass: collect indices of valid English single-turn conversations.
        valid_indices: List[int] = []

        for idx in range(len(dataset)):
            row = dataset[int(idx)]
            if not isinstance(row, MutableMapping):
                row = dict(row)

            language = str(row.get("language") or "").lower()
            if language != "english":
                continue

            conversation = row.get("conversation")
            if not isinstance(conversation, list) or len(conversation) != 2:
                continue

            user_msg = conversation[0]
            asst_msg = conversation[1]

            if (
                str(user_msg.get("role", "")) != "user"
                or str(asst_msg.get("role", "")) != "assistant"
            ):
                continue

            user_content = str(user_msg.get("content", "")).strip()
            asst_content = str(asst_msg.get("content", "")).strip()

            if not user_content or not asst_content:
                continue

            valid_indices.append(int(idx))

        # Preserve original behavior: shuffle valid rows, then limit samples.
        if seed is not None:
            random.Random(seed).shuffle(valid_indices)

        if max_samples is not None:
            valid_indices = valid_indices[:max_samples]

        # Second pass: retrieve only selected rows and build evaluation records.
        self._records = []

        for record_idx, dataset_idx in enumerate(valid_indices):
            raw = dataset[int(dataset_idx)]
            if not isinstance(raw, MutableMapping):
                raw = dict(raw)

            conversation = raw["conversation"]
            user_content = str(conversation[0].get("content", "")).strip()
            asst_content = str(conversation[1].get("content", "")).strip()

            self._records.append(
                EvalRecord(
                    record_id=f"wildchat-{record_idx}",
                    problem=user_content,
                    reference=asst_content,
                    category="chat",
                    subject="conversation",
                    metadata={
                        "model": str(raw.get("model", "")),
                        "language": "english",
                    },
                )
            )

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)


__all__ = ["WildChatDataset"]
