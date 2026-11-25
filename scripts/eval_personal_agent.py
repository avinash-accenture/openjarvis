#!/usr/bin/env python3
"""Evaluate the personal agent against your real mail and calendar.

Run on the Mac, from the repository root, with `jarvis serve` running:

    .venv/bin/python scripts/eval_personal_agent.py --topic insurance

For each question it computes the ground truth by calling the read-only tools
directly, then asks the agent through the server API and checks its citations:

* every cited id must appear in the ground truth (no invented sources);
* when the ground truth has items, the answer must cite at least one;
* when the ground truth is empty, the answer must cite nothing.

Answers are printed for you to read; nothing is sent anywhere but localhost.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Callable, List, Tuple

import httpx

CITATION = re.compile(r"\[((?:m\d+)|(?:e[0-9a-f]{6,10}))\]")


def ids(text: str) -> set[str]:
    return set(CITATION.findall(text or ""))


def _api_key() -> str:
    from openjarvis.cli.serve import _resolve_api_key

    key = _resolve_api_key()
    if not key:
        sys.exit("No API key configured (jarvis auth create-key).")
    return key


def ask(url: str, key: str, model: str, question: str, timeout: float) -> str:
    response = httpx.post(
        f"{url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": question}]},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def cases(topic: str, sender: str) -> List[Tuple[str, Callable[[], str]]]:
    from openjarvis.tools.personal_data import (
        GetCalendarEventsTool,
        SearchEmailTool,
        UpcomingSpecialDatesTool,
    )

    items: List[Tuple[str, Callable[[], str]]] = [
        (
            "What do I have on my calendar tomorrow?",
            lambda: GetCalendarEventsTool().execute(start_date="tomorrow").content,
        ),
        (
            "What is on my calendar over the next 7 days?",
            lambda: (
                GetCalendarEventsTool()
                .execute(start_date="today", end_date="in 6 days")
                .content
            ),
        ),
        (
            "Are there any birthdays or anniversaries in the next two weeks?",
            lambda: UpcomingSpecialDatesTool().execute(days_ahead=14).content,
        ),
        (
            f"Find my most recent email about {topic}.",
            lambda: SearchEmailTool().execute(query=topic, days_back=90).content,
        ),
        (
            "What did the email from zebra-unicorn-holdings say?",
            lambda: (
                SearchEmailTool()
                .execute(sender="zebra-unicorn-holdings", days_back=365)
                .content
            ),
        ),
    ]
    if sender:
        items.append(
            (
                f"Do I have any unread email from {sender}?",
                lambda: (
                    SearchEmailTool()
                    .execute(sender=sender, unread_only=True, days_back=30)
                    .content
                ),
            )
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="")
    parser.add_argument("--topic", default="insurance")
    parser.add_argument("--sender", default="", help="e.g. part of an address")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    from openjarvis.core.config import load_config

    config = load_config()
    model = args.model or config.intelligence.default_model or "qwen3.5:2b"
    key = _api_key()

    failures = 0
    for question, truth_fn in cases(args.topic, args.sender):
        truth = truth_fn()
        truth_ids = ids(truth)
        started = time.monotonic()
        try:
            answer = ask(args.url, key, model, question, args.timeout)
        except httpx.HTTPError as exc:
            answer = f"(request failed: {exc})"
        seconds = time.monotonic() - started
        cited = ids(answer)
        problems = []
        if re.search(r"failed|not available|could not", truth, re.IGNORECASE):
            problems.append(f"setup problem, the tool itself failed: {truth[:120]}")
        if cited - truth_ids:
            problems.append(f"cites ids not in the data: {sorted(cited - truth_ids)}")
        if truth_ids and not cited:
            problems.append("data exists but the answer cites nothing")
        if not truth_ids and cited:
            problems.append("no data exists but the answer cites sources")
        failures += bool(problems)
        print("=" * 78)
        print(f"Q: {question}   ({seconds:.0f}s)")
        print(f"Ground truth ({len(truth_ids)} item(s)):")
        for line in truth.splitlines()[:8]:
            print(f"   {line}")
        print("Answer:")
        for line in answer.splitlines():
            print(f"   {line}")
        print(
            "Result: " + ("PASS" if not problems else "FAIL — " + "; ".join(problems))
        )
    print("=" * 78)
    print(
        "All citation checks passed."
        if not failures
        else f"{failures} question(s) failed."
    )
    print("Also read each answer: are times, names and dates exactly right?")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
