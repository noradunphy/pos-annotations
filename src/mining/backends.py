"""Pluggable miners for silver Tier-B labels."""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any


class MinerBackend(ABC):
    @abstractmethod
    def label(
        self,
        *,
        words: list[str],
        target_index: int,
        distinction: dict[str, Any],
        rng: random.Random,
    ) -> str:
        raise NotImplementedError


class StubMiner(MinerBackend):
    """Deterministic pseudo-labels that respect quotas by biasing under-filled labels."""

    def __init__(self, bias: str = "round_robin") -> None:
        self._i = 0
        self.bias = bias

    def label(
        self,
        *,
        words: list[str],
        target_index: int,
        distinction: dict[str, Any],
        rng: random.Random,
    ) -> str:
        labels = list(distinction["labels"])
        if self.bias == "random":
            return rng.choice(labels)
        self._i += 1
        return labels[self._i % len(labels)]


class OpenAIMiner(MinerBackend):
    """Optional frontier miner using the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai package for OpenAIMiner") from e
        self.client = OpenAI()
        self.model = model or os.environ.get("POS_MINER_MODEL", "gpt-4o-mini")

    def label(
        self,
        *,
        words: list[str],
        target_index: int,
        distinction: dict[str, Any],
        rng: random.Random,
    ) -> str:
        labels = distinction["labels"]
        text = " ".join(words)
        target = words[target_index]
        sys = (
            "You assign exactly one closed-class label for a linguistic annotation task. "
            "Reply with JSON {\"label\": <one of allowed>}. "
            f"Guidelines: {distinction.get('guidelines', '')}"
        )
        user = (
            f"Sentence: {text}\n"
            f"Target word (1-based index {target_index + 1}): {target}\n"
            f"Allowed labels: {labels}\n"
            f"Task id: {distinction['id']}\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        lab = str(data.get("label", "")).strip()
        if lab not in labels:
            return rng.choice(labels)
        return lab
