"""Gemini API tagger for subclass annotation.

Replaces the local vLLM tagger with calls to the Google Gemini API.
System and user message parts are routed through proper chat roles via
the google-genai SDK, with no dependency on bootstrapped ICL files.
Handcrafted ICL examples per distinction are embedded in the system prompt
(see prompts/gemini_icl.yaml and prompt_render.render_distinction_messages).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.tagging.prompt_render import render_distinction_messages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEMINI_CFG = PROJECT_ROOT / "configs" / "gemini.yaml"


@dataclass
class GeminiConfig:
    flash_model: str
    frontier_model: str
    temperature: float
    max_output_tokens: int
    thinking_budget: int  # 0 = disabled; -1 = model default; >0 = token cap

    @staticmethod
    def load(path: Path | None = None) -> "GeminiConfig":
        p = path or DEFAULT_GEMINI_CFG
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return GeminiConfig(
            flash_model=str(raw.get("flash_model", "gemini-2.5-flash")),
            frontier_model=str(raw.get("frontier_model", "gemini-2.5-pro")),
            temperature=float(raw.get("temperature", 0.1)),
            max_output_tokens=int(raw.get("max_output_tokens", 1024)),
            thinking_budget=int(raw.get("thinking_budget", 0)),
        )

    def resolve_model(self, alias: str) -> str:
        """Resolve 'flash' / 'frontier' aliases to full model names."""
        if alias == "flash":
            return self.flash_model
        if alias == "frontier":
            return self.frontier_model
        # Treat any other string as a literal model name
        return alias


class GeminiTagger:
    """Annotate tokens using the Google Gemini Chat API.

    Uses google-genai SDK. Reads GEMINI_API_KEY from the environment unless
    api_key is passed explicitly.

    The tagger shares the predict_label / predict_batch interface with
    VLLMTagger so it can be dropped in as a replacement in annotate.py.
    Unlike VLLMTagger it does NOT accept an icl_path argument; ICL examples
    are baked into the system prompt via prompts/gemini_icl.yaml.
    """

    def __init__(
        self,
        model: str = "flash",
        cfg: GeminiConfig | None = None,
        cfg_path: Path | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types  # noqa: F401 — ensure SDK is complete
        except ImportError as e:
            raise ImportError(
                "google-genai is required for GeminiTagger. "
                "Install it with: pip install google-genai"
            ) from e

        self.cfg = cfg or GeminiConfig.load(cfg_path)
        self.model_name = self.cfg.resolve_model(model)
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in the environment "
                "or pass api_key= to GeminiTagger."
            )
        from google import genai as _genai

        self._client = _genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, system_msg: str, user_msg: str) -> str:
        """Send one system+user turn; return raw response text.

        For thinking models (e.g. gemini-2.5-pro) the SDK may return multiple
        content parts — some with thought=True (internal reasoning) and one
        with the actual answer. resp.text can be None/empty in that case, so
        we collect non-thought text parts explicitly as a fallback.
        """
        from google.genai import types

        gen_cfg: dict[str, Any] = {
            "system_instruction": system_msg,
            "temperature": self.cfg.temperature,
            "max_output_tokens": self.cfg.max_output_tokens,
        }
        # thinking_budget=-1 means "model default" — don't pass ThinkingConfig at all.
        # thinking_budget>0 sets an explicit token cap.
        # thinking_budget=0 disables thinking (only valid for flash; pro requires thinking).
        if self.cfg.thinking_budget >= 0:
            gen_cfg["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.cfg.thinking_budget
            )
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=user_msg,
            config=types.GenerateContentConfig(**gen_cfg),
        )

        # Fast path: resp.text works for non-thinking models
        if resp.text:
            return resp.text

        # Fallback: manually collect non-thought text parts
        # (needed for gemini-2.5-pro whose thinking parts shadow resp.text)
        text_parts: list[str] = []
        for candidate in resp.candidates or []:
            content = getattr(candidate, "content", None)
            parts = (getattr(content, "parts", None) or []) if content else []
            for part in parts:
                is_thought = getattr(part, "thought", False)
                part_text = getattr(part, "text", None)
                if part_text and not is_thought:
                    text_parts.append(part_text)

        if text_parts:
            return "".join(text_parts)

        # Nothing found — emit a one-time diagnostic to stderr so the caller
        # can see the raw response structure without crashing.
        if not getattr(self, "_empty_resp_warned", False):
            import sys
            self._empty_resp_warned = True
            print("\n[DIAG] Empty response — full resp object:", file=sys.stderr)
            print(f"  resp.text        = {resp.text!r}", file=sys.stderr)
            print(f"  resp.candidates  = {resp.candidates!r}", file=sys.stderr)
            try:
                print(f"  resp.model_dump  = {resp.model_dump()}", file=sys.stderr)
            except Exception:
                pass
            try:
                print(f"  resp (repr)      = {repr(resp)}", file=sys.stderr)
            except Exception:
                pass
        return ""

    def _parse(self, text: str, labels: list[str]) -> str:
        """Parse JSON response; fall back to first label on failure.

        Matching is case-insensitive so that frontier models which capitalise
        label names (e.g. "Common" instead of "common") are handled correctly.
        If the full response isn't valid JSON, we try to extract the first
        JSON object embedded anywhere in the text (handles models that add
        explanation prose around the JSON).
        """
        labels_lower = {lb.lower(): lb for lb in labels}

        def match(raw: str) -> str:
            """Return the canonical label for raw, or '' if no match."""
            return labels_lower.get(raw.strip().lower(), "")

        text = text.strip()

        # 1. Strip markdown fences
        if "```" in text:
            text = re.sub(r"```[a-z]*\n?", "", text).strip()

        # 2. Try parsing the whole response as JSON
        try:
            data = json.loads(text)
            lab = match(str(data.get("label", "")))
            if lab:
                return lab
        except (json.JSONDecodeError, AttributeError):
            pass

        # 3. Extract the first {...} block and parse that
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                lab = match(str(data.get("label", "")))
                if lab:
                    return lab
            except (json.JSONDecodeError, AttributeError):
                pass

        # 4. Last resort: scan for any label string in the response
        text_lower = text.lower()
        for lb_lower, lb in labels_lower.items():
            if lb_lower in text_lower:
                return lb

        return labels[0]

    # ------------------------------------------------------------------
    # Public interface (mirrors VLLMTagger)
    # ------------------------------------------------------------------

    def predict_label(
        self,
        *,
        distinction: dict[str, Any],
        words: list[str],
        target_index: int,
        icl_path: Path | None = None,  # accepted but ignored; ICL is in system prompt
        log_response: bool = False,
    ) -> tuple[str, str]:
        """Predict one label. Returns (label, system_msg) for inspection.

        Set log_response=True to also print the raw API response to stderr,
        which is useful for diagnosing label-format issues.
        """
        labels = list(distinction["labels"])
        system_msg, user_msg = render_distinction_messages(
            distinction=distinction,
            words=words,
            target_index=target_index,
        )
        raw = self._call(system_msg, user_msg)
        if log_response:
            import sys
            print(f"    [raw] {raw!r}", file=sys.stderr)
        lab = self._parse(raw, labels)
        return lab, system_msg

    def predict_batch(
        self,
        *,
        distinction: dict[str, Any],
        items: list[tuple[list[str], int]],
        icl_path: Path | None = None,  # accepted but ignored
    ) -> list[str]:
        """Predict labels for a batch of (words, target_index) pairs.

        Gemini has no native batch endpoint so calls are made sequentially.
        The system prompt (which contains guidelines and ICL examples) is
        identical for every item in the batch, so it is built once.
        """
        labels = list(distinction["labels"])
        results: list[str] = []
        for words, target_index in items:
            system_msg, user_msg = render_distinction_messages(
                distinction=distinction,
                words=words,
                target_index=target_index,
            )
            text = self._call(system_msg, user_msg)
            results.append(self._parse(text, labels))
        return results
