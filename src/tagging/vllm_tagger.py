"""Run vLLM with structured JSON outputs for subclass tagging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.tagging.prompt_render import label_json_schema, render_distinction_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAGGER_CFG = PROJECT_ROOT / "configs" / "tagger.yaml"


@dataclass
class TaggerConfig:
    model: str
    trust_remote_code: bool
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    dtype: str
    quantization: str | None
    temperature: float
    max_tokens: int
    mock: bool

    @staticmethod
    def load(path: Path | None = None) -> "TaggerConfig":
        p = path or DEFAULT_TAGGER_CFG
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return TaggerConfig(
            model=str(raw.get("model", "Qwen/Qwen2.5-14B-Instruct-AWQ")),
            trust_remote_code=bool(raw.get("trust_remote_code", True)),
            tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(raw.get("max_model_len", 8192)),
            dtype=str(raw.get("dtype", "auto")),
            quantization=raw.get("quantization"),
            temperature=float(raw.get("temperature", 0.1)),
            max_tokens=int(raw.get("max_tokens", 128)),
            mock=bool(raw.get("mock", False)),
        )


class VLLMTagger:
    def __init__(self, cfg: TaggerConfig | None = None, cfg_path: Path | None = None) -> None:
        self.cfg = cfg or TaggerConfig.load(cfg_path)
        self._llm = None
        if not self.cfg.mock:
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError("vllm is required unless tagger.mock=true") from e
            kwargs: dict[str, Any] = {
                "model": self.cfg.model,
                "trust_remote_code": self.cfg.trust_remote_code,
                "tensor_parallel_size": self.cfg.tensor_parallel_size,
                "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
                "max_model_len": self.cfg.max_model_len,
                "dtype": self.cfg.dtype,
            }
            if self.cfg.quantization:
                kwargs["quantization"] = self.cfg.quantization
            self._llm = LLM(**kwargs)

    def predict_label(
        self,
        *,
        distinction: dict[str, Any],
        words: list[str],
        target_index: int,
        icl_path: Path,
    ) -> tuple[str, str]:
        labels = list(distinction["labels"])
        prompt = render_distinction_prompt(
            distinction=distinction,
            words=words,
            target_index=target_index,
            icl_path=icl_path,
            max_icl=3,
        )
        if self.cfg.mock:
            return labels[0], prompt
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        schema = label_json_schema(labels)
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        assert self._llm is not None
        outs = self._llm.generate([prompt], sampling_params=sp)
        text = outs[0].outputs[0].text.strip()
        data = json.loads(text)
        lab = str(data.get("label", ""))
        if lab not in labels:
            lab = labels[0]
        return lab, prompt

    def predict_batch(
        self,
        *,
        distinction: dict[str, Any],
        items: list[tuple[list[str], int]],
        icl_path: Path,
    ) -> list[str]:
        """Predict labels for a batch of (words, target_index) pairs sharing one distinction."""
        labels = list(distinction["labels"])
        prompts = [
            render_distinction_prompt(
                distinction=distinction, words=w, target_index=ti,
                icl_path=icl_path, max_icl=3,
            )
            for w, ti in items
        ]
        if self.cfg.mock:
            return [labels[0]] * len(prompts)
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        schema = label_json_schema(labels)
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        assert self._llm is not None
        outs = self._llm.generate(prompts, sampling_params=sp)
        results: list[str] = []
        for o in outs:
            text = o.outputs[0].text.strip()
            try:
                data = json.loads(text)
                lab = str(data.get("label", ""))
            except json.JSONDecodeError:
                lab = ""
            if lab not in labels:
                lab = labels[0]
            results.append(lab)
        return results
