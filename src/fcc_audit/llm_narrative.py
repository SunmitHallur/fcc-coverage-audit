"""Optional pluggable LLM narrative backend for case-file prose drafting.

Default: ``none`` — no LLM called; the deterministic casefile.py output is used as-is.

Backends:
  - ``none``   (default) — identity pass-through; returns the pre-built Markdown.
  - ``local``  — calls Ollama REST API (http://localhost:11434) or llama.cpp server.
  - ``gemini`` — calls the Gemini API via the ``google-generativeai`` package.

The deterministic scoring/flagging core is never modified by this module. The LLM
is only allowed to *rephrase* the recommendation and bullet sections, not invent facts.
All structured numeric data and flags come from the pipeline; the LLM only improves
prose readability.

Usage:
  backend = build_backend("none")          # no LLM
  backend = build_backend("local")         # Ollama on localhost
  backend = build_backend("gemini", api_key="...")

  narrative = backend.draft(case_file_markdown, row_dict)
"""
from __future__ import annotations

import json
import logging
import textwrap
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = textwrap.dedent("""\
You are a telecom-policy analyst at the FCC.
You are given a structured case file for a broadband coverage audit.
Your job is to rewrite ONLY the "Recommendation" section (Section 4) in plain, professional language
suitable for a regulatory memo.

Rules:
- Do NOT change any numbers, percentages, or scores.
- Do NOT add new factual claims not present in the input.
- Keep the structure: flag status, recommended action, rationale.
- Aim for 3-5 sentences, formal but readable.
- Respond with ONLY the replacement text for Section 4 (no heading, no preamble).

--- CASE FILE ---
{case_file}
--- END ---
""")


# ── Abstract base ─────────────────────────────────────────────────────────────

class NarrativeBackend(ABC):
    """Base class for LLM narrative backends."""

    @abstractmethod
    def draft(self, case_file_md: str, row: dict[str, Any]) -> str:
        """Return the case file markdown with an improved recommendation section.

        If the backend fails or is unavailable, return *case_file_md* unchanged.
        """

    def _inject_recommendation(self, original_md: str, new_rec: str) -> str:
        """Replace the text under '## 4. Recommendation' with *new_rec*."""
        lines = original_md.split("\n")
        out: list[str] = []
        in_rec = False
        injected = False
        for line in lines:
            if line.startswith("## 4. Recommendation"):
                in_rec = True
                out.append(line)
                out.append("")
                out.append(new_rec.strip())
                out.append("")
                injected = True
                continue
            if in_rec and line.startswith("## "):
                in_rec = False
            if not in_rec:
                out.append(line)
        if not injected:
            out.append("\n" + new_rec.strip())
        return "\n".join(out)


# ── None backend (default) ────────────────────────────────────────────────────

class NoneBackend(NarrativeBackend):
    """Identity pass-through — returns case file unmodified."""

    def draft(self, case_file_md: str, row: dict[str, Any]) -> str:
        return case_file_md


# ── Local backend (Ollama / llama.cpp) ───────────────────────────────────────

class LocalBackend(NarrativeBackend):
    """Calls a locally running Ollama or llama.cpp server.

    Expects an OpenAI-compatible ``/v1/chat/completions`` endpoint
    (Ollama ≥ 0.1.29 exposes this at http://localhost:11434/v1).
    Falls back to Ollama's native ``/api/generate`` endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def draft(self, case_file_md: str, row: dict[str, Any]) -> str:
        prompt = _PROMPT_TEMPLATE.format(case_file=case_file_md)
        try:
            import requests  # already a required dep
            # Try OpenAI-compat endpoint first (Ollama ≥ 0.1.29)
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "temperature": 0.3},
                timeout=self.timeout,
            )
            if resp.ok:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                return self._inject_recommendation(case_file_md, text)

            # Fallback: Ollama native generate
            resp2 = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            if resp2.ok:
                text = resp2.json().get("response", "").strip()
                return self._inject_recommendation(case_file_md, text)

            log.warning("Local LLM request failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            log.warning("Local LLM backend error: %s", exc)
        return case_file_md


# ── Gemini backend ────────────────────────────────────────────────────────────

class GeminiBackend(NarrativeBackend):
    """Calls the Google Gemini API via ``google-generativeai``."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    def draft(self, case_file_md: str, row: dict[str, Any]) -> str:
        prompt = _PROMPT_TEMPLATE.format(case_file=case_file_md)
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            m = genai.GenerativeModel(self.model)
            response = m.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 512},
            )
            text = response.text.strip()
            return self._inject_recommendation(case_file_md, text)
        except ImportError:
            log.warning("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as exc:
            log.warning("Gemini backend error: %s", exc)
        return case_file_md


# ── Factory ───────────────────────────────────────────────────────────────────

def build_backend(
    backend: str = "none",
    *,
    local_url: str = "http://localhost:11434",
    local_model: str = "llama3",
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-1.5-flash",
    timeout: int = 60,
) -> NarrativeBackend:
    """Build and return the appropriate narrative backend.

    Parameters
    ----------
    backend : {"none", "local", "gemini"}
        Which backend to use. ``"none"`` is the default and safe choice.
    local_url :
        Base URL of the Ollama/llama.cpp server (only for ``"local"``).
    local_model :
        Model name on the local server (e.g. ``"llama3"``, ``"mistral"``).
    gemini_api_key :
        Gemini API key (only for ``"gemini"``; can also come from
        ``GEMINI_API_KEY`` env var if not passed explicitly).
    gemini_model :
        Gemini model name (e.g. ``"gemini-1.5-flash"``).
    timeout :
        HTTP timeout in seconds (only for ``"local"``).
    """
    backend = (backend or "none").lower()
    if backend == "none":
        return NoneBackend()
    if backend == "local":
        return LocalBackend(base_url=local_url, model=local_model, timeout=timeout)
    if backend == "gemini":
        import os
        key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            log.warning("Gemini backend: no API key provided. Falling back to none.")
            return NoneBackend()
        return GeminiBackend(api_key=key, model=gemini_model)

    log.warning("Unknown LLM backend %r — falling back to none.", backend)
    return NoneBackend()


# ── Config integration ────────────────────────────────────────────────────────

def backend_from_config(cfg: Any) -> NarrativeBackend:
    """Build a backend from the pipeline config object.

    Reads ``cfg.llm.backend``, ``cfg.llm.local_url``, ``cfg.llm.model``,
    ``cfg.llm.gemini_api_key``.  All fields are optional; default is ``none``.
    """
    llm_cfg: dict[str, Any] = {}
    try:
        llm_cfg = dict(cfg.raw.get("llm", {}))
    except Exception:
        pass

    return build_backend(
        backend=llm_cfg.get("backend", "none"),
        local_url=llm_cfg.get("local_url", "http://localhost:11434"),
        local_model=llm_cfg.get("model", "llama3"),
        gemini_api_key=llm_cfg.get("gemini_api_key"),
        gemini_model=llm_cfg.get("gemini_model", "gemini-1.5-flash"),
    )
