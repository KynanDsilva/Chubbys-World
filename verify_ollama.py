"""Phase 1 verification script: check Ollama connectivity and schema-constrained JSON output."""

from __future__ import annotations

import json
import sys
import urllib.error

from pydantic import BaseModel

import db
from llm import OllamaClient


class ToyVerificationSchema(BaseModel):
    """Minimal schema for verifying Ollama structured JSON generation."""

    greeting: str
    battery_ok: bool
    next_idea: str


def main() -> int:
    cfg = db.load_config()
    client = OllamaClient(cfg)
    messages = [
        {
            "role": "system",
            "content": "You are Chubby, a small round scrapyard robot. Reply as JSON matching the schema.",
        },
        {
            "role": "user",
            "content": "Say hello and share one tiny idea for today.",
        },
    ]
    try:
        raw = client._post_chat(messages, ToyVerificationSchema.model_json_schema())
        validated = ToyVerificationSchema.model_validate_json(raw)
        print(json.dumps(validated.model_dump(), indent=2))
        return 0
    except urllib.error.URLError as exc:
        print(
            f"Ollama unreachable at {client.base_url}: {exc}. "
            "Install/start Ollama and run `ollama pull qwen3:4b` before running with --policy llm.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
