#!/usr/bin/env python3
"""Fail when repository source files appear to contain secrets or private data files."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "artifacts", "data", "uploads",
}
PRIVATE_SUFFIXES = {
    ".csv", ".db", ".sqlite", ".sqlite3", ".joblib", ".pkl", ".pickle",
}
SOURCE_SUFFIXES = {
    ".py", ".js", ".css", ".html", ".md", ".txt", ".json", ".toml", ".yaml", ".yml",
}
SECRET_PATTERNS = {
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    "Generic secret assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,}"
    ),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def source_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in {".env", ".DS_Store"}:
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES or path.name in {
            ".gitignore", ".env.example",
        }:
            files.append(path)
    return files


def main() -> int:
    failures = []
    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)}: possible {label}")

    private_files = [
        path.relative_to(ROOT)
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in PRIVATE_SUFFIXES
    ]
    if failures:
        print("Privacy check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Privacy check passed for repository source files.")
    if private_files:
        print("Local private files detected and expected to remain ignored:")
        for path in private_files:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
