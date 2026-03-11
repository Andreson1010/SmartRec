"""
.claude/hooks/pre_write_py.py
─────────────────────────────
PreToolUse hook: antes de escrever qualquer arquivo .py,
valida formatação (black) e lint (flake8).

Comportamento por exit code:
  0  → permite o tool use normalmente
  2  → BLOQUEIA o tool use e envia stderr ao Claude como feedback
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


def main() -> None:
    payload = json.load(sys.stdin)
    tool_input: dict = payload.get("tool_input", {})
    file_path: str = tool_input.get("file_path", "")

    if not file_path.endswith(".py"):
        sys.exit(0)

    # Write: conteúdo vem no payload. Edit: lê o arquivo existente.
    content: str | None = tool_input.get("content")
    if content is None:
        if not os.path.exists(file_path):
            sys.exit(0)
        with open(file_path, encoding="utf-8") as fh:
            content = fh.read()

    # Escreve em arquivo temporário para poder rodar as ferramentas
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    errors: list[str] = []

    try:
        # ── black ─────────────────────────────────────────────────────────
        black_result = subprocess.run(
            ["black", "--check", "--diff", "--quiet", tmp_path],
            capture_output=True,
            text=True,
        )
        if black_result.returncode != 0:
            diff = black_result.stdout.replace(tmp_path, file_path)
            errors.append(f"[black] O arquivo não está formatado corretamente:\n{diff}")

        # ── flake8 ────────────────────────────────────────────────────────
        flake8_result = subprocess.run(
            ["flake8", "--max-line-length=88", "--extend-ignore=E203,W503", tmp_path],
            capture_output=True,
            text=True,
        )
        if flake8_result.returncode != 0:
            output = flake8_result.stdout.replace(tmp_path, file_path)
            errors.append(f"[flake8] Violações de lint encontradas:\n{output.strip()}")

    finally:
        os.unlink(tmp_path)

    if errors:
        separator = "─" * 60
        print(separator, file=sys.stderr)
        print(f"  Hook SmartRec — pré-validação: {file_path}", file=sys.stderr)
        print(separator, file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        print(
            "\nCorrija os problemas acima antes de escrever o arquivo.",
            file=sys.stderr,
        )
        sys.exit(2)  # bloqueia o tool use com feedback ao Claude


if __name__ == "__main__":
    main()
