"""
.claude/hooks/post_test_file.py
────────────────────────────────
PostToolUse hook: após criar ou editar um arquivo de teste,
executa pytest automaticamente naquele arquivo e exibe o resultado.

Arquivos considerados teste:
  - test_*.py
  - *_test.py

Exit code sempre 0 — o hook é informativo e não bloqueia.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def is_test_file(path: str) -> bool:
    """Retorna True se o caminho aponta para um arquivo de teste."""
    basename = os.path.basename(path)
    return basename.startswith("test_") or basename.endswith("_test.py")


def main() -> None:
    payload = json.load(sys.stdin)
    tool_input: dict = payload.get("tool_input", {})
    file_path: str = tool_input.get("file_path", "")

    if not file_path.endswith(".py") or not is_test_file(file_path):
        sys.exit(0)

    if not os.path.exists(file_path):
        sys.exit(0)

    separator = "─" * 60
    print(separator)
    print(f"  Hook SmartRec — pytest automático: {file_path}")
    print(separator)

    result = subprocess.run(
        ["pytest", file_path, "-v", "--tb=short", "--no-header"],
        text=True,
        capture_output=False,  # deixa saída ir direto ao terminal
    )

    print(separator)
    if result.returncode == 0:
        print("  Todos os testes passaram.")
    else:
        print(f"  {result.returncode} teste(s) falharam — revise o arquivo acima.")
    print(separator)

    sys.exit(0)  # nunca bloqueia — apenas informativo


if __name__ == "__main__":
    main()
