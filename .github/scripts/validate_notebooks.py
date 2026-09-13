"""Confere que todo .ipynb do repositório é JSON válido.

Um notebook corrompido ou vazio passa despercebido em `git status`/`git diff`
e só quebra quando alguém tenta abrir ou executar o arquivo.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    failed = []
    for nb_path in REPO_ROOT.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in nb_path.parts:
            continue
        try:
            json.loads(nb_path.read_text())
        except Exception as exc:
            failed.append((nb_path.relative_to(REPO_ROOT), exc))

    if failed:
        for path, exc in failed:
            print(f"INVALID: {path}: {exc}")
        return 1

    print("Todos os notebooks são JSON válido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
