"""Contagem por classe e alerta numérico do cariótipo.

Regra do `karyotype_assembler.py` de Rosas-Alatriste et al. (Scientific Reports, 2026): 2
cromossomos por autossomo; se houver algum Y, o esperado é XY, senão XX. Qualquer classe com
contagem diferente do esperado vira um alerta -- é a base da detecção de aneuploidias (Fase 5).
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from voc_to_yolo import DENVER_CLASSES

AUTOSOMES = [c for c in DENVER_CLASSES if c not in ("X", "Y")]


def count_by_class(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    unknown = set(counts) - set(DENVER_CLASSES)
    if unknown:
        raise ValueError(f"classes desconhecidas: {sorted(unknown)}")
    return {c: counts.get(c, 0) for c in DENVER_CLASSES}


def expected_counts(observed: dict[str, int]) -> dict[str, int]:
    expected = {c: 2 for c in AUTOSOMES}
    if observed.get("Y", 0) > 0:
        expected.update(X=1, Y=1)
    else:
        expected.update(X=2, Y=0)
    return expected


def numerical_alerts(observed: dict[str, int]) -> dict[str, dict[str, int]]:
    """{classe: {"observed", "expected"}} para cada classe fora do esperado; vazio = cariótipo normal."""
    return {
        c: {"observed": observed.get(c, 0), "expected": e}
        for c, e in expected_counts(observed).items()
        if observed.get(c, 0) != e
    }
