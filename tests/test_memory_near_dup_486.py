"""Regression test for #486 defect (2): _find_near_duplicate_vec uses a FIXED
k=5 window and filters active=1 AFTER the kNN. When the 5 nearest neighbours
are all inactive (superseded), the active near-duplicate sitting just outside
the window is never seen and the function returns None — a false negative that
causes a duplicate to be written.

The scenario is built so the vec path is exercised (requires sqlite-vec).
RED on the k=5 code; GREEN after the k-escalation fix.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from pinky_memory.store import ReflectionStore
from pinky_memory.types import Reflection, ReflectionType


def _unit(vec: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / mag for x in vec]


def _fact(content: str, embedding: list[float]) -> Reflection:
    return Reflection(type=ReflectionType.fact, content=content, embedding=embedding)


def test_active_near_duplicate_found_past_a_full_window_of_inactive(tmp_path: Path):
    store = ReflectionStore(str(tmp_path / "nd486.db"))

    dims = 8

    def base(x1: float) -> list[float]:
        v = [0.0] * dims
        v[0] = 1.0
        v[1] = x1
        return _unit(v)

    # Query vector.
    query = base(0.0)

    # 5 INACTIVE neighbours, each strictly closer to the query than the active
    # duplicate below (tiny x1 → cosine ~1.0). They fill the k=5 window.
    inactive_ids = []
    for i in range(5):
        r = store.insert(_fact(f"superseded v{i}", base(0.0005 * (i + 1))))
        inactive_ids.append(r.id)

    # The ACTIVE near-duplicate: cosine ~0.98 with the query (well above the
    # 0.90 threshold) but farther than every inactive neighbour, so it lands at
    # position 6 — outside a fixed k=5 window.
    active = store.insert(_fact("real active duplicate", _unit([1.0, 0.2] + [0.0] * (dims - 2))))

    # Deactivate the 5 nearest so only the active duplicate should match.
    for rid in inactive_ids:
        store.deactivate_superseded(rid, superseded_by=active.id)

    if not store._vec_available or store._vec_dimensions != dims:
        pytest.skip("sqlite-vec not available — vec path not exercised")

    result = store.find_near_duplicate(query, threshold=0.90, active_only=True)

    assert result is not None, (
        "active near-duplicate was missed: the 5 nearest neighbours are all "
        "inactive and filled the fixed k window (#486 defect 2)"
    )
    sim, refl = result
    assert refl.id == active.id
    assert sim > 0.90


def test_no_false_positive_when_only_inactive_match(tmp_path: Path):
    """With no active duplicate at all, escalation must still return None
    (it must not start matching inactive rows)."""
    store = ReflectionStore(str(tmp_path / "nd486b.db"))
    dims = 8

    def base(x1: float) -> list[float]:
        v = [0.0] * dims
        v[0] = 1.0
        v[1] = x1
        return _unit(v)

    query = base(0.0)
    ids = [store.insert(_fact(f"superseded {i}", base(0.0005 * (i + 1)))).id for i in range(6)]
    for rid in ids:
        store.deactivate_superseded(rid)

    if not store._vec_available or store._vec_dimensions != dims:
        pytest.skip("sqlite-vec not available")

    assert store.find_near_duplicate(query, threshold=0.90, active_only=True) is None
