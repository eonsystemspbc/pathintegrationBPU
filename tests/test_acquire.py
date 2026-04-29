from __future__ import annotations

from src.acquire import resolve_cx_primary_rois


def test_resolve_cx_primary_rois_accepts_marked_primary_names() -> None:
    hierarchy = {
        "hemibrain": {
            "CX": {
                "AB(L)*": {},
                "AB(R)*": {},
                "EB*": {},
                "FB*": {},
                "NO*": {},
                "PB*": {},
            }
        }
    }
    assert resolve_cx_primary_rois(hierarchy) == ("EB", "PB", "FB", "NO")

