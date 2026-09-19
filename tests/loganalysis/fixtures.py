"""Builds tiny synthetic `.mmgl` files for testing `tools/loganalysis`
without needing a real match log. Each helper appends one JSON-lines
record in the same delta encoding the engine actually emits (see
`tools/loganalysis/parser.py`'s module docstring).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List


def _bot(id_, x, y, health=10.0, special=None, vx=0.0, vy=0.0, angle=0.0):
    return {
        "id": id_,
        "health": health,
        "pos": {"x": x, "y": y},
        "vel": {"x": vx, "y": vy},
        "angle": angle,
        "turn_vel": 0.0,
        "invulnerable_until_tick": 0,
        "special": special or {"Battle": {"next_fire_tick": 0, "shot": "None"}},
    }


def write_log(path: str, records: List[Dict[str, Any]], footer_lines: List[str] = None) -> None:
    header = {
        "max_ticks": 9000,
        "endgame_ticks": 3000,
        "bot": {
            "radius": 0.25,
            "speed": 0.05,
            "health": 10.0,
            "turn_speed": 3.0,
            "blaster_cooldown": 60,
            "blaster_range": 10.0,
            "blaster_damage": 3.0,
            "heal_per_tick": 0.05,
            "extract_rate": 0.05,
            "base_invulnerability_ticks": 15,
            "base_blaster_splash_radius": 0.3,
            "base_heal_range": 3.0,
            "base_heal_arc_deg": 90.0,
            "heal_stack_cap": 3.0,
            "base_extract_range": 5.0,
        },
        "payload": {"radius": 0.75, "capture_radius": 2.5, "speed": 0.02},
        "payload_path": [{"x": 16.0, "y": 16.0}],
        "deposit": {"pos": {"x": 23.0, "y": 30.0}, "radius": 0.5, "extractor_cap": 8},
        "fabricator": {"interval": 200, "rush_cost": 50.0, "starting_tokens": 800.0},
        "map": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        for line in footer_lines or []:
            f.write(line + "\n")


def write_tempfile(records: List[Dict[str, Any]], footer_lines: List[str] = None) -> str:
    fd, path = tempfile.mkstemp(suffix=".mmgl")
    os.close(fd)
    write_log(path, records, footer_lines)
    return path


__all__ = ["_bot", "write_log", "write_tempfile"]
