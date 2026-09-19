"""Parser for `.mmgl` replay logs (see `logs/*.mmgl`, produced by the
`mm` engine when a match is run).

The file is JSON Lines:

- line 1 is the match config (map, bot/payload/deposit/fabricator stats,
  `payload_path`) -- never a tick.
- every following line is one game tick. A key that's missing from a tick's
  record means "unchanged since the previous tick", including whole
  sub-objects (`fabricator_a`, `deposit_b`, ...). `fleet_a`/`fleet_b` is
  special: on the very first tick it's a full list of bots, and on every
  tick after that it's a delta dict with any of `added` (id -> full bot),
  `changed` (id -> partial fields to merge in) and `removed` (list of ids).
- the file may end with one or two `#`-prefixed comment lines (`# result:
  ...`, `# time elapsed: ...`) that aren't JSON and aren't ticks.

This module replays those deltas so callers work with a plain, fully
resolved snapshot at every tick instead of thinking about the encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

Team = str  # "a" or "b"

# The `special` dict's one key names the bot's class -- these are the three
# values the engine ever puts there.
ROLE_EXTRACTOR = "Extractor"
ROLE_HEALER = "Healer"
ROLE_BATTLE = "Battle"


@dataclass(frozen=True)
class BotSnap:
    """One bot's full state as of a given tick."""

    id: int
    health: float
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    turn_vel: float
    invulnerable_until_tick: int
    special: Dict[str, Any]

    @property
    def role(self) -> str:
        # `special` always has exactly one key: the bot's class.
        return next(iter(self.special), "")

    @property
    def pos(self) -> "tuple[float, float]":
        return (self.x, self.y)


@dataclass(frozen=True)
class DepositSnap:
    x: float
    y: float
    extractors_a: int
    extractors_b: int


@dataclass(frozen=True)
class FabricatorSnap:
    tokens: float
    next_bot_creation: int


@dataclass(frozen=True)
class TickSnapshot:
    """Fully resolved game state at one tick -- what every log line would
    say if the log weren't delta-encoded."""

    tick: int
    capture: float
    fleets: Dict[Team, Dict[int, BotSnap]]
    deposits: Dict[Team, Optional[DepositSnap]]
    fabricators: Dict[Team, Optional[FabricatorSnap]]
    # Bot ids removed from each fleet *at this tick* -- the engine deletes a
    # dead bot outright rather than ever emitting a final `health: 0`, so
    # this is the only way to tell "died this tick" from "log ended".
    removed: Dict[Team, List[int]]
    # Bot ids newly built into each fleet *at this tick*. The engine reuses
    # small ids once they free up, so this (paired with `removed`) is what
    # lets callers tell two different bots that happened to share an id
    # apart instead of splicing their histories together.
    added: Dict[Team, List[int]]

    def bots(self, team: Team) -> List[BotSnap]:
        return list(self.fleets.get(team, {}).values())

    def bots_by_role(self, team: Team, role: str) -> List[BotSnap]:
        return [b for b in self.bots(team) if b.role == role]


@dataclass
class MatchConfig:
    max_ticks: int
    endgame_ticks: int
    bot: Dict[str, Any]
    payload: Dict[str, Any]
    payload_path: List[Dict[str, float]]
    deposit: Dict[str, Any]
    fabricator: Dict[str, Any]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    """The engine's own verdict, read from the trailing `#`-prefixed
    comment lines (`# result: {...}`, `# time elapsed: ...s`) rather than
    reconstructed from tick data -- those are the only two lines in the
    file that aren't JSON ticks."""

    reason: str  # "elimination" | "payload" | "tie" | ...
    winner: Optional[str]  # "A" / "B" / None for a tie
    tick: int
    elapsed_seconds: Optional[float]


def _bot_from_dict(d: Dict[str, Any]) -> BotSnap:
    return BotSnap(
        id=d["id"],
        health=d["health"],
        x=d["pos"]["x"],
        y=d["pos"]["y"],
        vx=d["vel"]["x"],
        vy=d["vel"]["y"],
        angle=d["angle"],
        turn_vel=d["turn_vel"],
        invulnerable_until_tick=d["invulnerable_until_tick"],
        special=d["special"],
    )


def _merge_bot(old: BotSnap, changes: Dict[str, Any]) -> BotSnap:
    return BotSnap(
        id=old.id,
        health=changes.get("health", old.health),
        x=changes.get("pos", {}).get("x", old.x) if "pos" in changes else old.x,
        y=changes.get("pos", {}).get("y", old.y) if "pos" in changes else old.y,
        vx=changes.get("vel", {}).get("x", old.vx) if "vel" in changes else old.vx,
        vy=changes.get("vel", {}).get("y", old.vy) if "vel" in changes else old.vy,
        angle=changes.get("angle", old.angle),
        turn_vel=changes.get("turn_vel", old.turn_vel),
        invulnerable_until_tick=changes.get(
            "invulnerable_until_tick", old.invulnerable_until_tick
        ),
        special=changes.get("special", old.special),
    )


def _deposit_from_dict(
    old: Optional[DepositSnap], d: Dict[str, Any]
) -> DepositSnap:
    pos = d.get("pos")
    x = pos["x"] if pos else (old.x if old else 0.0)
    y = pos["y"] if pos else (old.y if old else 0.0)
    extractors = d.get("extractors")
    ea = extractors["a"] if extractors else (old.extractors_a if old else 0)
    eb = extractors["b"] if extractors else (old.extractors_b if old else 0)
    return DepositSnap(x=x, y=y, extractors_a=ea, extractors_b=eb)


def _fabricator_from_dict(
    old: Optional[FabricatorSnap], d: Dict[str, Any]
) -> FabricatorSnap:
    tokens = d.get("tokens", old.tokens if old else 0.0)
    next_bot_creation = d.get(
        "next_bot_creation", old.next_bot_creation if old else 0
    )
    return FabricatorSnap(tokens=tokens, next_bot_creation=next_bot_creation)


def parse_config(first_line: str) -> MatchConfig:
    raw = json.loads(first_line)
    return MatchConfig(
        max_ticks=raw["max_ticks"],
        endgame_ticks=raw["endgame_ticks"],
        bot=raw["bot"],
        payload=raw["payload"],
        payload_path=raw["payload_path"],
        deposit=raw["deposit"],
        fabricator=raw["fabricator"],
        raw=raw,
    )


def iter_ticks(path: str) -> Iterator[TickSnapshot]:
    """Yield a fully resolved `TickSnapshot` for every tick in `path`, in
    order. Each snapshot is an independent, immutable object -- safe to
    hold on to across iterations (later ticks never mutate an earlier
    snapshot's bots)."""

    fleets: Dict[Team, Dict[int, BotSnap]] = {"a": {}, "b": {}}
    deposits: Dict[Team, Optional[DepositSnap]] = {"a": None, "b": None}
    fabricators: Dict[Team, Optional[FabricatorSnap]] = {"a": None, "b": None}
    capture = 0.0

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        parse_config(first)  # validates the header; callers use `load_config`

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            tick = rec["tick"]
            capture = rec.get("capture", capture)
            removed_this_tick: Dict[Team, List[int]] = {"a": [], "b": []}
            added_this_tick: Dict[Team, List[int]] = {"a": [], "b": []}

            for team in ("a", "b"):
                fkey = f"fleet_{team}"
                if fkey in rec:
                    fleet_field = rec[fkey]
                    live = fleets[team]
                    if isinstance(fleet_field, list):
                        # Only the very first fleet record in the file is a
                        # full list rather than a delta.
                        live.clear()
                        for bd in fleet_field:
                            b = _bot_from_dict(bd)
                            live[b.id] = b
                            added_this_tick[team].append(b.id)
                    else:
                        for bd in fleet_field.get("added", {}).values():
                            b = _bot_from_dict(bd)
                            live[b.id] = b
                            added_this_tick[team].append(b.id)
                        for bid_str, changes in fleet_field.get("changed", {}).items():
                            bid = int(bid_str)
                            if bid in live:
                                live[bid] = _merge_bot(live[bid], changes)
                        for bid in fleet_field.get("removed", []):
                            live.pop(bid, None)
                            removed_this_tick[team].append(bid)

                dkey = f"deposit_{team}"
                if dkey in rec:
                    deposits[team] = _deposit_from_dict(deposits[team], rec[dkey])
                fkey2 = f"fabricator_{team}"
                if fkey2 in rec:
                    fabricators[team] = _fabricator_from_dict(
                        fabricators[team], rec[fkey2]
                    )

            yield TickSnapshot(
                tick=tick,
                capture=capture,
                fleets={"a": dict(fleets["a"]), "b": dict(fleets["b"])},
                deposits=dict(deposits),
                fabricators=dict(fabricators),
                removed=removed_this_tick,
                added=added_this_tick,
            )


def load_config(path: str) -> MatchConfig:
    with open(path, "r", encoding="utf-8") as f:
        return parse_config(f.readline())


_RESULT_PREFIX = "# result:"
_ELAPSED_PREFIX = "# time elapsed:"


def load_match_result(path: str) -> Optional[MatchResult]:
    """Read the trailing `# result: {...}` / `# time elapsed: ...` comment
    lines. Returns `None` if the log doesn't have one (e.g. it was cut off
    mid-match, or the engine crashed before writing it)."""

    reason = winner = tick = None
    elapsed: Optional[float] = None
    # These are the last one or two lines of a (potentially huge) file --
    # read backwards a bounded number of lines rather than the whole file.
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 4096)
        f.seek(size - chunk)
        tail = f.read(chunk).decode("utf-8", errors="replace")

    for line in tail.splitlines():
        line = line.strip()
        if line.startswith(_RESULT_PREFIX):
            payload = json.loads(line[len(_RESULT_PREFIX):].strip())
            reason = payload.get("reason")
            winner = payload.get("winner")
            tick = payload.get("tick")
        elif line.startswith(_ELAPSED_PREFIX):
            raw = line[len(_ELAPSED_PREFIX):].strip()
            if raw.endswith("s"):
                raw = raw[:-1]
            try:
                elapsed = float(raw)
            except ValueError:
                elapsed = None

    if reason is None:
        return None
    return MatchResult(reason=reason, winner=winner, tick=tick, elapsed_seconds=elapsed)
