"""Lightweight, duck-typed stand-ins for testing strategy/plan1 modules
without a live engine channel.

`Vec2`, `BotClass`, and `MapTile` come from the real generated bindings --
they're plain ctypes/enum types that work standalone, no engine handshake
needed. Everything else here (bots, fleets, deposits, config, game state) is
a minimal object exposing only the attributes the algorithms under test
actually read; engine-channel functions (`point_free`, `line_of_sight`,
`point_seg_dist`, `navigate_to`, `path_length`) are not faked here -- tests
that need them monkeypatch the name directly in the module under test
(e.g. `strategy.plan1.combat_los.line_of_sight`), since that's the local
binding each module's `from .. import X` actually created.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core._generated.bindings import BotClass, MapTile, Vec2  # noqa: E402

__all__ = [
    "Vec2",
    "BotClass",
    "MapTile",
    "FakeBot",
    "FakeFleet",
    "FakeTeamPair",
    "FakeDeposit",
    "FakeFabricator",
    "FakeBotConfig",
    "FakeDepositConfig",
    "FakePayloadConfig",
    "FakeFabricatorConfig",
    "FakeConf",
    "FakeState",
    "FakeMap",
    "FakeMapConf",
    "FakeBudget",
]


class FakeBudget:
    """Stands in for `Budget` (the `get_budget()` NamedTuple)."""

    def __init__(self, remaining: int = 500_000, last_charge: int = 0):
        self.remaining = remaining
        self.last_charge = last_charge


class FakeBot:
    """Stands in for `BotState`: only the fields the algorithms read."""

    def __init__(self, id, pos, class_=BotClass.Battle, health=10.0,
                 invulnerable_until_tick=0, angle=0.0):
        self.id = id
        self.pos = pos
        self.class_ = class_
        self.health = health
        self.invulnerable_until_tick = invulnerable_until_tick
        self.angle = angle


class FakeFleet:
    """Stands in for `BotArray`: iterable, `.get(id)`, `.ids()`, `len()`."""

    def __init__(self, bots):
        self._bots = {b.id: b for b in bots}

    def get(self, bot_id):
        return self._bots.get(bot_id)

    def ids(self):
        return list(self._bots.keys())

    def __iter__(self):
        return iter(self._bots.values())

    def __len__(self):
        return len(self._bots)


class FakeTeamPair:
    def __init__(self, me: int = 0, other: int = 0):
        self.me = me
        self.other = other


class FakeDeposit:
    def __init__(self, pos: Vec2, extractors_me: int = 0, extractors_other: int = 0):
        self.pos = pos
        self.extractors = FakeTeamPair(extractors_me, extractors_other)


class FakeFabricator:
    def __init__(self, tokens: float = 0.0, next_bot_creation: int = 0):
        self.tokens = tokens
        self.next_bot_creation = next_bot_creation


class FakeBotConfig:
    def __init__(self, radius=0.25, blaster_range=10.0, base_blaster_splash_radius=0.3,
                 health=10.0, base_heal_range=3.0, base_heal_arc_deg=90.0,
                 blaster_cooldown=60, blaster_damage=3.0, base_invulnerability_ticks=15,
                 speed=0.05, turn_speed=3.0, heal_per_tick=0.05, extract_rate=0.05,
                 heal_stack_cap=3.0, base_extract_range=5.0):
        self.radius = radius
        self.blaster_range = blaster_range
        self.base_blaster_splash_radius = base_blaster_splash_radius
        self.health = health
        self.base_heal_range = base_heal_range
        self.base_heal_arc_deg = base_heal_arc_deg
        self.blaster_cooldown = blaster_cooldown
        self.blaster_damage = blaster_damage
        self.base_invulnerability_ticks = base_invulnerability_ticks
        self.speed = speed
        self.turn_speed = turn_speed
        self.heal_per_tick = heal_per_tick
        self.extract_rate = extract_rate
        self.heal_stack_cap = heal_stack_cap
        self.base_extract_range = base_extract_range


class FakeDepositConfig:
    def __init__(self, radius=0.5, extractor_cap=8):
        self.radius = radius
        self.extractor_cap = extractor_cap


class FakePayloadConfig:
    def __init__(self, radius=0.75, capture_radius=2.5, speed=0.02):
        self.radius = radius
        self.capture_radius = capture_radius
        self.speed = speed


class FakeFabricatorConfig:
    def __init__(self, interval=200, rush_cost=50.0, starting_tokens=800.0):
        self.interval = interval
        self.rush_cost = rush_cost
        self.starting_tokens = starting_tokens


class FakeConf:
    def __init__(self, bot=None, deposit=None, payload=None, fabricator=None,
                 max_ticks=9000, endgame_ticks=3000, map=None):
        self.bot = bot or FakeBotConfig()
        self.deposit = deposit or FakeDepositConfig()
        self.payload = payload or FakePayloadConfig()
        self.fabricator = fabricator or FakeFabricatorConfig()
        self.max_ticks = max_ticks
        self.endgame_ticks = endgame_ticks
        self.map = map if map is not None else FakeMap(set())


class FakeState:
    def __init__(self, tick=0, fleet_me=None, fleet_other=None,
                 deposit_me=None, deposit_other=None, payload=None,
                 fabricator_me=None):
        self.tick = tick
        self.fleet_me = FakeFleet(fleet_me or [])
        self.fleet_other = list(fleet_other or [])
        self.deposit_me = deposit_me or FakeDeposit(Vec2(2.0, 2.0))
        self.deposit_other = deposit_other or FakeDeposit(Vec2(30.0, 30.0))
        self._payload = payload if payload is not None else Vec2(16.0, 16.0)
        self.fabricator_me = fabricator_me or FakeFabricator()

    def payload_pos(self):
        return self._payload


class FakeMap:
    """Stands in for `conf.map`: indexable `m[x][y] -> MapTile`, matching
    the engine's `map[x][y]` layout, from a plain set of wall cells."""

    def __init__(self, wall_cells):
        self._wall_cells = set(wall_cells)

    def __getitem__(self, x):
        wall_cells = self._wall_cells

        class _Row:
            def __init__(self, x):
                self.x = x

            def __getitem__(self, y):
                return MapTile.Wall if (self.x, y) in wall_cells else MapTile.Empty

        return _Row(x)


class FakeMapConf:
    def __init__(self, wall_cells=()):
        self.map = FakeMap(wall_cells)
