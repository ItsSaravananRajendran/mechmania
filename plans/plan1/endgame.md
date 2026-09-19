# `endgame.py` — the self-destruct call

## Problem

During the endgame window, the fabricator is dead (no more builds, no more
rushes) and a fleet hitting zero bots is an instant loss. If the match is
going to end on the health tiebreaker anyway, a bot that's going to die for
nothing is worth more as a self-destruct: it denies the enemy the tempo/kill
value of finishing it off, at no cost since it wasn't going to survive
regardless.

## `_maybe_endgame_self_destruct(state, conf, action, cache) -> None`

Only called once endgame has started and only until it fires once (guarded
by `cache["endgame_sd_done"]` at the call site in `__init__.py` — this
function itself doesn't check that flag, it just sets it).

Logic: if the enemy has any HP left at all (`enemy_hp > 0`, guarding against
a pointless ratio check against a wiped enemy) and our total fleet HP has
fallen under 75% of theirs, self-destruct our weakest **Extractor** (lowest
current health) — not a Battle or Healer, since those still contribute to
holding the payload or fighting; an Extractor is the bot with the least
further use once combat has turned against us.

## Edge cases this handles

- **Enemy already wiped** (`enemy_hp == 0`): skipped — no ratio makes sense
  against zero, and there's no reason to sacrifice a bot when the fight is
  already won.
- **No extractors left**: `if extractors:` guards the whole self-destruct —
  does nothing rather than trying to sacrifice a Battle or Healer.
- **Called every tick while behind**: the condition itself (`my_hp <
  enemy_hp * 0.75`) would keep re-triggering every tick if not for the
  caller's one-shot guard; this function assumes that guard is in place and
  doesn't re-check it itself.
