# RL improvement plan

Companion to `rl/README.md`: that file documents the pipeline as it exists (env, encoding,
rewards, opponents). This file documents *why* it's being changed and what "done" looks like,
so it stays a useful reference after the changes land instead of just a commit message.

## Context

The repo already has a working single-opponent PPO pipeline (`rl/train_ppo.py`) and a self-play
variant (`rl/train_self_play.py`), with one completed run (`rl/runs/plan1-tactics2-30k/`, ~30k
steps vs `plan1`). The game is a two-player zero-sum contest over a shared payload, with three
heterogeneous roles (extractor/miner, healer, attacker/battle) under one joint team policy per
side. That shape is closest to StarCraft II / capture-the-flag RL problems, so the reference
techniques are AlphaStar-style league play, OpenAI Five/CTF-style reward shaping, and standard
self-play robustness fixes — scaled down to what's practical on a laptop.

## Why self-play, and why the current self-play code isn't enough yet

A policy trained only against a fixed scripted opponent (e.g. `plan1`) will overfit to that
opponent's specific weaknesses and collapse against anything else — well-known in adversarial
RL, and the exact risk with the existing 30k-step `plan1`-only run. "Best possible" in a 2-player
zero-sum game means robust to a wide range of strategies, which requires training against a
*population* of opponents that itself improves over time, not a single fixed target.

`rl/train_self_play.py` already has the right skeleton (two PPO workers, snapshot exchange over
files), but each side keeps only its single latest snapshot, overwritten every
`--snapshot-every` steps. That's not actually a population, and it's exactly the setup that
produces **strategy cycling**: agent A learns to beat B's current style, B adapts and starts
beating that, A forgets how to beat B's earlier style, and the pair oscillates without net
progress — self-play reward keeps looking fine while real skill against a fixed reference
(`plan1..plan5`) can stay flat or regress.

## Opponent pool design

- Keep a directory of historical checkpoints per side instead of overwriting one file.
- Per episode, sample the opponent as ~70% the most-recent snapshot (keeps training on-policy
  and fast-improving) and ~30% a uniform-random older snapshot (prevents forgetting/cycling).
- Mix in the scripted `plan1..plan5` bots at a configurable rate throughout training — they're
  a free source of *stylistically diverse, non-drifting* opponents (rush, deathball, turtle,
  reactive, kiter — five distinct archetypes, not a difficulty ladder) that self-play alone may
  never discover on its own.
- This is a lightweight version of AlphaStar's "league" (current agents + past selves + fixed
  exploiters) without the full PFSP matchmaking math — uniform sampling over
  {recent snapshots, older snapshots, scripted bots} already fixes most of the cycling problem.

## Reward changes

- `tactics_reward` shapes extractor mining and battle-bot shooting/positioning individually but
  has no heal-usage term, so PPO gets no direct gradient toward learning to heal — only the very
  weak, delayed signal of terminal HP differential. Add a small dense term for landing heals on
  damaged allies, mirroring the existing extractor/blaster terms.
- Anneal all non-terminal shaping terms (everything except the terminal ±1) linearly to 0 over
  the training budget. Shaping exists to solve credit assignment early on; left permanent, it
  risks the agent optimizing the shaping proxy (e.g. endless mining/healing) instead of the real
  win condition. The terminal ±1 always stays fixed.

## Evaluation changes

- `evaluate.py` reports solid single-model-vs-single-opponent stats but has no cross-checkpoint
  tracking. Add a round-robin mode: every saved checkpoint vs. every `plan1..plan5`, reported as
  win-rate-over-checkpoint-index, so a real skill curve is visible instead of just "loss went
  down" or one checkpoint's snapshot stats.
- Watch specifically for **non-monotonic win rate against `plan1..plan5` while self-play reward
  keeps climbing** — that's the signature of strategy cycling / overfitting to the current pool,
  and the strongest signal the opponent pool needs to be bigger or more diverse.
- Fix the `max_ticks` mismatch between `train_self_play.py` (default 80) and `evaluate.py`
  (default 200) — training and eval should use the same episode horizon, or a policy is being
  evaluated on a different effective task than it trained on.

## What's explicitly out of scope for this pass

- `rl/encoding.py`'s observation encoding (map-grid + per-bot-list) already fits the game's
  continuous-position, variable-fleet-size structure — no CNN-grid rewrite.
- `rl/encoding.py`'s action space (fixed-size composite continuous vector per bot slot) already
  functions as a "Commander"-style joint action — no auto-regressive or MultiDiscrete rewrite.
- Per-role/multi-agent policies (separate networks per bot class, PettingZoo/MAPPO) — revisit
  only if the single joint policy plateaus after the changes below.
- Engine-level class-restricted curriculum (e.g. "1 attacker only" early phases) — plausible, but
  depends on unconfirmed engine/fabricator support; not attempted until the changes below show
  results.

## Build order

1. **`rl/rewards.py`** — add a healer-specific dense reward term; add a shaping-decay schedule
   that anneals all non-terminal reward components to 0 over a configurable step budget.
2. **`rl/train_ppo.py` / `rl/train_self_play.py`** — align the `--max-ticks` default with
   `rl/evaluate.py`.
3. **`rl/train_self_play.py`** — replace the single mutable snapshot per side with a directory
   of historical checkpoints; sample the opponent per episode (~70% latest / ~30% older
   snapshot, plus a configurable rate of substituting a `plan1..plan5` scripted bot).
4. **`rl/evaluate.py`** — add a round-robin mode across a directory of checkpoints vs.
   `plan1..plan5`, output as a win-rate-over-checkpoint table/CSV.

Each step lands independently with a quick sanity check (a short training/eval smoke run)
before moving to the next, rather than batching all four and debugging at the end.
