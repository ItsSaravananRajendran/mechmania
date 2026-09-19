"""CLI: analyze one or more `.mmgl` replay logs.

    python -m tools.loganalysis logs/log.mmgl
    python -m tools.loganalysis logs/log.mmgl --team b --json
    python -m tools.loganalysis logs/*.mmgl --team b   # aggregate summary

By default this reports on team "b" (the opponent, in the convention this
repo's `strategy/` package already uses: our bot is fleet "a"). Pass
`--team a` to turn the same analysis on our own fleet instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List

from .analyze import MatchReport, build_report
from .parser import iter_ticks, load_config, load_match_result
from .report import render_text


def _analyze_one(path: str, team: str, sample_every: int) -> MatchReport:
    config = load_config(path)
    guard_range = config.bot.get("blaster_range", 10.0)
    extract_range = config.bot.get("base_extract_range", 5.0)
    extractor_cap = config.deposit.get("extractor_cap", 8)
    ticks = list(iter_ticks(path))
    return build_report(
        log_path=path,
        ticks=ticks,
        team=team,
        extract_range=extract_range,
        guard_range=guard_range,
        extractor_cap=extractor_cap,
        sample_every=sample_every,
        match_result=load_match_result(path),
    )


def _report_to_jsonable(report: MatchReport) -> dict:
    return dataclasses.asdict(report)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="one or more .mmgl log files")
    parser.add_argument(
        "--team",
        choices=["a", "b"],
        default="b",
        help="which fleet to analyze (default: b, the usual opponent slot)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=20,
        help="tick interval for formation/mining snapshots (default: 20)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full report as JSON instead of text"
    )
    args = parser.parse_args(argv)

    reports = [_analyze_one(p, args.team, args.sample_every) for p in args.logs]

    if args.json:
        payload = [_report_to_jsonable(r) for r in reports]
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2))
        return 0

    for r in reports:
        print(render_text(r))
        print()

    if len(reports) > 1:
        _print_aggregate(reports)

    return 0


def _print_aggregate(reports: List[MatchReport]) -> None:
    print(f"== Aggregate across {len(reports)} logs ==")
    total_hits = sum(r.hit_reactions.total_hits for r in reports)
    total_deaths = sum(r.hit_reactions.deaths for r in reports)
    total_fled = sum(r.hit_reactions.fled_count for r in reports)
    total_healed = sum(r.hit_reactions.healed_within_window_count for r in reports)
    print(f"  hits: {total_hits}, deaths: {total_deaths}, fled: {total_fled}, healed-in-window: {total_healed}")

    shapes: dict[str, int] = {}
    for r in reports:
        shapes[r.battle_formation.dominant_shape] = shapes.get(r.battle_formation.dominant_shape, 0) + 1
    print(f"  dominant battle formation shape across matches: {shapes}")


if __name__ == "__main__":
    sys.exit(main())
