"""Renders a `MatchReport` (see `analyze.py`) as human-readable text."""

from __future__ import annotations

from .analyze import MatchReport


def _fmt_pos(p) -> str:
    return f"({p[0]:.1f}, {p[1]:.1f})"


def render_text(report: MatchReport) -> str:
    lines: list[str] = []
    lines.append(f"== {report.log_path} -- team {report.team} ({report.ticks_analyzed} ticks) ==")

    if report.match_result:
        mr = report.match_result
        winner = mr.winner or "nobody (tie)"
        lines.append(
            f"result: {winner} won by {mr.reason} at tick {mr.tick}"
            + (f" ({mr.elapsed_seconds:.1f}s real time)" if mr.elapsed_seconds is not None else "")
        )

    pc = report.payload_contest
    lines.append("")
    lines.append("-- Payload contest --")
    lines.append(
        f"  capture: {pc.initial_capture:.3f} -> {pc.final_capture:.3f} "
        f"(net {pc.net_change:+.3f}, leader at end: {pc.leader_at_end})"
    )
    lines.append(
        f"  ticks pushing toward A win: {pc.ticks_pushed_toward_a_win}, "
        f"toward B win: {pc.ticks_pushed_toward_b_win}, "
        f"stalled/contested: {pc.ticks_stalled} ({pc.pct_stalled:.0f}%)"
    )

    hr = report.hit_reactions
    lines.append("")
    lines.append(f"-- Hit reactions ({hr.total_hits} hits taken) --")
    if hr.total_hits:
        lines.append(f"  deaths: {hr.deaths}")
        lines.append(
            f"  fled after being hit: {hr.fled_count}  |  held ground: {hr.held_ground_count}"
        )
        lines.append(
            f"  reached by a healer within window: {hr.healed_within_window_count}"
            + (
                f" (avg {hr.avg_ticks_to_heal:.1f} ticks)"
                if hr.avg_ticks_to_heal is not None
                else ""
            )
        )
        lines.append(f"  hits by role: {hr.by_role}")
    else:
        lines.append("  (no hits recorded)")

    eg = report.engagements
    lines.append("")
    lines.append(f"-- Engagements ({eg.count} distinct fights) --")
    if eg.count:
        lines.append(
            f"  avg duration {eg.avg_duration_ticks:.0f} ticks, avg {eg.avg_hits:.1f} hits, "
            f"avg {eg.avg_deaths:.1f} deaths per fight"
        )
        d = eg.deadliest
        lines.append(
            f"  deadliest: ticks {d.start_tick}-{d.end_tick}, {d.deaths} deaths, "
            f"{d.bots_hit} bots hit, fled {d.fled_count} / held {d.held_ground_count}"
        )
    else:
        lines.append("  (no engagements)")

    eco = report.economy
    lines.append("")
    lines.append("-- Economy --")
    lines.append(
        f"  tokens: avg {eco.token_avg:.0f}, max {eco.token_max:.0f}, end {eco.token_end:.0f}"
    )
    lines.append(f"  builds by role: {eco.build_counts_by_role} ({len(eco.builds)} total)")
    for dteam in ("a", "b"):
        lines.append(
            f"  deposit {dteam} slot occupancy: avg {100 * eco.deposit_occupancy_avg[dteam]:.0f}%, "
            f"max {100 * eco.deposit_occupancy_max[dteam]:.0f}%"
        )

    hm = report.healer_movement
    lines.append("")
    lines.append(f"-- Healer movement ({hm.healer_count} healers) --")
    for p in hm.profiles:
        lines.append(
            f"  healer #{p.bot_id}: home {_fmt_pos(p.home_pos)}, "
            f"wander radius {p.wander_radius:.2f}, avg speed {p.avg_speed:.3f} "
            f"(max {p.max_speed:.3f}), had a target {p.pct_ticks_with_target:.0f}% of the time, "
            f"healed {p.unique_targets_healed} distinct allies"
        )
    if not hm.profiles:
        lines.append("  (no healers)")

    md = report.mining_deployment
    lines.append("")
    lines.append("-- Mining deployment / guard coverage --")
    for dteam in ("a", "b"):
        lines.append(
            f"  deposit {dteam}: avg miners {md.avg_miners[dteam]:.1f} "
            f"(max {md.max_miners[dteam]}), avg guards {md.avg_guards[dteam]:.1f} "
            f"(max {md.max_guards[dteam]})"
        )

    for label, fs in (
        ("Battle formation", report.battle_formation),
        ("Healer formation", report.healer_formation),
    ):
        lines.append("")
        lines.append(f"-- {label} shape (dominant: {fs.dominant_shape}) --")
        shape_counts: dict[str, int] = {}
        for s in fs.samples:
            shape_counts[s.shape] = shape_counts.get(s.shape, 0) + 1
        lines.append(f"  shape distribution over {len(fs.samples)} samples: {shape_counts}")
        if fs.samples:
            last = fs.samples[-1]
            lines.append(
                f"  last sample (tick {last.tick}): {last.count} bots, "
                f"centroid {_fmt_pos(last.centroid)}, spread {last.spread_radius:.2f}, "
                f"aspect ratio {last.aspect_ratio:.2f}, shape '{last.shape}'"
            )

    return "\n".join(lines)
