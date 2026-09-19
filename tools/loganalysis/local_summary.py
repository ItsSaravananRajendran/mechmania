"""Quick summary of a single .mmgl log: healer counts over time, payload net,
hits, deaths, builds. Used by the iterative H1-H7 verification workflow.

Usage:
    .venv/bin/python tools/loganalysis/local_summary.py <logfile> [logfile ...]

Prints a per-log summary including healer count peak/average/end and whether
healers stayed alive past tick 1000 (the bug check).
"""
from __future__ import annotations
import sys
import json
import math
from collections import Counter

sys.path.insert(0, '/home/thunderbolt/gitRepos/mm')

from tools.loganalysis.parser import load_match_result, load_config, iter_ticks


def summarise(path: str) -> dict:
    config = load_config(path)
    ticks = list(iter_ticks(path))
    result = load_match_result(path)

    # Healer count timeline
    h_timeline = []
    e_timeline = []
    for t in ticks[::100]:
        ha = sum(1 for b in t.bots('a') if b.role == 'Healer')
        hb = sum(1 for b in t.bots('b') if b.role == 'Healer')
        ea = sum(1 for b in t.bots('a') if b.role == 'Extractor')
        eb = sum(1 for b in t.bots('b') if b.role == 'Extractor')
        h_timeline.append((t.tick, ha, hb))
        e_timeline.append((t.tick, ea, eb))

    # Hits
    prev_hp_a, prev_hp_b = {}, {}
    hits_a, hits_b = 0, 0
    hits_a_on_h, hits_b_on_h = 0, 0
    for t in ticks:
        cur_a = {b.id: b.health for b in t.bots('a')}
        cur_b = {b.id: b.health for b in t.bots('b')}
        for bid, hp in cur_b.items():
            if bid in prev_hp_b and 2.7 < prev_hp_b[bid] - hp < 3.3:
                hits_a += 1
                if next(iter(t.fleets['b'][bid].special)) == 'Healer':
                    hits_a_on_h += 1
        for bid, hp in cur_a.items():
            if bid in prev_hp_a and 2.7 < prev_hp_a[bid] - hp < 3.3:
                hits_b += 1
                if next(iter(t.fleets['a'][bid].special)) == 'Healer':
                    hits_b_on_h += 1
        prev_hp_a, prev_hp_b = cur_a, cur_b

    # First healer death per side
    first_h_death_a = first_h_death_b = None
    for i, t in enumerate(ticks):
        if i == 0:
            continue
        prev = ticks[i-1]
        for bid in t.removed.get('a', []):
            if bid in prev.fleets['a'] and prev.fleets['a'][bid].role == 'Healer':
                first_h_death_a = t.tick
        for bid in t.removed.get('b', []):
            if bid in prev.fleets['b'] and prev.fleets['b'][bid].role == 'Healer':
                first_h_death_b = t.tick

    # Healer ever-built counts (unique bot ids per side)
    h_built_a = h_built_b = 0
    seen_a, seen_b = set(), set()
    for t in ticks:
        for bid in t.added.get('a', []):
            bot = t.fleets['a'].get(bid)
            if bot and bot.role == 'Healer':
                seen_a.add(bid)
        for bid in t.added.get('b', []):
            bot = t.fleets['b'].get(bid)
            if bot and bot.role == 'Healer':
                seen_b.add(bid)
    h_built_a = len(seen_a)
    h_built_b = len(seen_b)

    # Final fleet sizes
    final_a = len(ticks[-1].bots('a'))
    final_b = len(ticks[-1].bots('b'))

    return {
        'path': path,
        'winner': result.winner if result else None,
        'reason': result.reason if result else None,
        'length': len(ticks),
        'final_a': final_a, 'final_b': final_b,
        'net_capture': ticks[-1].capture - ticks[0].capture,
        'h_peak_a': max(h for _, h, _ in h_timeline),
        'h_peak_b': max(h for _, _, h in h_timeline),
        'h_avg_a': sum(h for _, h, _ in h_timeline) / len(h_timeline),
        'h_avg_b': sum(h for _, _, h in h_timeline) / len(h_timeline),
        'h_end_a': h_timeline[-1][1],
        'h_end_b': h_timeline[-1][2],
        'h_at_1000_a': next((h for tk, h, _ in h_timeline if tk >= 1000), 0),
        'h_at_1000_b': next((h for tk, _, h in h_timeline if tk >= 1000), 0),
        'h_built_a': h_built_a, 'h_built_b': h_built_b,
        'first_h_death_a': first_h_death_a, 'first_h_death_b': first_h_death_b,
        'hits_a': hits_a, 'hits_b': hits_b,
        'hits_a_on_h': hits_a_on_h, 'hits_b_on_h': hits_b_on_h,
    }


def print_one(s: dict) -> None:
    name = s['path'].split('/')[-1]
    print(f"\n=== {name} ===")
    print(f"  result: winner={s['winner']} ({s['reason']})  len={s['length']}")
    print(f"  final fleet A={s['final_a']}  B={s['final_b']}  net_capture={s['net_capture']:+.3f}")
    print(f"  healers A: peak={s['h_peak_a']}  avg={s['h_avg_a']:.2f}  "
          f"end={s['h_end_a']}  @tick1000={s['h_at_1000_a']}  ever_built={s['h_built_a']}  "
          f"first_death={'never' if s['first_h_death_a'] is None else s['first_h_death_a']}")
    print(f"  healers B: peak={s['h_peak_b']}  avg={s['h_avg_b']:.2f}  "
          f"end={s['h_end_b']}  @tick1000={s['h_at_1000_b']}  ever_built={s['h_built_b']}  "
          f"first_death={'never' if s['first_h_death_b'] is None else s['first_h_death_b']}")
    print(f"  hits: A landed {s['hits_a']} (on healers: {s['hits_a_on_h']})  "
          f"B landed {s['hits_b']} (on healers: {s['hits_b_on_h']})")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print_one(summarise(p))
