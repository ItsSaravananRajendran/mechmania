"""Offline analysis of `.mmgl` match replay logs.

`parser.py` reconstructs full per-tick game state from the log's delta
encoding; `analyze.py` turns a sequence of ticks into behavioral stats
(hit reactions, healer movement, mining deployment, formation shape,
payload contest, economy, engagements); `report.py` renders those as
text; `cli.py` is the `python -m tools.loganalysis <log...>` CLI entry
point; `mcp_server.py` exposes the same analyses as MCP tools (needs the
`mcp` package -- see `requirements.txt` -- and is registered for Claude
Code in `.mcp.json`) so a match's behavior can be queried interactively
instead of only read off a CLI dump.
"""
