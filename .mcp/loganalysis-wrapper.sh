#!/bin/bash
# Launches the mm-loganalysis MCP server (see tools/loganalysis/mcp_server.py)
# from the repo's own venv, regardless of where the repo is cloned or what
# directory Claude Code happens to launch this from -- Claude Code's
# .mcp.json doesn't honor a "cwd" field or path variables, so this script
# resolves everything relative to its own location instead.
cd "$(dirname "$0")/.." || exit 1

if [ ! -x .venv/bin/python ]; then
    echo "mm-loganalysis MCP server: no .venv found at repo root." >&2
    echo "Set it up once with:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r tools/loganalysis/requirements.txt" >&2
    exit 1
fi

exec .venv/bin/python -m tools.loganalysis.mcp_server
