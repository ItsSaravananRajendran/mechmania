#!/bin/sh
# Wrapper so `mm-engine` can spawn `rl/relay_bot.py` as one of its two bot slots.
#
# The engine takes the path to an executable and runs it as
#   <BOT_PATH> <SHMEM_PATH>
# A Python script with a shebang would work too, but the engine's spawn path is
# flakier on some filesystems when the bot binary's path has a `.py` extension -- a
# shell wrapper that exec's the interpreter is the path that has actually been run
# against the engine in this repo.

exec python3 "$(dirname "$0")/relay_bot.py" "$@"
