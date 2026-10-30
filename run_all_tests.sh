#!/usr/bin/env bash
# Each task has its own module names (gateway.py exists twice), so run pytest per task directory.
set -e
for t in task1_mcp_server task2_mcp_gateway task3_pii_stream task4_router; do
  echo "=== $t ==="; (cd "$t" && python -m pytest -q)
done
