#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
w="$1"
while true; do
  clear
  echo "  WORKER ${w}   $(date +%H:%M:%S)   $(wc -l < "logs/${w}.log" 2>/dev/null || echo 0) events"
  echo "  ------------------------------------------------------------"
  tail -n 26 "logs/${w}.log" 2>/dev/null || echo "  waiting for worker ${w}..."
  sleep 2
done
