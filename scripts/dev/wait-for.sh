#!/usr/bin/env bash
# wait-for host:port -t timeout
set -e
hostport="$1"; shift
timeout="${1:-60}"
host="${hostport%:*}"
port="${hostport#*:}"

end=$((SECONDS+timeout))
while ! nc -z "$host" "$port"; do
  if [ $SECONDS -ge $end ]; then
    echo "Timeout waiting for $host:$port"
    exit 1
  fi
  sleep 1
done
