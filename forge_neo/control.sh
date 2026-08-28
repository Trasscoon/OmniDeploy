#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

trap 'error_exit "### ERROR ###"' ERR

echo "### Command received ###"
file="/tmp/forge_neo.pid"

if [[ $1 == "reload" ]]; then
    log "Reloading Forge Neo"
    pkill -f "launch.py" 2>/dev/null || true
    sleep 1
    bash main.sh

elif [[ $1 == "start" ]]; then
    log "Starting Forge Neo"
    bash main.sh

elif [[ $1 == "stop" ]]; then
    log "Stopping Forge Neo"
    # Kill all Forge Neo launch.py instances (not just the PID)
    pkill -f "launch.py" 2>/dev/null || true
    # Also kill any service_loop running main.sh
    pkill -f "main.sh" 2>/dev/null || true
    rm -f /tmp/forge_neo.pid

else
    echo "Invalid argument"
fi

echo "### Done ###"
