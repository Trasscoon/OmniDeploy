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
    kill_pid $file
    sleep 1
    bash main.sh

elif [[ $1 == "start" ]]; then
    log "Starting Forge Neo"
    bash main.sh

elif [[ $1 == "stop" ]]; then
    log "Stopping Forge Neo"
    kill_pid $file

else
    echo "Invalid argument"
fi

echo "### Done ###"
