#!/usr/bin/env bash
# Pull the latest images for the services in docker-compose.yml
# and recreate the containers whose images have changed.
set -euo pipefail

if [ "${EUID}" -ne 0 ] && ! id -nG | grep -qw docker; then
    echo "You are not in the docker group. Please run this script with sudo." >&2
    exit 1
fi

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Download new images. The running containers keep running during this step.
echo "Pulling updated images."
docker compose pull

# Containers whose image and config are unchanged are left running.
# Containers with a new image or changed config are stopped, removed and recreated,
# which causes a brief downtime for that service and interrupts any active streams.
# Volumes and bind mounts are preserved.
# Stopped containers are started.
# --remove-orphans removes containers of services no longer in docker-compose.yml.
echo "Restarting containers with updated images."
docker compose up -d --remove-orphans

# Remove only dangling images that no container uses, so running services are unaffected.
echo "Removing old dangling images."
docker image prune -f

echo "Update complete."
