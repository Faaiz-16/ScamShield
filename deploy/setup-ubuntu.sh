#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 server to run ScamShield with Docker.
#
#   sudo bash deploy/setup-ubuntu.sh
#
# Adds 2 GB of swap (small servers need it to build the image and run MySQL),
# installs Docker with the Compose plugin, and lets the current user run docker
# without sudo. Safe to run more than once.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this with sudo: sudo bash deploy/setup-ubuntu.sh" >&2
    exit 1
fi

. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
    echo "This script expects Ubuntu (found ${PRETTY_NAME:-unknown})." >&2
    exit 1
fi

if swapon --show --noheadings | grep -q .; then
    echo "Swap already enabled, skipping."
else
    echo "Creating a 2 GB swap file..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    # Use swap only when memory is really short.
    echo 'vm.swappiness=10' > /etc/sysctl.d/99-scamshield.conf
    sysctl -q -p /etc/sysctl.d/99-scamshield.conf
fi

echo "Installing Docker..."
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io docker-compose-v2
systemctl enable --now docker

target_user="${SUDO_USER:-}"
if [ -n "$target_user" ] && [ "$target_user" != "root" ]; then
    usermod -aG docker "$target_user"
fi

echo
echo "Done. Docker $(docker --version | cut -d' ' -f3 | tr -d ,) is installed."
echo "Log out and back in once, so you can run docker commands without sudo."
