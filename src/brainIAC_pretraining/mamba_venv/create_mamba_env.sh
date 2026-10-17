#!/usr/bin/env bash
# Run with bash on Linux; mamba must already be installed.
set -euo pipefail

env_name="${1:-brainiac_pretrain_py310}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Creating an environment over an existing prefix can replace its contents.
environments="$(mamba env list)"
if printf '%s\n' "$environments" | awk -v name="$env_name" '$1 == name {found=1} END {exit !found}'; then
    echo "Environment already exists: $env_name. Choose a new name." >&2
    exit 1
fi

mamba create --yes --name "$env_name" --override-channels --channel conda-forge \
    python=3.10 'pip>=25' setuptools=80.9.0
mamba run --name "$env_name" python -m pip install \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
mamba run --name "$env_name" python -m pip install -r "$script_dir/requirements-python310.txt"
mamba run --name "$env_name" python -m pip check
mamba run --name "$env_name" python "$script_dir/check_environment.py"

printf '\nEnvironment ready. Activate in your current shell:\n  mamba activate %s\n' "$env_name"
