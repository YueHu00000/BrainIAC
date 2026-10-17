#!/usr/bin/env bash
# Run with bash on Linux; mamba must already be installed.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="${1:-/mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/venv}"

# Creating an environment over an existing prefix can replace its contents.
if [[ -e "$env_prefix" || -L "$env_prefix" ]]; then
    echo "Environment path already exists: $env_prefix. Choose a new path." >&2
    exit 1
fi
env_prefix="$(realpath -m -- "$env_prefix")"

mamba create --yes --prefix "$env_prefix" --override-channels --channel conda-forge \
    python=3.10 'pip>=25' setuptools=80.9.0
mamba run --prefix "$env_prefix" python -m pip install \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
mamba run --prefix "$env_prefix" python -m pip install -r "$script_dir/requirements-python310.txt"
mamba run --prefix "$env_prefix" python -m pip check
mamba run --prefix "$env_prefix" python "$script_dir/check_environment.py"

printf '\nEnvironment ready. Activate in your current shell:\n  mamba activate %q\n' "$env_prefix"
