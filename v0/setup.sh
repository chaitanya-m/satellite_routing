#!/usr/bin/env bash
set -e

# Install required tools
sudo apt-get update -y
sudo apt-get install -y curl bzip2

# Download micromamba (static binary)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj bin/micromamba

# Set micromamba root
export MAMBA_ROOT_PREFIX="$HOME/mamba"

# Create environment from environment.yml
./bin/micromamba create -y -f environment.yml

# Extract environment name
ENV_NAME=$(grep '^name:' environment.yml | awk '{print $2}')

# Activate it for this setup run
eval "$(./bin/micromamba shell hook -s bash)"
micromamba activate "$ENV_NAME"


# --- BEGIN micromamba PATH + init + auto-activation ---

# Ensure micromamba knows its root prefix
export MAMBA_ROOT_PREFIX="$HOME/mamba"
export MAMBA_EXE="$HOME/bin/micromamba"

# Put micromamba on PATH permanently
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
mkdir -p "$HOME/bin"
cp bin/micromamba "$HOME/bin/micromamba"

# Extract environment name
ENV_NAME=$(grep '^name:' environment.yml | awk '{print $2}')

# Initialize micromamba shell for future terminals
echo 'export MAMBA_ROOT_PREFIX="$HOME/mamba"' >> ~/.bashrc
echo 'export MAMBA_EXE="$HOME/bin/micromamba"' >> ~/.bashrc
echo 'eval "$($HOME/bin/micromamba shell hook -s bash)"' >> ~/.bashrc

# Auto-activate the environment in all new shells
echo "micromamba activate $ENV_NAME" >> ~/.bashrc

# Activate environment for this script run
eval "$($HOME/bin/micromamba shell hook -s bash)"
micromamba activate "$ENV_NAME"

# --- END micromamba PATH + init + auto-activation ---
