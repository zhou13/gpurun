#!/bin/bash

# This script is used by pre-commit to lint python files with pyright.
# It verifies that pyright has been installed into user's virtual environments
# and reminder the user to install if not.

pyright_version=$(grep "^pyright" requirements-dev.lock)

if command -v rye &>/dev/null && [ -d ".venv" ]; then
  rye run pyright "$@"
elif python3 -c "import pkg_resources; pkg_resources.require(\"$pyright_version\")" &>/dev/null; then
  pyright "$@"
else
  echo "It seems that pyright not installed into your virtual environment."
  echo "You should not see this if you are using rye to manager your venv."
  echo "If you are using rye, you can fix this by running:"
  echo "      rye sync"
  echo "If you are manageing you vnev manually, you can fix this by running:"
  echo "      python3 -m pip install $pyright_version"
  exit 1
fi
