#!/bin/bash
# One-time setup for Mac (and Linux).
# Run from inside the pumpkin-detector folder:   bash setup_mac.sh
set -e

echo "== 1/4 Checking Python =="
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Install it from https://www.python.org/downloads/ then run this again."
  exit 1
fi
python3 --version

echo "== 2/4 Creating virtual environment (venv) =="
python3 -m venv venv
source venv/bin/activate

echo "== 3/4 Installing packages (this can take 5-10 minutes) =="
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "== 4/4 Downloading AI models and testing everything =="
python smoke_test.py

echo ""
echo "All done! Next time, start with:   source venv/bin/activate"
