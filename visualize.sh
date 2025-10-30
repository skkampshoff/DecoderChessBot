#!/bin/bash
# Run a chess competition between two bots and visualize it.

# Exit immediately if a command fails
set -e

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/bot1 /path/to/bot2"
    exit 1
fi

BOT1="${1%/}"
BOT2="${2%/}"

# Run the competition and visualization
python -m competition_moderator "$BOT1" "$BOT2" | python visualizer.py "$BOT1" "$BOT2"
