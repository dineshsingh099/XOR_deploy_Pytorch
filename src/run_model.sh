#!/bin/sh

echo "Model training ... & Starting prediction..."
python3 /app/src/predict.py

# Keep the container running
tail -f /dev/null