#!/bin/bash

DATA_ROOT="data"
MODEL_DIR="./models"
CSV_FILE="multidomain_all_results.csv"
SCRIPT="test.py"

# Header
echo "domain,percent,dice,iou,f1,recall,vi" > $CSV_FILE

DOMAINS=(
  "c-elegans-dauer-stage"
  "dhanyasi-P14-mouse-cerebellum"
  "fly"
  "katz-lab-berghia-connective"
  "micron"
  "octopus-vulgaris-vertical-lobe-glia-deep-neuropil"
  "octopus-vulgaris-vertical-lobe-sfltract"
  "snemi"
  "whole-mouse-brain"
)

PCTS=(10 20 40 60 80 100)

for domain in "${DOMAINS[@]}"; do
  for pct in "${PCTS[@]}"; do
    MODEL_PATH="${MODEL_DIR}/${domain}/multidomain_final_${pct}.pth"
    LINE=$(python $SCRIPT --domain "$domain" --model_path "$MODEL_PATH")
    echo "$LINE" >> $CSV_FILE
  done
done
