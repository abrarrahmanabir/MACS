#!/bin/bash


all_domains=(
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

target_domains=(
  "micron"
  "octopus-vulgaris-vertical-lobe-glia-deep-neuropil"
  "octopus-vulgaris-vertical-lobe-sfltract"
  "snemi"
  "c-elegans-dauer-stage"
  "dhanyasi-P14-mouse-cerebellum"
  "fly"
  "katz-lab-berghia-connective"
  "whole-mouse-brain"
)



for budget in 10 20 40 60 80 100; do
  for target in "${target_domains[@]}"; do
    sources=()
    ckpts=()
    for source in "${all_domains[@]}"; do
      if [ "$source" != "$target" ]; then
        sources+=("$source")
        ckpts+=("models/$source/model_final.pth")
      fi
    done

    source_domains=$(IFS=' '; echo "${sources[*]}")
    source_ckpts=$(IFS=' '; echo "${ckpts[*]}")

    python multidomain.py \
      --data_root data \
      --source_domains $source_domains \
      --source_ckpts $source_ckpts \
      --target_domain $target \
      --out_path models/$target/multidomain_final_$budget.pth \
      --active_iters 6 --annot_budget $budget --train_epochs_per_iter 5 --batch_size 4
  done
done
