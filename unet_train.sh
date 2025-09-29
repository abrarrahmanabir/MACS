#!/bin/bash


python train_single.py \
  --domain fly \
  --out_path models/fly/model.pth \
  --epochs 100

python train_single.py \
  --domain snemi \
  --out_path models/snemi/model.pth \
  --epochs 100


python train_single.py \
  --domain micron \
  --out_path models/micron/model.pth \
  --epochs 100
  --device=cuda:2


python train_single.py \
  --domain c-elegans-dauer-stage \
  --out_path models/c-elegans-dauer-stage/model.pth \
  --epochs 100

python train_single.py \
  --domain dhanyasi-P14-mouse-cerebellum \
  --out_path models/dhanyasi-P14-mouse-cerebellum/model.pth \
  --epochs 100

python train_single.py \
  --domain katz-lab-berghia-connective \
  --out_path models/katz-lab-berghia-connective/model.pth \
  --epochs 100

python train_single.py \
  --domain octopus-vulgaris-vertical-lobe-glia-deep-neuropil \
  --out_path models/octopus-vulgaris-vertical-lobe-glia-deep-neuropil/model.pth \
  --epochs 100

python train_single.py \
  --domain octopus-vulgaris-vertical-lobe-sfltract \
  --out_path models/octopus-vulgaris-vertical-lobe-sfltract/model.pth \
  --epochs 100

python train_single.py \
  --domain whole-mouse-brain \
  --out_path models/whole-mouse-brain/model.pth \
  --epochs 500
