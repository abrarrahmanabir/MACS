# MACS: Multi Domain Adaptation Facilitates Accurate Connectomics Segmentation

This repository contains the implementation of **MACS**, a deep learning framework for connectomics segmentation that introduces the first ever multi-domain adaptation method for connectomics and achieves robust and accurate segmentation performance across diverse connectomics EM datasets.

### Model Architecture
![Model Architecture](macs_model.png)

## Installation
```bash
git clone https://github.com/abrarrahmanabir/MACS.git
cd MACS
```

## Dataset
All preprocessed datasets used in this study are publicly available and include the train, validation, and test splits to ensure reproducibility. You can access the full dataset at the following link:
https://drive.google.com/file/d/1-tuV1zWgcsaz9tEUDRpsrZPoEj9e_glZ/view?usp=sharing
 

## Code Structure
`multidomain.py` : This file contains the complete implementation of our proposed approach **MACS**  and the corresponding source code.

`train_single.py`: This script is used for training each individual source model.

./test.py: This script contains the code for evaluating the trained models.


##  Training

To train the model on a selected problem instance, simply run:
```bash
python run.py

```


