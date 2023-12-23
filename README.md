# MSDDAEF
This is the PyTorch implementation of the Multi-Source Deep Domain Adaptation Ensemble Framework for Cross-Dataset Motor Imagery EEG Transfer Learning

This is an example when GIST is the source domain and openBMI is the target domain. The code is similar when the target domain is exchanged with the source domain.

![](https://github.com/HZUBCI/MSDDAEF/blob/main/MSDDAEF.png)

The aim of this work is to explore the feasibility  of cross-dataset knowledge transfer. This can largely relax the constraint of training samples for MI BCIs and thus has important practical sense.

# Resources
## Datasets
openBMI:[Link](http://gigadb.org/dataset/100542)

GIST:[Link](http://gigadb.org/dataset/100295)

## Sample pre-trained models
For openBMI:[Link](https://github.com/HZUBCI/MSDDAEF/tree/main/pretrain/pretrain_model_54)

## Sample multi-source adaptation models
For openBMI:[Link](https://github.com/HZUBCI/MSDDAEF/tree/main/transfer/model)

# Dependencies

It is recommended to create a virtual environment with python version 3.7 and running the following:

    pip install -r requirements.txt
## Obtain the raw dataset
Download the raw dataset from the [resources](https://github.com/yzmmmzjhu/CT-adaptTL/blob/main/README.md#datasets) above(Please download the ME/MI data in mat file format), and save them to the same folder. 

        datasets/GIST/s01.mat
                     /s02.mat
                     /...

        datasets/openBMI/sess01_subj01_EEG_MI.mat
                        /sess01_subj02_EEG_MI.mat
                        /...
                        /sess02_subj01_EEG_MI.mat
                        /sess02_subj02_EEG_MI.mat
                        /...
