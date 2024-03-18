# MSDDAEF
This is the PyTorch implementation of the Multi-Source Deep Domain Adaptation Ensemble Framework for Cross-Dataset Motor Imagery EEG Transfer Learning

This is an example when GIST is the source domain and openBMI is the target domain.

This is an example when base network is Deep Convnet and distance metric is coral. 

The code is similar when the target domain is exchanged with the source domain.

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

We only provided three examples of target subjects, please create a complete directory to save the multi-source domain models when you actually run the project:

        transfer/model/sub0
                     /sub1
                     /...
                     /sub51
# Dependencies

It is recommended to create a virtual environment with python version 3.7 and running the following:

    pip install -r requirements.txt
## Obtain the raw dataset
Download the raw dataset from the [resources](https://github.com/HZUBCI/MSDDAEF/blob/main/README.md#datasets) above(Please download the MI data in mat file format), and save them to the same folder. 

        process/GIST/s01.mat
                     /s02.mat
                     /...

        process/openBMI/sess01_subj01_EEG_MI.mat
                        /sess01_subj02_EEG_MI.mat
                        /...
                        /sess02_subj01_EEG_MI.mat
                        /sess02_subj02_EEG_MI.mat
                        /...
# If used, please cite:
Multi-Source Deep Domain Adaptation Ensemble Framework for Cross-Dataset Motor Imagery EEG Transfer Learning. submitted to PHYSIOLOGICAL MEASUREMENT. 2023.

# Acknowledgment
We thank Kaishuo Zhang et al and Schirrmeister et al for their wonderful works.

Zhang, Kaishuo, et al. "Adaptive transfer learning for EEG motor imagery classification with deep convolutional neural network." Neural Networks 136 (2021): 1-10.https://doi.org/10.1016/j.neunet.2020.12.013

Schirrmeister, Robin Tibor, et al. "Deep learning with convolutional neural networks for EEG decoding and visualization." Human brain mapping 38.11 (2017): 5391-5420. https://doi.org/10.1002/hbm.23730
