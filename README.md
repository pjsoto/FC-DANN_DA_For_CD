# FC-DANN_DA_For_CD (Fully Convolutional DANN Domain Adaptation for Change Detection)
In the current project, we provide the code used in [] to perform Domain Adaptation based on Domain-Adversarial Neural Network (DANN) [1] for change detection in remote sensing images adapted to a fully convolutional scheme, specifically for deforestation detection in tropical forests such as the Amazon rainforest and the Brazilian savannah. The code here presented

The following figure shows the proposed methodology. The domain adaptation process begins by selecting class-wise balanced training samples from both domains. A traditional down/upsampling strategy can be adopted for the source domain because the class labels are available. However, such a balancing procedure can not be applied straightforwardly for the target domain because the target labels are unknown during training. We used the methodology proposed in [3], which proposes a pseudo-labeling scheme based on Change Vector Analysis (CVA) and a thresholding technique based on the OTSU to produce a pseudo-label map used to select a less imbalanced training set in target domains. In this work, the target samples are also forwarded through the label predictor, where the classification loss is computed using the before-mentioned pseudo label.



# Prerequisites
1- Python 3.7.4

2- Tensorflow 1.14

# Docker
Aiming at simplifying Python environment issues, we provide the [docker container](https://hub.docker.com/repository/docker/psoto87/tf1.15.5-gpu/general) used to conduct the experiments discussed in the paper.

# Dataset
Such implementation has been evaluated in a change detection task, namely deforestation detection, and aiming at reproducing the results obtained in [2] and [3], we make available the images used in this project which can be found in the following links for the [Amazon Biome](https://drive.google.com/drive/folders/1V4UdYors3m3eXaAHXgzPc99esjQOc3mq?usp=sharing) as well as for the [Cerrado](https://drive.google.com/drive/folders/14Jsw0LRcwifwBSPgFm1bZeDBQvewI8NC?usp=sharing). In the same way, the references can be obtained by clicking on [Amazon references](https://drive.google.com/drive/folders/15i04inGjme56t05gk98lXErSRgRnU30x?usp=sharing) and [Cerrado references](https://drive.google.com/drive/folders/1n9QZA_0V0Xh8SrW2rsFMvpjonLNQPJ96?usp=sharing).





# References

[1] Ganin and V. Lempitsky, “Unsupervised   domain   adaptation  by backpropagation,”arXiv preprint arXiv:1409.7495, 2014.

[2] Vega, P. J. S. (2021). DEEP LEARNING-BASED DOMAIN ADAPTATION FOR CHANGE DETECTION IN TROPICAL FORESTS (Doctoral dissertation, PUC-Rio).

[3] Soto, Pedro J., et al. "Domain-adversarial neural networks for deforestation detection in tropical forests." IEEE Geoscience and Remote Sensing Letters 19 (2022): 1-5.
