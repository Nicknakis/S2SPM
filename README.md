# Signed Two-Space Proximity Model

Python 3.8.3 and Pytorch 1.12.1 implementation of the Signed Two-Space Proximity Model for Learning Representations in Protein-Protein Interaction Networks.

## Description

Accurately predicting complex protein-protein interactions (PPIs) is crucial for decoding biological processes, from cellular functioning to disease mechanisms. However, experimental methods for determining PPIs are computationally expensive. Thus, attention has been recently drawn to machine learning approaches. Furthermore, insufficient effort has been made toward analyzing signed PPI networks, which capture both activating (positive) and inhibitory (negative) interactions. 
To accurately represent biological relationships, we present the Signed Two-Space Proximity Model (S2-SPM) for signed PPI networks, which explicitly incorporates positive and negative interactions, reflecting the complex regulatory mechanisms within biological systems. This is achieved by leveraging two independent latent spaces to differentiate between positive and negative interactions while representing protein similarity through proximity in these spaces. Our approach also enables the identification of archetypes, representing extreme protein profiles. S2-SPM's superior performance in predicting the presence and sign of interactions in SPPI networks is demonstrated in link prediction tasks against relevant baseline methods. Additionally, the biological prevalence of the identified archetypes is confirmed by an enrichment analysis of Gene Ontology (GO) terms, which reveals that distinct biological tasks are associated with archetypal groups formed by both interactions. This study is also validated regarding statistical significance and sensitivity analysis, providing insights into the functional roles of different interaction types. Finally, the robustness and consistency of the extracted archetype structures are confirmed using the Bayesian Normalized Mutual Information (BNMI) metric, proving the model's reliability in capturing meaningful SPPI patterns. 

## Installation

### Create a Python 3.8.3 environment with conda

```
conda create -n ${env_name} python=3.8.3  
```

### Activate the environment

```
conda activate ${env_name} 
```

### Please install the required packages

```
pip install -r requirements.txt
```

### Additional packages

Our Pytorch implementation uses the [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) package. Installation guidelines can be found at the corresponding [Github repository](https://github.com/rusty1s/pytorch_sparse).

#### For a cpu installation please use: 

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu.html```

#### For a gpu installation please use:

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html```

where ${CUDA} should be replaced by either cu102, cu113, or cu116 depending on your PyTorch installation.



## For learning embeddings in Signed Protein-Protein Interaction Networks with S2SPM:

**RUN:** &emsp; ```python main.py```

optional arguments:

**--epochs**  &emsp;  number of epochs for training (default: 5K)

**--cuda**  &emsp;    CUDA training (default: True)

**--LP**   &emsp;     performs link prediction (default: True)

**--D**   &emsp;      dimensionality of the embeddings (default: 8)

**--lr**   &emsp;     learning rate for the ADAM optimizer (default: 0.05)

**--dataset** &emsp;  dataset to apply S2SPM on (default: Sapiens)

**--sample_percentage** &emsp;  sample size network percentage, it should be equal or less than 1 (default: 0.3)



### Additional example for learning eight-dimensional embeddings running on cpu:

**RUN:** &emsp; ```python main.py --cuda False --D 8 --pretrained False```


### CUDA Implementation

The code has been primarily constructed and optimized for running in a GPU-enabled environment, running the code in CPU is significantly slower.

## Reference



