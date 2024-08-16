# DGR-MIL: Exploring Diverse Global Representation in Multiple Instance Learning for Whole Slide Image Classification (ECCV 2024) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2407.03575)

Paper link (preprint): [https://arxiv.org/abs/2407.03575]

## News :fire:
- **June 17, 2024:** We will release the extracted features later and gradually improve this code repository !
- **June 17, 2024:** Congratulations ! Paper has been accepted by ECCV 2024 !

<img align="right" width="50%" height="100%" src="https://github.com/ChongQingNoSubway/DGR-MIL/blob/main/img/network.jpg">

> **Abstract.**   Multiple instance learning (MIL) stands as a powerful approach in weakly supervised learning, regularly employed in histological whole slide image (WSI) classification for detecting tumorous lesions. However, existing mainstream MIL methods focus on modeling correlation between instances while overlooking the inherent diversity among instances. However, few MIL methods have aimed at diversity modeling, which empirically show inferior performance but with a high computational cost. To bridge this gap, we propose a novel MIL aggregation method based on diverse global representation (DGR-MIL), by modeling diversity among instances through a set of global vectors that serve as a summary of all instances. First, we turn the instance correlation into the similarity between instance embeddings and the predefined global vectors through a cross-attention mechanism. This stems from the fact that similar instance embeddings typically would result in a higher correlation with a certain global vector. Second, we propose two mechanisms to enforce the diversity among the global vectors to be more descriptive of the entire bag: (i) positive instance alignment and (ii) a novel, efficient, and theoretically guaranteed diversification learning paradigm. Specifically, the positive instance alignment module encourages the global vectors to align with the center of positive instances (e.g., instances containing tumors in WSI). To further diversify the global representations, we propose a novel diversification learning paradigm leveraging the determinantal point process. The proposed model outperforms the state-of-the-art MIL aggregation models by a substantial margin on the CAMELYON-16 and the TCGA-lung cancer datasets.

> **Result.**  Demostrating proposed method outperform the SOTA method across three groups extract feature based on the CAMELYON-16 and the TCGA-lung cancer datasets.
<img src="https://github.com/ChongQingNoSubway/DGR-MIL/blob/main/img/res.png">


## Enviroment Installation

## Download Extracted Feature 


## Run the Code
 


## Thanks for the code provided by:
- SwinUnet: https://github.com/HuCaoFighting/Swin-Unet
- HiFormer: https://github.com/amirhossein-kz/hiformer
- CASCADE: https://github.com/SLDGroup/CASCADE
- UCTransNet: https://github.com/mcgregorwwww/uctransnet
- ...
