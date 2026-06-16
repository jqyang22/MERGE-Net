<div align="center">

<h1>Exploring Directional Sparsity and Cross-Modal Contrast-Consensus: MERGE-Net for HSI-SAR/LiDAR Joint Classification</h1>

<h2>Under Review</h2>
<h2>(The code will be made publicly available upon publication. For early access, please feel free to reach out.)</h2>


[Jiaqi Yang](https://jqyang22.github.io/)<sup>a</sup>, [Bo Du](https://cs.whu.edu.cn/info/1019/2892.htm/)<sup>b</sup>, [Rong Liu](https://gp.sysu.edu.cn/teacher/3702/)<sup>c</sup>, [Jiayang Huang]()<sup>c</sup>, [Shizhen Chang](https://shizhenchang.github.io/)<sup>d</sup>, [Liangpei Zhang](https://www.zhangliangpei.cn/)<sup>b</sup>

<sup>a</sup> University of Wisconsin-Madison,
<sup>b</sup> Wuhan University,
<sup>c</sup> Sun Yat-sen University, 
<sup>d</sup> Linköping University.

</div>

<div align="center">

<!--
<p align='center'>
  <a href="https://ieeexplore-ieee-org.ezproxy.library.wisc.edu/document/11494135"><img alt="Pape" src="https://img.shields.io/badge/TGRS-Paper-6D4AFF?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-project-structure">Project Structure</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-citation">Citation</a>
</p>
-->

</div>


# 🧩 Overview

Multimodal satellite imagery can provide more comprehensive information, which shows a new perspective for observing Earth’s surface. Particularly, the joint use of hyperspectral image (HSI), synthetic aperture radar (SAR), or light detection and ranging (LiDAR) have gained widespread attention due to the concurrent acquisition of spectral-spatial information, structural information or ground elevation. However, most of existing works do not explicitly learn effective intra- and inter-modal mutual information. This induces information redundancy and curtails feature diversity, stalling performance advances across heterogeneous modalities. To address the above challenge, a unified Mutual lEaRning with Geometric Equilibrium network (MERGE-Net) is proposed for HSI-SAR/LiDAR collaborative classification. Concretely, a direction-oriented sparse modeling module is first designed for intra-modal mutual learning that selectively capturing information among structurally representative neighbors, thereby suppressing redundancy while enhancing discriminative spectral-spatial cues. Then, a continuous contrast-consensus integration mechanism is put forward for inter-modal mutual learning, where complementary information and consistent semantics are iteratively propagated to align heterogeneous modalities at multiple semantic depths. Last, a geometric equilibrium equiangular tight frame head is devised to suppresses minority collapse and stabilize decision boundaries existed in the marker-poor remote sensing data. In this way, MERGE-Net can simultaneously explore intra- and inter-modal mutual relationships while enhance discriminative and balanced class separation, offering a new pathway to multimodal classification. Experiments and analysis on three benchmark datasets implicate the effectiveness of the proposed approach compared to state-of-the-art (SOTA) methods.</a>


<figure>
<div align="center">
<img src=Fig/MERGE-Net.bmp width="80%">
</div>

<div align='center'>
 
**Figure 1. Flowchart of MERGE-Net.**

</div>

<div align='center'>

</div>
<br>

<!--
# 📁 Project Structure

```
THSGR/
├── main.py                       # full pipeline
├── config/
│   └── config.yaml               # dataset / network / output configuration
├── loadData/
│   ├── data_pipe.py              # DataLoader assembly
│   ├── data_reader.py            # raw .mat readers
│   └── split_data.py             # train / val / test split utilities
└── models/
    ├── THSGR.py                   # main THSGR network
    └── transformer.py            # transformer structure
```
-->

# 🌍 Datasets
HSI-SAR Augsburg: https://github.com/danfenghong/ISPRS_S2FL <br>
HSI-LiDAR Houston: https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/ <br>
HSI-SAR Berlin: https://github.com/danfenghong/ISPRS_S2FL <br>

<div align='center'>
</div>


<!--
# 📦 Installation

This project is implemented with **PyTorch**:

| Package | Version |
| :------ | :------ |
| pytorch | 1.7+ |
| numpy | 1.21.4 |
| matplotlib | 3.3.3 |
| scikit-learn | 0.23.2 |
| einops | 0.4+ |
| timm | 0.6+ |
| thop | 0.1+ |
| pyyaml | 5.4+ |
| scipy | 1.7+ |
| pandas | 1.3+ |

Install with:

```bash
conda create -n thsgr python=3.8 -y
conda activate thsgr
pip install torch==1.7.1 torchvision==0.8.2
pip install numpy==1.21.4 matplotlib==3.3.3 scikit-learn==0.23.2 \
            einops timm thop pyyaml scipy pandas xlwt
```

# 🔨 Usage
## 1. Prepare data
Download the HSI / LiDAR `.mat` files and place them under one folder.

## 2. Configure paths
Edit `config/config.yaml` for the data path.

## 3. Train & test

```bash
python main.py --device cuda:0  --path-config /your/path/THSGR/config/config.yaml
```

# ⭐ Citation

If you find this work helpful, please give a ⭐ and cite it as follows:

```
@ARTICLE{11494135,
  author={Yang, Jiaqi and Du, Bo and Liu, Rong and Mao, Zhu and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Boosting Multimodal Remote Sensing Image Classification with Transformer-based Heterogeneously Salient Graph Representation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Earth Observing System;Sentinel-1;Sentinel-2;Apertures;Feeds;Antennas;Filtering;Filters;Modulation;Communications technology;Multimodal classification;HSI-SAR/LiDAR imagery;heterogeneously salient graph representation;transformer},
  doi={10.1109/TGRS.2026.3686762}}
```
 -->
