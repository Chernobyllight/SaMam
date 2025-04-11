# [CVPR 2025 Highlight] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer

Pytorch implementation of our CVPR 2025 highlight paper ***SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer***.

## Introduction

**TL;DL:** We introduce a Mamba backbone for arbitrary image style transfer, which strikes a balance between generation quality and efficiency.



Global effective receptive field plays a crucial role for image style transfer (ST) to obtain high-quality stylized results. However, existing ST backbones (e.g., CNNs and Transformers) suffer huge computational complexity to achieve global receptive fields. Recently, State Space Model (SSM), especially the improved variant Mamba, has shown great potential for long-range dependency modeling with linear complexity, which offers an approach to resolve the above dilemma. In this paper, we develop a Mamba-based style transfer framework, termed SaMam. Specifically, a mamba encoder is designed to efficiently extract content and style information. In addition, a style-aware mamba decoder is developed to flexibly adapt to various styles. {Moreover, to address the problems of local pixel forgetting, channel redundancy and spatial discontinuity of existing SSMs, we introduce local enhancement and zigzag scan mechanisms.} Qualitative and quantitative results demonstrate that our SaMam outperforms state-of-the-art methods in terms of both accuracy and efficiency.

![](Figs/Fig1.png)

*(a) An overview of our SaMam framework; (B) An illustration of the selective scan methods in Vision mamba and VMamba.*

![](Figs/Fig2.png)

*The detailed architecture of Style-aware Vision State Space Module (SAVSSM).*



## Dependencies

- python=3.10.4
- torch=2.3.0
- torchvision=0.18.1
- pytorch-lightning=2.3.0
- trion=2.3.1
- causal-conv1d=1.4.0
- mamba-ssm=2.2.2
- cuda=12.6 (>=12.0)



## Dataset Preparation

We use [wikiart](https://www.kaggle.com/competitions/painter-by-numbers/data) as our style dataset and [MS_COCO](https://cocodataset.org/#download) as our content dataset. Furthermore, the folder structure should be like:

```
Dataset
├── wikiart
│   ├── lhq_1024_jpg
│   ├── 0000001.jpg
│   ├── 0000002.jpg
│   ├── 0000003.jpg
│   ├── ...
├── MS_COCO
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── 000003.jpg
│   ├── ...
├── test_data
│   ├── content
│   │   ├── c1.jpg
│   │   ├── c2.jpg
│   │   ├── ...
│   ├── style
│   │   ├── s1.jpg
│   │   ├── s2.jpg
│   │   ├── ...
```



## Train

:blush:**Training on mamba_ssm:** (default)

If your device is equipped with [mamba_ssm](https://github.com/Dao-AILab/causal-conv1d/releases) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases), you can train our SaMam to strike a fast convergence.



:worried:**Training on pure torch:**

If you just use windows platform or the device can not be equipped with mamba_ssm, you can train our SaMam with only torch. However, the convergence speed is **very very slow**!



## Test

:blush:**Test on mamba_ssm:** (default)

If your device is equipped with [mamba_ssm](https://github.com/Dao-AILab/causal-conv1d/releases) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases), you can test our SaMam with quick inference.



:worried:**Test on pure torch:**

If you are a windows platform player or don't install mamba_ssm, you can also generate stylized results by pure torch.



## Citation

If you find our work useful in your research, please cite our [paper](https://arxiv.org/pdf/2503.15934)~

```
@article{liu2025samam,
  title={SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer},
  author={Liu, Hongda and Wang, Longguang and Zhang, Ye and Yu, Ziru and Guo, Yulan},
  journal={arXiv preprint arXiv:2503.15934},
  year={2025}
}
```

## Acknowledgement

This repository is heavily built upon the amazing works [AdaConv](https://github.com/RElbers/ada-conv-pytorch) and [StyTR-2](https://github.com/diyiiyiii/StyTR-2). Thanks for their great effort.

## Contact

[Hongda Liu](2946428816@qq.com)