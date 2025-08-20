# [CVPR 2025 Highlight] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer

Pytorch implementation of our CVPR 2025 highlight paper ***SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer*** [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_SaMam_Style-aware_State_Space_Model_for_Arbitrary_Image_Style_Transfer_CVPR_2025_paper.pdf)][[supple](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_SaMam_Style-aware_State_CVPR_2025_supplemental.pdf)].

## :newspaper: $\mathrm{I}$ - Introduction

**TL;DR:** We introduce a novel Mamba backbone for arbitrary image style transfer, which strikes a balance between generation quality and efficiency.



Global effective receptive field plays a crucial role for image style transfer (ST) to obtain high-quality stylized results. However, existing ST backbones (e.g., CNNs and Transformers) suffer huge computational complexity to achieve global receptive fields. Recently, State Space Model (SSM), especially the improved variant Mamba, has shown great potential for long-range dependency modeling with linear complexity, which offers an approach to resolve the above dilemma. In this paper, we develop a Mamba-based style transfer framework, termed SaMam. Specifically, a mamba encoder is designed to efficiently extract content and style information. In addition, a style-aware mamba decoder is developed to flexibly adapt to various styles. Moreover, to address the problems of local pixel forgetting, channel redundancy and spatial discontinuity of existing SSMs, we introduce local enhancement and zigzag scan mechanisms. Qualitative and quantitative results demonstrate that our SaMam outperforms state-of-the-art methods in terms of both accuracy and efficiency.

![](Figs/Fig1.png)

*(a) An overview of our SaMam framework; (b) An illustration of the selective scan methods in Vision mamba and VMamba.*

![](Figs/Fig2.png)

*The detailed architecture of Style-aware Vision State Space Module (SAVSSM).*

## :wrench: $\mathrm{II}$ - Dependencies

- cuda=12.6 (>=12.0)
- python=3.10.4
- torch=2.3.0
- torchvision=0.18.1
- tensorboardX=2.6.2.2
- pytorch-lightning=2.3.0
- \($\lhd$\) triton=2.3.1
- \($\lhd$\) [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases)=1.4.0
- \($\lhd$\) [mamba-ssm](https://github.com/state-spaces/mamba/releases)=2.2.2

Note that, if you just plan to [***test** our SaMam model with **pure torch***](#test_with_puretorch), the **un-necessary** package tools are marked with "\($\lhd$\)".


## :bank: $\mathrm{III}$ - Dataset Preparation

For training, we use [wikiart](https://www.kaggle.com/competitions/painter-by-numbers/data) as our style dataset and [MS_COCO](https://cocodataset.org/#download) as our content dataset. For testing, the content and style images are put in folders "./TEST/test_images/content/" and "./TEST/test_images/style/", respectively. Furthermore, the folder structure should be like:

```
TRAIN
├── train_SaMam.py
├── Dataset
│   ├── wikiart
│   │   ├── 0000001.jpg
│   │   ├── 0000002.jpg
│   │   ├── 0000003.jpg
│   │   ├── ...
│   ├── MS_COCO
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── ...
TEST
├── test_image.py
├── test_images
│   ├── content
│   │   ├── content_01.jpg
│   │   ├── content_02.jpg
│   │   ├── ...
│   ├── style
│   │   ├── style_01.jpg
│   │   ├── style_02.jpg
│   │   ├── ...
```

## :lock: $\mathrm{IV}$ - Pretrained VGG Preparation

Download [pretrained VGG](https://drive.google.com/file/d/13BzdootYTuwCiV4VW0sjSxjopWJiiRZg/view?usp=drive_link), and put the VGG checkpoint (vgg_normalised.pth) into folder "./LOSS/vgg_ckp/", which is like:

```
LOSS
├── vgg_ckpt
│   ├── vgg_normalised.pth
```

## :bullettrain_front: $\mathrm{V}$ - Run

### :running: $\mathrm{V}$.1 - Train

Get into training folder "./TRAIN/":

```
cd ./TRAIN/
```

All the training settings are provided in function "parse_args()" of the file "train_SaMam.py". You can adapt them manually.

#### :blush:**Training on mamba_ssm** (default)

If your device is equipped with [mamba-ssm](https://github.com/state-spaces/mamba/releases) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases), you can train our SaMam to strike a fast convergence. With the default setting in "train_SaMam.py", you can train a SaMam model. Run:

```
python train_SaMam.py --content [train_content_dataset_folder] --style [train_style_dataset_folder]
```

For instance, if you plan to train SaMam model on **2 GPUs**, and image patch size is set to 8, a command sample is: 

- *python train_SaMam.py --content ./Dataset/MS_COCO/ --style ./Dataset/wikiart/ --gpus 0 1 --patch-size 8*

#### :worried:**Training on pure torch**

If you just use windows platform or the device can not be equipped with mamba-ssm, you can train our SaMam with only torch. However, you should be patient when training because the convergence speed is **very very very slow**! You can specify hyper-parameter **"mamba-from-trion" to 0**. Run:

```
python train_SaMam.py --content [train_content_dataset_folder] --style [train_style_dataset_folder] --mamba-from-trion 0
```

For instance, if you plan to train SaMam model on **1 GPU** without mamba_ssm, and image patch size is set to 4, a command sample is: 

- *python train_SaMam.py --content ./Dataset/MS_COCO/ --style ./Dataset/wikiart/ --gpus 0 --patch-size 4 --mamba-from-trion 0*

### :walking: $\mathrm{V}$.2 - Test

Please get into test folder "./TEST/".

```
cd ./TEST/
```

All the test settings are provided in function "parse_args()" of the file "test_image.py". You can adapt them manually when you use your own trained model with your own architecture setting.

#### :white_check_mark: Checkpoint

We provide 2 SaMam checkpoints trained on ***patch-size=8*** and ***patch-size=4*** for reference, respectively.

|     patch size      |                              8                               |                              4                               |
| :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **checkpoint link** | [**SaMam_patchsize_x8.ckpt**](https://drive.google.com/file/d/1HH_cSdtUzdgUMspwJ30Osax-LpdF9zag/view?usp=drive_link) | [**SaMam_patchsize_x4.ckpt**](https://drive.google.com/file/d/1fgs0JR06WBh2ACuI5OboDYrutI2G9tC7/view?usp=drive_link) |

Please download them and put them in folder "./TEST/checkpoint/".

#### :blush:**Test on mamba_ssm** (default)

If your device is equipped with [mamba-ssm](https://github.com/state-spaces/mamba/releases) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases), you can test our SaMam with quick inference speed. Run:

```
python test_image.py --content-dir [your_test_content_folder] --style-dir [your_test_style_folder] --output-dir [output_folder] --model_ckpt [SaMam_ckpt_path]
```

The 2 checkpoints trained on different hyper-parameters (e.g., ***patch-size***) can be load by pytorch-lightning automatically. The command samples are: 

- *python test_image.py --content-dir ./test_images/content/ --style-dir ./test_images/style/ --output-dir ./test_images/output --model_ckpt ./checkpoint/SaMam_patchsize_x8.ckpt*
- *python test_image.py --content-dir ./test_images/content/ --style-dir ./test_images/style/ --output-dir ./test_images/output --model_ckpt ./checkpoint/SaMam_patchsize_x4.ckpt*

The stylized results are listed in './TEST/test_images/output/'

#### :worried:**Test on pure torch** <a id="test_with_puretorch"></a>

If you are a windows platform player or don't install mamba-ssm, you can also generate stylized results by pure torch. You should also specify hyper-parameter **"mamba-from-trion" to 0**. Run:

```
python test_image.py --content-dir [your_test_content_folder] --style-dir [your_test_style_folder] --output-dir [output_folder] --model_ckpt [SaMam_ckpt_path] --mamba-from-trion 0
```

To use 2 checkpoints trained on different patch sizes, you should specify hyper-parameter ***patch-size*** manually. The command samples are: 

- *python test_image.py --content-dir ./test_images/content/ --style-dir ./test_images/style/ --output-dir ./test_images/output --model_ckpt ./checkpoint/SaMam_patchsize_x8.ckpt --patch-size 8 --mamba-from-trion 0*
- *python test_image.py --content-dir ./test_images/content/ --style-dir ./test_images/style/ --output-dir ./test_images/output --model_ckpt ./checkpoint/SaMam_patchsize_x4.ckpt  --patch-size 4 --mamba-from-trion 0*

## :star: $\mathrm{VI}$ - Citation

If you find our work useful in your research, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_SaMam_Style-aware_State_Space_Model_for_Arbitrary_Image_Style_Transfer_CVPR_2025_paper.html)~ Thank you!

**[CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_SaMam_Style-aware_State_Space_Model_for_Arbitrary_Image_Style_Transfer_CVPR_2025_paper.pdf)**
```
@inproceedings{liu2025samam,
  title={SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer},
  author={Liu, Hongda and Wang, Longguang and Zhang, Ye and Yu, Ziru and Guo, Yulan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28468--28478},
  year={2025}
}
```

**[Arxiv](https://arxiv.org/pdf/2503.15934)**
```
@article{liu2025samam,
  title={SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer},
  author={Liu, Hongda and Wang, Longguang and Zhang, Ye and Yu, Ziru and Guo, Yulan},
  journal={arXiv preprint arXiv:2503.15934},
  year={2025}
}
```

## :yum: $\mathrm{VII}$ - Acknowledgement

This repository is heavily built upon the amazing works [AdaConv](https://github.com/RElbers/ada-conv-pytorch), [StyTR-2](https://github.com/diyiiyiii/StyTR-2) and [VMamba](https://github.com/MzeroMiko/VMamba). Thanks for their great effort to community.

## :e-mail: $\mathrm{VIII}$ - Contact

[Hongda Liu](mailto:2946428816@qq.com)
