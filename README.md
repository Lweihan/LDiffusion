## L-Diffusion:  Laplace Diffusion for Efficient Pathology Image Segmentation

[![Diffusion: Stable Diffusion v1.5](https://img.shields.io/badge/Diffusion-Stable%20Diffusion%20v1.5-0a66c2)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)
[![Training: DeepSpeed ZeRO-3](https://img.shields.io/badge/Training-DeepSpeed%20ZeRO--3-0052cc)](https://www.deepspeed.ai/)
[![Segmentation: nnUNetv2](https://img.shields.io/badge/Segmentation-nnUNetv2-6f42c1)](https://github.com/MIC-DKFZ/nnUNet)
[![Segmentation: Cellpose](https://img.shields.io/badge/Segmentation-Cellpose-2ea44f)](https://www.cellpose.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](https://www.python.org/)

## 🔖 Jump To

- [✨ Demo of Segmentation Results](#demo)
- [🛠️ Easy Environment Setup](#env)
- [📜 Simple Commands, Efficient Results](#commands)
- [🔬 Task [1]: Cell Segmentation](#cell)
- [🔬 Task [2]: Tissue Segmentation](#tissue)
- [⚡ Quick Evaluation Metrics](#eval)
- [📄 Certificates and Citation](#cert)

<a id="demo"></a>
### ✨ Demo of Segmentation Results

![demo](https://raw.githubusercontent.com/Lweihan/LDiffusion/refs/heads/main/attachment/show.gif)

<a id="env"></a>
### 🛠️ Easy Environment Setup

```shell
conda env create -f environment.yml
cd model
pip install -e .
```

<a id="commands"></a>
### 📜 Simple Commands, Efficient Results

#### Download Pre-trained Stable-Diffusion v1.5
You can download the pre-trained Stable-Diffusion v1.5 model from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)

#### Download PUMA Dataset
You can download the PUMA dataset from [PUMA Dataset](https://drive.google.com/drive/folders/1lzNSJFtSb0IedTC-Y1gKmTHIvCirn9yL?usp=sharing), and the label details can be obtained from [PUMA Label Details](https://puma.grand-challenge.org/dataset/)

<a id="cell"></a>
#### 🔬 Task [1]: Cell Segmentation

###### Training for Cell Segmentation
 - In `ldiffusion.py`, you can set the `level` parameter to `cell` to train the model for cell segmentation.
 - If your GPU memory is enough, you can set the `component` parameter to `all`. It will train both the segmentor and the L-Diffusion model. The segmentor will be trained first, and then the segmentor will be trained using the L-Diffusion's weight.
 - If your GPU memory is limited, you can first set the `component` parameter to `ldiffusion` to warm up L-Diffusion, and then set the `component` parameter to `segmentor` to train the segmentor model, and set the `ldiffusion_weight` parameter to the path of the L-Diffusion model weight.

---

###### GPU Enough
```python
if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    trainer = LDiffusionModel(args.diffusion_path, level="cell")
    trainer.train(args, component="all", ldiffusion_weight=None, controlnet_weight=None)
````
###### GPU Limited

- First, use the following code to warm up L-Diffusion:
```python
if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    trainer = LDiffusionModel(args.diffusion_path, level="cell")
    trainer.train(args, component="ldiffusion", ldiffusion_weight=None, controlnet_weight=None)
````
- Then, use the following code to train the segmentor model:
```python
if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    trainer = LDiffusionModel(args.diffusion_path, level="cell")
    trainer.train(args, component="segmentor", ldiffusion_weight='your ldiffusion weight path', controlnet_weight=None)
````

---

Current terminal location is here:
```shell
$ pwd
~/project

project/
├── LDiffusion/
```
Run the following command to train the model:
```shell
python -m LDiffusion.ldiffusion --diffusion-path [stable_diffusion_v1.5] --image-dir .../PUMA/01_training_dataset_tif_ROIs --label-dir .../PUMA/01_training_dataset_png_ROIs_nuclei --num-epochs 100 --batch-size 1 --num-inference-steps 5 --num-classes 11
```

For deepspeed memory optimization:
```shell
deepspeed --num_gpus 8 --master_port 29051 --module LDiffusion.ldiffusion --diffusion-path [stable_diffusion_v1.5] --image-dir .../PUMA/01_training_dataset_tif_ROIs --label-dir .../PUMA/01_training_dataset_png_ROIs_nuclei --num-epochs 100 --batch-size 1 --num-inference-steps 5 --num-classes 11
```

###### Inference for Cell Segmentation

```python
import matplotlib.pyplot as plt
from PIL import Image
from LDiffusion.ldiffusion import LDiffusionModel

ldiffusion_model = LDiffusionModel(
    diffusion_path="../stable_diffusion_v1.5",
    level="cell"
)

ldiffusion_image, mask = ldiffusion_model.inference(
    image_path="../PUMA/01_training_dataset_tif_ROIs/training_set_primary_roi_002.tif",
    dtm_path=None,
    ldiffusion_weight='../LDiffusion/train_save/unet/XX_XX_XX',
    segmentor_weight='../LDiffusion/train_save/cellclassifier/XX_XX_XX',
    num_classes=11
)

image = Image.open("../PUMA/01_training_dataset_tif_ROIs/training_set_primary_roi_002.tif").convert('RGB')

gt = convert_labels("../PUMA/01_training_dataset_png_ROIs_tissue/training_set_primary_roi_002.png")

...
```

<img src="./attachment/cell_sample_show.png" alt="Tissue Segmentation Results" style="display: block; margin: 0 auto; width: 100%; border: none; box-shadow: none;" />

*** 

<a id="tissue"></a>
#### 🔬 Task [2]: Tissue Segmentation

###### Training for Tissue Segmentation
 - In `ldiffusion.py`, you can set the `level` parameter to `tissue` to train the model for tissue segmentation.
 - If your GPU memory is enough, you can set the `component` parameter to `all`. It will train both the segmentor and the L-Diffusion model. The segmentor will be trained first, and then the segmentor will be trained using the L-Diffusion's weight.
 - If your GPU memory is limited, you can first set the `component` parameter to `ldiffusion` to warm up L-Diffusion, and then set the `component` parameter to `segmentor` to train the segmentor model, and set the `ldiffusion_weight` parameter to the path of the L-Diffusion model weight.

```python
if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    trainer = LDiffusionModel(args.diffusion_path, level="tissue")
    trainer.train(args, component="all", ldiffusion_weight='your ldiffusion weight path')
```

Run the following command to train the model:

```shell
python -m LDiffusion.ldiffusion --diffusion-path [stable_diffusion_v1.5] --image-dir .../PUMA/01_training_dataset_tif_ROIs --label-dir .../PUMA/01_training_dataset_png_ROIs_tissue --num-epochs 100 --batch-size 2 --num-inference-steps 5 --num-classes 7
```

###### Inference for Tissue Segmentation

```shell
export nnUNet_raw_data_base="/your_path_to/nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/your_path_to/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/your_path_to/nnUNet/nnUNet_results"
```

Single Image Inference for Tissue Segmentation
```python
import matplotlib.pyplot as plt
from PIL import Image
from LDiffusion.ldiffusion import LDiffusionModel

ldiffusion_model = LDiffusionModel(
    diffusion_path="../stable_diffusion_v1.5",
    level="tissue"
)

ldiffusion_image, mask = ldiffusion_model.inference(
    image_path="../PUMA/01_training_dataset_tif_ROIs/training_set_primary_roi_002.tif",
    dtm_path=None,
    ldiffusion_weight='../LDiffusion/train_save/unet/XX_XX_XX',
    segmentor_weight='../LDiffusion/model/nnunetv2/nnunetv2_hist/nnUNet_results/XXX/nnUNetTrainer__nnUNetPlans__2d/fold_0',
    num_classes=7
)

image = Image.open("../PUMA/01_training_dataset_tif_ROIs/training_set_primary_roi_002.tif").convert('RGB')

gt = convert_labels("../PUMA/01_training_dataset_png_ROIs_tissue/training_set_primary_roi_002.png")

...
```

<img src="./attachment/tissue_sample_show.png" alt="Tissue Segmentation Results" style="display: block; margin: 0 auto; width: 100%; border: none; box-shadow: none;" />

Batch Inference for Tissue Segmentation
```python
import matplotlib.pyplot as plt
from PIL import Image
from LDiffusion.ldiffusion import LDiffusionModel

ldiffusion_model = LDiffusionModel(
    diffusion_path="../stable_diffusion_v1.5",
    level="tissue"
)

ldiffusion_image, mask = ldiffusion_model.inference(
    image_path="../PUMA/01_training_dataset_tif_ROIs/",
    dtm_path=None,
    output_path="../PUMA/predicted_results/",
    ldiffusion_weight='../LDiffusion/train_save/unet/XX_XX_XX',
    segmentor_weight='../LDiffusion/model/nnunetv2/nnunetv2_hist/nnUNet_results/XXX/nnUNetTrainer__nnUNetPlans__2d/fold_0',
    num_classes=7
)
```

<a id="eval"></a>
### ⚡ Quick Evaluation Metrics

```shell
python -m LDiffusion.evaluate --image-dir [predicted images directory] --label-dir [ground truth images directory] --num-classes X
```

The evaluation results will be saved in `./LDiffusion/eval/eval_report`

<a id="cert"></a>
## 📄 Certificates and Citation

[![Model Card: Stable Diffusion](https://img.shields.io/badge/Model%20Card-Stable%20Diffusion%20v1.5-blue)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[![Engine: DeepSpeed](https://img.shields.io/badge/Engine-DeepSpeed-informational)](https://www.deepspeed.ai/)
[![Method: nnUNetv2](https://img.shields.io/badge/Method-nnUNetv2-purple)](https://github.com/MIC-DKFZ/nnUNet)

If your institution requires project certificates, you can append links here:

- Model governance certificate
- Data compliance certificate
- Clinical validation certificate

Recommended citation entry:

```bibtex
@inproceedings{li2025diffusion,
  title={L-Diffusion: Laplace Diffusion for Efficient Pathology Image Segmentation},
  author={Li, Weihan and Zhou, Linyun and Zhang, Shengxuming and Du, Xiangtong and Zhang, Xiuming and Zhang, Jing and Song, Mingli and Feng, Zunlei and others},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
