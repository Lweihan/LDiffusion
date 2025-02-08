# LDiffusion
L-Diffusion: Laplace Diffusion for Efficient Pathology Image Segmentation

## L-Diffusion:  Laplace Diffusion for Efficient Pathology Image Segmentation

### ✨ Demo of Segmentation Results

![demo]()

### 🛠️ Easy Environment Setup

```shell
conda env create -f environment.yml
```

### 📜 Simple Commands, Efficient Results

#### 🔥 Mastering Laplace Diffusion Training:

```shell
python diffusion_train.py --diffusion-path Stable Diffusion Pretrained Models Path --image-dir /dataset/PUMA/01_training_dataset_tif_ROIs --label-dir /dataset/PUMA/01_training_dataset_png_ROIs_tissue --num-epochs 100 --batch-size 1 --num-inference-steps X
```

#### ⚡Pixel Latent Vector Extraction:

```shell
python pixel_latent_vector.py --diffusion-path Stable Diffusion Pretrained Models Path --save-model /train_save/unet/XX_XX_XX --image-dir /dataset/PUMA/01_training_dataset_tif_ROIs --label-dir /dataset/PUMA/01_training_dataset_png_ROIs_tissue --num-inference-steps X
```

#### 🔬 Training a Segmentation Model:

```shell
python classifier_train.py --vector-path /eval/vector_set/XX_XX_XX --num-inference-steps X --categories N --epochs 10
```
#### 🔍 Evaluating Your Segmentation Models:
```shell
python classifier_train.py --vector-path /eval/vector_set/XX_XX_XX --num-inference-steps X --categories N --epochs 10 --evaluation
```
