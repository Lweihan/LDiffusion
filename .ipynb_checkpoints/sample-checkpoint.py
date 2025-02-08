import torch
import torch.nn as nn
import torch.nn.functional as F
import transforms
import numpy as np
from PIL import Image 
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 Stable Diffusion 模型
model_path = '/home/disk3/lwh/image_process/evaluation/models/stable_diffusion_v1.5'
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path).to(device)
vae = pipeline.vae
vae.encoder.conv_in = nn.Conv2d(4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vae = vae.to(device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整大小为 512x512
    transforms.ToTensor(),           # 转换为 Tensor
])

img_name = ''
label_name = ''

image = Image.open(img_name).convert("RGB")  # 转换为 RGB 格式
label = Image.open(label_name).convert("L").resize((512, 512)) # 标签通常是单通道

# 将像素值为 255 的位置转成 5
image_array = np.array(label)
image_array[image_array == 255] = 0
pixel_values = image_array // 50
image_tensor = torch.tensor(pixel_values, dtype=torch.uint8)
# 将数组调整为适当的形状，并增加一个维度以符合 [1, 512, 512] 形状
label = image_tensor.view(512, 512).unsqueeze(0)  # 在最前面添加一个维度

linear_layer = nn.Linear(768, 1280).to(device)
input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
input_ids = torch.tensor(input_ids).to(device)
text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
# text_embeddings = linear_layer(text_embeddings)
text_embeddings = torch.tensor(text_embeddings).to(device)

image, label = image.to(device), label.to(device)
with torch.no_grad():
    latents = vae.encode(image).latent_dist.mean

with torch.no_grad():
    decoded_images = vae.decode(latents).sample
    
to_pil = transforms.ToPILImage()
decoded_image = to_pil(decoded_images[0])


num_inference_steps = 10
pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

n = len(pipeline.scheduler.timesteps)
unet = pipeline.unet.to(device)
for i, t in enumerate(pipeline.scheduler.timesteps):
    latents = pipeline.scheduler.scale_model_input(latents, t)
    output = unet(latents, t, text_embeddings)
    latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(device)
    decoded_images = vae.decode(latents).sample
    decoded_image = to_pil(decoded_images[0])
    decoded_image.save('/home/disk3/lwh/image_process/evaluation/code/sample/sample_' + str(i) + '.png')