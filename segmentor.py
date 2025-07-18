import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torchvision import transforms
from .model.loss import CombinedLoss
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DConditionModel, StableDiffusionImg2ImgPipeline
from PIL import Image

class Segmentor:
    def __init__(self, train_loader, val_loader, level, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.level = level
        self.num_classes = num_classes
        self.model = None
        self.train_loader, self.val_loader = train_loader, val_loader

    def initialize_model(self, level, num_classes):
        if level == "tissue":
            # Initialize tissue-level model (SAM + DeepLab)
            from .model.conductor import TissueSegNet
            model = TissueSegNet(num_classes)
        elif level == "cell":
            from .model.conductor import CellSegClassifier
            model = CellSegClassifier(num_classes)
        else:
            raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

        model.to(self.device)
        return model

    def load_ldiffusion(self, ldiffusion_weight, diffusion_path):
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).eval().to(self.device)
        vae = pipeline.vae.to(self.device)
        return pipeline, unet, vae

    def ldiffusion_augment(self, inputs, pipeline, unet, vae):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # 调整大小以匹配模型的输入要求
            transforms.ToTensor(),
        ])

        linear_layer = nn.Linear(768, 1280).to(self.device)
        input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
        input_ids = torch.tensor(input_ids).to(self.device)
        text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
        text_embeddings = linear_layer(text_embeddings)
        text_embeddings = text_embeddings.clone().detach().to(self.device)

        decoded_image_list = []
        for index in range(len(inputs)):
            image = inputs[index].unsqueeze(0).to(self.device)
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=self.device)
            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]
            decoded_image_tensor = transform(decoded_image).unsqueeze(0)
            decoded_image_list.append(decoded_image_tensor)

        decoded_images_tensor = torch.cat(decoded_image_list, dim=0)
        return decoded_images_tensor.to(self.device)

    def micro_dice(self, predicted_labels, true_labels, num_classes=7):

        predicted_labels = torch.argmax(predicted_labels, dim=1)  # shape: (batch_size, height, width)

        true_labels = true_labels.view(-1)  # shape: (batch_size * height * width,)
        predicted_labels = predicted_labels.view(-1)  # shape: (batch_size * height * width,)

        dice_scores = torch.zeros(num_classes, device=true_labels.device)

        for class_id in range(num_classes):
            true_class = (true_labels == class_id).float()
            predicted_class = (predicted_labels == class_id).float()

            if torch.sum(true_class) == 0 and torch.sum(predicted_class) == 0:
                dice_scores[class_id] = 1
            else:
                # 计算TP, FP, FN
                TP = torch.sum(true_class * predicted_class)
                FP = torch.sum((1 - true_class) * predicted_class)
                FN = torch.sum(true_class * (1 - predicted_class))

                if TP + FP + FN == 0:
                    dice_scores[class_id] = 0
                else:
                    dice_scores[class_id] = 2 * TP / (2 * TP + FP + FN)

        average_dice = torch.mean(dice_scores)

        return dice_scores, average_dice

    def build_augmented_dataloader(self, dataloader, augment_fn, pipeline, unet, vae, device, batch_size, category, num_workers=0):
        all_aug_inputs = []
        all_masks = []

        for inputs, masks, _ in tqdm(dataloader, desc=f"Cache L-Diffusion Augmented {category} Inputs"):
            inputs, masks = inputs.to(device), masks.to(device)
            with torch.no_grad():
                aug_inputs = augment_fn(inputs, pipeline, unet, vae)
            all_aug_inputs.append(aug_inputs.cpu())
            all_masks.append(masks.cpu())

        aug_inputs_tensor = torch.cat(all_aug_inputs, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        dataset = TensorDataset(aug_inputs_tensor, masks_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return loader

    def train_tissue_model(self, epochs, ldiffusion_weight, diffusion_path):
        print("Training tissue-level model...")
        if self.model is None and self.level == "tissue":
            self.model = self.initialize_model("tissue", self.num_classes)
        criterion, checkpoint = CombinedLoss(num_classes=self.num_classes), 0.0
        optimizer = torch.optim.AdamW([
            {'params': self.model.backbone.parameters(), 'lr': 1e-5},
            {'params': self.model.decoder.parameters(), 'lr': 1e-4},
        ])
        pipeline, unet, vae = self.load_ldiffusion(ldiffusion_weight, diffusion_path)
        current_date = datetime.now().strftime("%y_%m_%d")

        aug_train_loader = self.build_augmented_dataloader(self.train_loader, self.ldiffusion_augment, pipeline, unet, vae, self.device, batch_size=2, category="Train", num_workers=4)
        aug_val_loader = self.build_augmented_dataloader(self.val_loader, self.ldiffusion_augment, pipeline, unet, vae, self.device, batch_size=1, category="Validation", num_workers=4)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, masks in tqdm(aug_train_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Train"):
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_running_loss = running_loss / len(aug_train_loader)
            print(f"Epoch {epoch + 11}/{epochs + 10}, Loss: {avg_running_loss:.4f}")

            self.model.eval()
            total_dice = 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(aug_val_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Validation"):
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    outputs = self.model(imgs)['out']
                    dice_categories, average_dice = self.micro_dice(outputs, masks, num_classes=self.num_classes)
                    total_dice += average_dice

            total_dice = total_dice / len(aug_val_loader)
            if total_dice > checkpoint:
                checkpoint = total_dice
                os.makedirs(f"./LDiffusion/train_save/convnext_tiny/{current_date}", exist_ok=True)
                torch.save(self.model.state_dict(),f"./LDiffusion/train_save/convnext_tiny/{current_date}/convnext_tiny.pth")
                print(f"\033[32mNew Best Validation Dice Score: {total_dice:.4f}\033[0m")
            else:
                print(f"Validation Dice Score: {total_dice:.4f}")

    def train_cell_model(self, epochs, ldiffusion_weight, diffusion_path):
        print("Training cell-level model...")
        if self.model is None:
            self.model = self.initialize_model("cell", self.num_classes)
        criterion, checkpoint = CombinedLoss(num_classes=self.num_classes), 0.0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        pipeline, unet, vae = self.load_ldiffusion(ldiffusion_weight, diffusion_path)
        current_date = datetime.now().strftime("%y_%m_%d")

        aug_train_loader = self.build_augmented_dataloader(self.train_loader, self.ldiffusion_augment, pipeline, unet,
                                                           vae, self.device, batch_size=1, category="Train",
                                                           num_workers=4)
        aug_val_loader = self.build_augmented_dataloader(self.val_loader, self.ldiffusion_augment, pipeline, unet, vae,
                                                         self.device, batch_size=1, category="Validation",
                                                         num_workers=4)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, masks in tqdm(aug_train_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Train"):
                masks = masks.to(self.device)
                image_np = inputs[0].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]

                optimizer.zero_grad()
                outputs = self.model(image_np)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            avg_running_loss = running_loss / len(aug_train_loader)
            print(f"Epoch {epoch + 11}/{epochs + 10}, Loss: {avg_running_loss:.4f}")

            self.model.eval()
            total_dice = 0.0
            with torch.no_grad():
                for images, labels in tqdm(aug_val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val"):
                    image_np = images[0].cpu().permute(1, 2, 0).numpy()
                    labels = labels.to(self.device)
                    outputs = self.model(image_np)["out"]
                    dice_scores, avg_dice = self.micro_dice(
                        predicted_labels=outputs,
                        true_labels=labels,
                        num_classes=self.num_classes
                    )
                    total_dice += avg_dice.item()

            total_dice = total_dice / len(aug_val_loader)
            if total_dice > checkpoint:
                checkpoint = total_dice
                os.makedirs(f"./LDiffusion/train_save/cellclassifier/{current_date}", exist_ok=True)
                torch.save(self.model.state_dict(),
                           f"./LDiffusion/train_save/cellclassifier/{current_date}/cellclassifier.pth")
                print(f"\033[32mNew Best Validation Dice Score: {total_dice:.4f}\033[0m")
            else:
                print(f"Validation Dice Score: {total_dice:.4f}")

    def inference_tissue_model(self, image_path, diffusion_path, ldiffusion_weight, segmentor_weight):
        print("Running inference on tissue-level model...")
        if self.model is None:
            self.model = self.initialize_model("tissue", self.num_classes)

        # Load pipeline and models
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(segmentor_weight, "convnext_tiny.pth"), weights_only=True))
        self.model.to(self.device)

        # Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        if image.mode != "RGB":
            raise ValueError(f"Input image is not in RGB mode: {image.mode}")
        width, height = image.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            latents = pipeline.vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=self.device)
            linear_layer = nn.Linear(768, 1280).to(self.device)
            input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
            input_ids = torch.tensor(input_ids).to(self.device)
            text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
            text_embeddings = linear_layer(text_embeddings)
            text_embeddings = text_embeddings.clone().detach().to(self.device)

            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]

            # Pass the decoded image through the model
            model_input = transform(decoded_image).unsqueeze(0).to(self.device)
            outputs = self.model(model_input)["out"]
            pred_mask = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize((width, height), resample=Image.NEAREST)
            pred_mask_resized = np.array(pred_mask_resized)

            decoded_image = decoded_image.resize((width, height), Image.BILINEAR)

            return decoded_image, pred_mask_resized

    def inference_cell_model(self, image_path, diffusion_path, ldiffusion_weight, segmentor_weight):
        print("Running inference on cell-level model...")
        if self.model is None:
            self.model = self.initialize_model("cell", self.num_classes)

        # Load pipeline and models
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(segmentor_weight, "cellclassifier.pth"), weights_only=True))
        self.model.to(self.device)

        # Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        if image.mode != "RGB":
            raise ValueError(f"Input image is not in RGB mode: {image.mode}")
        width, height = image.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()

        # Ensure image dimensions are correct
        if image.dim() != 4:
            raise ValueError(f"Input image tensor has invalid dimensions: {image.dim()} (expected 4).")

        # Perform inference
        with torch.no_grad():
            latents = pipeline.vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=self.device)
            linear_layer = nn.Linear(768, 1280).to(self.device)
            input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
            input_ids = torch.tensor(input_ids).to(self.device)
            text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
            text_embeddings = linear_layer(text_embeddings)
            text_embeddings = text_embeddings.clone().detach().to(self.device)

            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]

            # Pass the decoded image through the model
            model_input = transform(decoded_image).unsqueeze(0).to(self.device)
            model_input = model_input[0].cpu().permute(1, 2, 0).numpy()
            outputs = self.model(model_input)["out"]
            pred_mask = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize((width, height),
                                                                                   resample=Image.NEAREST)
            pred_mask_resized = np.array(pred_mask_resized)

            decoded_image = decoded_image.resize((width, height), Image.BILINEAR)

            return decoded_image, pred_mask_resized

