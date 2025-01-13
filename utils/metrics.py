import os
import json
import torch
import numpy as np
import pyiqa
import wandb

from tqdm.auto import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.multimodal import CLIPImageQualityAssessment


def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img / 255.0
    return img


def clip_score_and_iqa(image_folder, text, out_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    prompt_metric = ("quality", "colorfullness", "sharpness")
    clipiqa_model = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-base-patch16", prompts=prompt_metric, data_range=1.0)

    images = [Image.open(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if "png" in f or "jpg" in f]
    if text.endswith('.txt'):
        with open(text, 'r') as f:
            prompt = f.readline()
    else:
        prompt = text

    scores = torch.zeros((len(prompt_metric), len(images)), device=clipiqa_model.device)
    clip_scores = torch.zeros(len(images), device=model.device)

    pbar = tqdm(images, desc="Calc CLIP Score and CLIP IQA")
    for idx, image in enumerate(pbar):
        img_torch = pil_to_torch(image, model.device, normalize=False)
        inputs = processor(text=[prompt], images=img_torch, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        clip_scores[idx] = outputs.logits_per_image.detach()    
        for prompt_idx in range(len(prompt_metric)):
            scores[prompt_idx][idx] = clipiqa_model(img_torch.unsqueeze(dim=0))[prompt_metric[prompt_idx]].detach()

    wandb.log({
        'clip_score': clip_scores.mean().cpu().numpy().item(),
        'clipiqa-quality': scores[0].mean().cpu().numpy().item(),
        'clipiqa-colorful': scores[1].mean().cpu().numpy().item(),
        'clipiqa-sharp': scores[2].mean().cpu().numpy().item()
    })

    print("CLIP Score", clip_scores.mean().cpu().numpy())
    print("CLIP IQA")
    print("quality", scores[0].mean().cpu().numpy())
    print("colorful", scores[1].mean().cpu().numpy())
    print("sharp", scores[2].mean().cpu().numpy())


def brisque_and_niqe_score(image_folder, out_path):       # pyiqa
    images = [Image.open(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if "png" in f or "jpg" in f]
    images_tensor = []
    for image in images:
        image_t = pil_to_torch(image, "cpu", normalize=True)  
        images_tensor.append(image_t)
    stack_images_tensor = torch.stack(images_tensor, dim=0)   


    brisque_metric = pyiqa.create_metric('brisque')
    brisque_scores = brisque_metric(stack_images_tensor).tolist()
    print("BRISQUE", np.mean(brisque_scores))

    niqe_metric = pyiqa.create_metric('niqe')
    niqe_scores = niqe_metric(stack_images_tensor).tolist()
    print("NIQE", np.mean(niqe_scores))

    wandb.log({
        'brisque': np.mean(brisque_scores),
        'niqe': np.mean(niqe_scores)
    })
