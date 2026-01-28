# -*- coding: utf-8 -*-
'''
This script extracts features for composite retrieval with four image types:
subject, object, second object, and relation images.
'''

import os
import argparse
import logging
import json
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace(""", "\"").replace(""", "\"")
    return text


def enhance_ancient_text(text, text_type):
    """
    Enhance ancient Chinese stone relief descriptions with modern language
    """
    if not text or not text.strip():
        return text

    # Ancient to modern mappings
    ancient_to_modern = {
        # Characters and figures
        "人": "人物",
        "龙": "神龙",
        "女娲": "人首蛇身的女娲神",
        "神鸟": "飞鸟",
        "马": "马匹",
        "鱼": "鱼类",
        "青蛙": "青蛙",
        "鸟": "鸟类",
        "鼓": "鼓乐器",
        "长矛": "武器",
        "马车": "两匹马拉动的车",

        # Actions and relations
        "交谈": "正在交谈",
        "飞舞": "在空中飞舞",
        "浮于": "飘浮在",
        "迎战": "正在迎战",
        "敲鼓": "正在敲鼓",
        "朝拜": "正在朝拜",
        "战斗": "正在战斗",
        "游动": "在水中游动",
        "飞": "在空中飞翔",

        # Locations and contexts
        "池中": "池塘中",
        "空中": "天空中",
        "天上": "天空中",
        "图中": "画面中",
    }

    # Enhance based on text type
    enhanced_text = text

    # Apply general enhancements
    for ancient, modern in ancient_to_modern.items():
        enhanced_text = enhanced_text.replace(ancient, modern)

    # Type-specific enhancements
    if text_type == 'subject':
        # Add descriptive context for subjects
        if "人" in enhanced_text and "人物" not in enhanced_text:
            enhanced_text = enhanced_text.replace("人", "人物")
        if "龙" in enhanced_text and "神龙" not in enhanced_text:
            enhanced_text = enhanced_text.replace("龙", "神龙")
        if "女娲" in enhanced_text and "女娲神" not in enhanced_text:
            enhanced_text = enhanced_text.replace("女娲", "女娲神")

    elif text_type == 'object':
        # Add descriptive context for objects
        if "马" in enhanced_text and "马匹" not in enhanced_text:
            enhanced_text = enhanced_text.replace("马", "马匹")
        if "鼓" in enhanced_text and "鼓乐器" not in enhanced_text:
            enhanced_text = enhanced_text.replace("鼓", "鼓乐器")
        if "马车" in enhanced_text and "马车" not in enhanced_text:
            enhanced_text = enhanced_text.replace("马车", "马车")

    elif text_type == 'second':
        # Add descriptive context for secondary objects
        if "鱼" in enhanced_text and "鱼类" not in enhanced_text:
            enhanced_text = enhanced_text.replace("鱼", "鱼类")
        if "鸟" in enhanced_text and "鸟类" not in enhanced_text:
            enhanced_text = enhanced_text.replace("鸟", "鸟类")
        if "青蛙" in enhanced_text and "青蛙" not in enhanced_text:
            enhanced_text = enhanced_text.replace("青蛙", "青蛙")

    elif text_type == 'relation':
        # Add descriptive context for relations
        if "交谈" in enhanced_text and "正在交谈" not in enhanced_text:
            enhanced_text = enhanced_text.replace("交谈", "正在交谈")
        if "飞舞" in enhanced_text and "在空中飞舞" not in enhanced_text:
            enhanced_text = enhanced_text.replace("飞舞", "在空中飞舞")
        if "朝拜" in enhanced_text and "正在朝拜" not in enhanced_text:
            enhanced_text = enhanced_text.replace("朝拜", "正在朝拜")
        if "战斗" in enhanced_text and "正在进行战斗" not in enhanced_text:
            enhanced_text = enhanced_text.replace("战斗", "正在进行战斗")

    # Add contextual prefixes based on content
    if "人物" in enhanced_text or "神龙" in enhanced_text or "女娲神" in enhanced_text:
        if not enhanced_text.startswith("画面中的"):
            enhanced_text = f"画面中的{enhanced_text}"

    # Add descriptive suffixes for better understanding
    if text_type == 'relation' and enhanced_text and not any(
            word in enhanced_text for word in ["正在", "进行", "画面"]):
        enhanced_text = f"画面中{enhanced_text}"

    return enhanced_text


class CompositeImgDataset(Dataset):
    """Dataset for composite image features extraction"""

    def __init__(self, image_dir, image_type, resolution=224):
        self.image_dir = image_dir
        self.image_type = image_type
        self.image_paths = []

        # Get all image files in the directory
        if os.path.exists(image_dir):
            for filename in sorted(os.listdir(image_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(image_dir, filename))

        self.transform = self._build_transform(resolution)
        print(f"Found {len(self.image_paths)} {image_type} images in {image_dir}")

    def _build_transform(self, resolution):
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return Compose([
            Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = os.path.basename(image_path).split('.')[0]

        try:
            image = Image.open(image_path)
            image = self.transform(image)
            return int(image_id), image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return int(image_id), dummy_image


class CompositeTxtDataset(Dataset):
    """Dataset for text features extraction - extracts four separate text components"""

    def __init__(self, jsonl_file, text_type, max_txt_length=52, enhance_text=True):
        self.texts = []
        self.text_type = text_type
        self.max_txt_length = max_txt_length
        self.enhance_text = enhance_text

        with open(jsonl_file, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                text_id = obj['text_id']

                # Extract specific text component based on text_type
                if text_type == 'subject':
                    text = obj.get('subject', '')
                elif text_type == 'object':
                    text = obj.get('object', '')
                elif text_type == 'second':
                    text = obj.get('second', '')
                elif text_type == 'relation':
                    text = obj.get('relation', '')
                else:
                    text = ''

                # Only add non-empty texts
                if text.strip():
                    self.texts.append((text_id, text))

        print(f"Loaded {len(self.texts)} {text_type} text samples from {jsonl_file}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_id, text = self.texts[idx]
        # Enhance ancient text with modern language if enabled
        if self.enhance_text:
            enhanced_text = enhance_ancient_text(str(text), self.text_type)
        else:
            enhanced_text = str(text)
        text_tok = tokenize([_preprocess_text(enhanced_text)], context_length=self.max_txt_length)[0]
        return text_id, text_tok


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-image-feats', action="store_true", default=False,
                        help="Whether to extract image features.")
    parser.add_argument('--extract-text-feats', action="store_true", default=False,
                        help="Whether to extract text features.")
    parser.add_argument('--subject-dir', type=str, default="com/subject",
                        help="Path to subject images directory.")
    parser.add_argument('--object-dir', type=str, default="com/object",
                        help="Path to object images directory.")
    parser.add_argument('--second-object-dir', type=str, default="com/second object",
                        help="Path to second object images directory.")
    parser.add_argument('--relation-dir', type=str, default="com/relation",
                        help="Path to relation images directory.")
    parser.add_argument('--text-data', type=str, default="create.jsonl",
                        help="Path to text data file.")
    parser.add_argument('--output-dir', type=str, default="features",
                        help="Output directory for features.")
    parser.add_argument('--img-batch-size', type=int, default=32, help="Image batch size.")
    parser.add_argument('--text-batch-size', type=int, default=32, help="Text batch size.")
    parser.add_argument('--context-length', type=int, default=52,
                        help="The maximum length of input text.")
    parser.add_argument('--resume', default="clip_cn_vit-b-16.pt", type=str,
                        help="path to model checkpoint")
    parser.add_argument('--precision', choices=["amp", "fp16", "fp32"], default="amp",
                        help="Floating point precision.")
    parser.add_argument('--vision-model',
                        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
                        default="ViT-B-16", help="Name of the vision backbone to use.")
    parser.add_argument('--text-model',
                        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
                        default="RoBERTa-wwm-ext-base-chinese", help="Name of the text backbone to use.")
    parser.add_argument('--enhance-text', action="store_true", default=True,
                        help="Whether to enhance ancient text with modern language.")
    parser.add_argument('--debug', default=False, action="store_true",
                        help="If true, more information is logged.")

    return parser.parse_args()


def extract_image_features(model, dataset, output_path, batch_size, device):
    """Extract image features and save to file"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    features = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image features"):
            image_ids, images = batch
            images = images.to(device)

            image_features = model(images, None)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()

            for image_id, feature in zip(image_ids, image_features):
                # Convert image_id to string to avoid JSON serialization issues
                features[str(image_id)] = feature.tolist()

    # Save features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(features)} image features to {output_path}")
    return features


def extract_text_features(model, dataset, output_path, batch_size, device):
    """Extract text features and save to file"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    features = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting text features"):
            text_ids, texts = batch
            texts = texts.to(device)

            text_features = model(None, texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()

            for text_id, feature in zip(text_ids, text_features):
                # Convert text_id to string to avoid JSON serialization issues
                features[str(text_id)] = feature.tolist()

    # Save features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(features)} text features to {output_path}")
    return features


def main():
    args = parse_args()

    assert args.extract_image_feats or args.extract_text_feats, \
        "--extract-image-feats and --extract-text-feats cannot both be False!"

    # Log params
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    vision_model_config_file = Path(
        __file__).parent / f"cn_clip/clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    text_model_config_file = Path(
        __file__).parent / f"cn_clip/clip/model_configs/{args.text_model.replace('/', '-')}.json"

    print('Loading vision model config from', vision_model_config_file)
    print('Loading text model config from', text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info)
    convert_weights(model)

    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.to(device)
    if args.precision == "fp16":
        convert_weights(model)

    # Load checkpoint
    print(f"Loading model checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract image features for each type
    if args.extract_image_feats:
        image_types = ['subject', 'object', 'second_object', 'relation']
        image_dirs = [args.subject_dir, args.object_dir, args.second_object_dir, args.relation_dir]

        for img_type, img_dir in zip(image_types, image_dirs):
            if os.path.exists(img_dir):
                print(f"\nExtracting {img_type} image features...")
                dataset = CompositeImgDataset(img_dir, img_type)
                output_path = os.path.join(args.output_dir, f"{img_type}_features.json")
                extract_image_features(model, dataset, output_path, args.img_batch_size, device)
            else:
                print(f"Warning: {img_dir} does not exist, skipping {img_type} features")

    # Extract text features for each component
    if args.extract_text_feats:
        text_types = ['subject', 'object', 'second', 'relation']

        for text_type in text_types:
            print(f"\nExtracting {text_type} text features...")
            dataset = CompositeTxtDataset(args.text_data, text_type, args.context_length, args.enhance_text)
            if len(dataset) > 0:  # Only extract if there are texts of this type
                output_path = os.path.join(args.output_dir, f"{text_type}_text_features.json")
                extract_text_features(model, dataset, output_path, args.text_batch_size, device)
            else:
                print(f"No {text_type} texts found, skipping...")

    print("Feature extraction completed!")


if __name__ == "__main__":
    main()
