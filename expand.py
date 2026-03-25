import argparse
import json
import os
import random
import shutil
import itertools


from collections import defaultdict

import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from PIL import Image
from tqdm.auto import tqdm

from augmentation import (ConstantAugment, RandomNormalAugment,
                      RandomUniformAugment, SinusoidalAugment, XAugment)

AUG_MAP = {
    'ConstantAugment': ConstantAugment,
    'RandomNormalAugment': RandomNormalAugment,
    'RandomUniformAugment': RandomUniformAugment,
    'SinusoidalAugment': SinusoidalAugment
}

# Rotation augment: rotates tensor by a random angle in [-max_angle, max_angle]
class RotationAugment:
    def __init__(self, max_angle):
        self.max_angle = max_angle
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, x, *args, **kwargs):
        # pick magnitude between 10 and 20 degrees and randomly choose direction
        mag = random.uniform(10.0, 20.0)
        angle = mag if random.choice([True, False]) else -mag
        img = self.to_pil(x)
        img = img.rotate(angle, resample=Image.BICUBIC)
        return self.to_tensor(img)

# Register rotation in AUG_MAP so it can be referenced/created like other aug classes
AUG_MAP['Rotate'] = RotationAugment

def expand(augmentations, model, image_paths, image_size, expand_ratio, output_data_dir, mode):
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    for image_path in tqdm(image_paths, desc='expanding'):
        category = os.path.basename(os.path.dirname(image_path))
        save_dir = os.path.join(output_data_dir, 'train', category)
        os.makedirs(save_dir, exist_ok=True)

        # load & resize original
        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        x = to_tensor(image)

        if mode == 'self_image':
            # augmentations: dict[category] = (list_of_noise_tensors, list_of_strengths)
            for category, (xnoises, strengths) in augmentations.items():
                save_dir = os.path.join(output_data_dir, 'train', category)
                os.makedirs(save_dir, exist_ok=True)

                # Generate multiple augmentations per image based on expand_ratio
                for i1 in range(len(xnoises)):
                    base_x = xnoises[i1]
                    orig_path = noise_conf[category]['image_path'][i1]
                    name, ext = os.path.splitext(os.path.basename(orig_path))
                    
                    # Create expand_ratio number of augmentations for each base image
                    for idx in range(expand_ratio):
                        # Random noise partner for each augmentation
                        i2 = random.randrange(len(xnoises))
                        noise_x = xnoises[i2]
                        s = random.choice(strengths)
                        aug = XAugment(s, noise_x)

                        # Apply augment to the base tensor
                        x_out = aug(base_x, model, model)
                        img = to_image(x_out)

                        # Save augmented image
                        out_name = f"{name}_aug_{i1}_{idx}{ext}"
                        img.save(os.path.join(save_dir, out_name))
                        
                        if idx % 10 == 0:  # Progress indicator every 10 augmentations
                            print(f"Created {idx+1}/{expand_ratio} augmentations for image {i1+1}/{len(xnoises)} in {category}")

            # done—skip the rest
            return


        else:
            augs = random.choices(augmentations, k=expand_ratio)

        for idx, aug in enumerate(augs):
            if model == 'random_image' or mode == 'self_image':
                x_ = aug(x, model, model)
            else:
                x_ = aug(x, model)

            image_ = to_image(x_)
            image_.save(os.path.join(save_dir, f'aug_{idx}_{os.path.basename(image_path)}'))
            del image_


def copy_split(image_paths, output_data_dir, split):
    for image_path in tqdm(image_paths, desc=f'copying original {split}'):
        category = os.path.basename(os.path.dirname(image_path))
        save_dir = os.path.join(output_data_dir, split, category)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(image_path, save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='expansion script - self images')

    parser.add_argument('--model_repo', default='black-forest-labs/FLUX.1-schnell')
    parser.add_argument('--input_data_dir', required=True)
    parser.add_argument('--mode', choices=['random', 'random_image', 'self_image'], required=True)
    parser.add_argument('--noise_conf_json', required=True)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--expand_ratio', type=int, default=1)
    parser.add_argument('--output_data_dir', required=True)
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    print(f"{args.mode}: {args.noise_conf_json}")

    assert args.expand_ratio >= 1, f'codebase only supports expand_ratio>=1, got f{args.expand_ratio}'
    assert not os.path.exists(args.output_data_dir), f'output directory already exists: {args.output_data_dir}'
    print(f"Output: {args.output_data_dir}")

    # support loading a local saved checkpoint dir (diffusers save_pretrained) or a hub repo
    load_dtype = torch.float32 if args.full_precision else torch.float16
    if os.path.isdir(args.model_repo):
        # saved checkpoint likely has VAE files at the root
        model = AutoencoderKL.from_pretrained(args.model_repo, torch_dtype=load_dtype)
    else:
        # hub repos sometimes put the VAE under a "vae" subfolder; try that first then fallback
        try:
            model = AutoencoderKL.from_pretrained(args.model_repo, subfolder="vae", torch_dtype=load_dtype)
        except Exception:
            model = AutoencoderKL.from_pretrained(args.model_repo, torch_dtype=load_dtype)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    noise_conf = json.load(open(args.noise_conf_json))

    if args.mode == 'random':
        augmentations = []
        # Prefer explicit augment names defined in AUG_MAP if present in JSON.
        aug_keys_in_json = [k for k in noise_conf.keys() if k in AUG_MAP]
        if aug_keys_in_json:
            for aug_type in aug_keys_in_json:
                aug_cls = AUG_MAP[aug_type]
                strengths = noise_conf[aug_type]
                for strength in strengths:
                    augmentations.append(aug_cls(strength))
        else:
            # JSON appears to be organized by category (e.g. "alzheimers": {"image_path": [...], "strength": [...]})
            # Collect all unique strengths across categories and apply every AUG_MAP augment to them.
            strengths_set = set()
            for conf in noise_conf.values():
                if isinstance(conf, dict) and 'strength' in conf:
                    strengths_set.update(conf['strength'])
                elif isinstance(conf, list):
                    # support the case where noise_conf directly maps augment-names to list-of-strengths
                    strengths_set.update(conf)
            strengths = sorted(list(strengths_set))
            if not strengths:
                raise ValueError("No 'strength' values found in noise_conf for random mode.")
            for aug_type, aug_cls in AUG_MAP.items():
                for strength in strengths:
                    augmentations.append(aug_cls(strength))
    elif args.mode == 'random_image':
        augmentations = []
        to_tensor = transforms.ToTensor()
        for noise_image_path in noise_conf['image_path']:
            image = Image.open(noise_image_path).convert("RGB")
            image = image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            xnoise = to_tensor(image)
            for strength in noise_conf['strength']:        
                augmentations.append(XAugment(strength, xnoise))
    elif args.mode == 'self_image':
        # For self_image we just need the raw noise‐tensors and strength list
        augmentations = {}
        to_tensor = transforms.ToTensor()

        for category, conf in noise_conf.items():
            # load & tensor-ify all noise reference images
            xnoises = []
            for noise_path in conf['image_path']:
                img = (Image.open(noise_path)
                        .convert("RGB")
                        .resize((args.image_size, args.image_size),
                                Image.Resampling.LANCZOS))
                xnoises.append(to_tensor(img))

            # keep each category’s (list_of_tensors, list_of_strengths)
            augmentations[category] = (xnoises, conf['strength'])



    train_image_paths = [os.path.join(args.input_data_dir, 'train', category, train_image_filename) \
                         for category in os.listdir(os.path.join(args.input_data_dir, 'train')) \
                            for train_image_filename in os.listdir(os.path.join(args.input_data_dir, 'train', category))]
    validation_image_paths = [os.path.join(args.input_data_dir, 'validation', category, validation_image_filename) \
                         for category in os.listdir(os.path.join(args.input_data_dir, 'validation')) \
                            for validation_image_filename in os.listdir(os.path.join(args.input_data_dir, 'validation', category))]

    expand(augmentations, model, train_image_paths, args.image_size, args.expand_ratio, args.output_data_dir, args.mode)
    #copy original images
    copy_split(train_image_paths, args.output_data_dir, 'train')
    copy_split(validation_image_paths, args.output_data_dir, 'validation')
