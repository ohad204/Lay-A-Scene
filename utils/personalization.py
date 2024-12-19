"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import itertools
import logging
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import datasets
import diffusers
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler,
)
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusers.models.cross_attention import CrossAttention
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami
from utils.DIFT import DiftDiffusionPipeline
from utils.ptp_utils import AttentionStore
from torch.utils.data import Dataset

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class arguments():
    def __init__(self):
        # Model Options
        self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
        self.revision = None
        self.tokenizer_name = None
        self.concepts_list = 'shapes/assests/concept_list_hospital_2.json'
        self.instance_data_dir = "examples/creature"
        self.class_data_dir = 'inputs/data_dir'
        self.class_prompt = "a photo at the beach"
        self.with_prior_preservation = True
        self.prior_loss_weight = 1.0
        self.num_class_images = 100

        # Training Options
        self.output_dir = "outputs"
        self.seed = None
        self.resolution = 512
        self.center_crop = False
        self.train_text_encoder = True
        self.train_batch_size = 1
        self.sample_batch_size = 4
        self.num_train_epochs = 1
        self.phase1_train_steps = 400
        self.phase2_train_steps = 400
        self.checkpointing_steps = 5000
        self.resume_from_checkpoint = None
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = False
        self.learning_rate = 2e-6
        self.initial_learning_rate = 5e-4
        self.scale_lr = False
        self.lr_scheduler = "constant"
        self.lr_warmup_steps = 0
        self.lr_num_cycles = 1
        self.lr_power = 1.0
        self.use_8bit_adam = False
        self.dataloader_num_workers = 0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        self.hub_token = None
        self.hub_model_id = None
        self.logging_dir = "logs"
        self.allow_tf32 = False
        self.report_to = "tensorboard"
        self.mixed_precision = "fp16"
        self.prior_generation_precision = None
        self.local_rank = -1
        self.enable_xformers_memory_efficient_attention = False
        self.set_grads_to_none = False
        self.lambda_attention = 1e-2
        self.img_log_steps = 200
        self.num_of_assets = 1
        self.initializer_tokens = ['furniture']
        self.placeholder_token = "<asset>"
        self.apply_masked_loss = True
        self.log_checkpoints = True


def parse_args():
    args = arguments()

    assert len(args.initializer_tokens) == 0 or len(args.initializer_tokens) == args.num_of_assets
    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            training_data_dir,
            placeholder_tokens,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
            flip_p=0.5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        self.concepts_list = concepts_list
        self.background = imageio.imread('utils/gray_blue_512_512.png')

        # TEMP: READ FROM FILE
        COMBINATIONS_DIR = training_data_dir
        images = os.listdir(COMBINATIONS_DIR)
        self.train_images_path = [os.path.join(COMBINATIONS_DIR, n) for n in images]

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.placeholder_tokens = placeholder_tokens
        self._length = 1

        self.class_data_root = Path(class_data_root)
        self.class_data_root.mkdir(parents=True, exist_ok=True)
        self.class_images_path = list(self.class_data_root.iterdir())
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self._length)
        self.class_prompt = class_prompt

        max_length = max(len(self.train_images_path), len(self.class_images_path))
        self.train_images_path += random.choices(self.train_images_path, k=max_length - len(self.train_images_path))
        self.class_images_path += random.choices(self.class_images_path, k=max_length - len(self.class_images_path))
        random.shuffle(self.train_images_path)
        random.shuffle(self.class_images_path)
        self.random_pairs = [(self.train_images_path[i], self.class_images_path[i]) for i in range(max_length)]

        self.train_length = len(self.random_pairs)

    def __len__(self):
        return self.train_length

    def __getitem__(self, index):
        example = {}
        train_path, class_path = self.random_pairs[index]
        img_path = os.path.join(train_path, 'img.png')

        num_of_tokens = random.randrange(1, len(self.placeholder_tokens) + 1)
        tokens_ids_to_use = random.sample(
            range(len(self.placeholder_tokens)), k=num_of_tokens
        )
        tokens_to_use = [self.placeholder_tokens[tkn_i] for tkn_i in tokens_ids_to_use]
        example["token_ids"] = torch.tensor(tokens_ids_to_use)
        prompt = "a photo of " + " and ".join(tokens_to_use)

        img = imageio.imread(img_path)
        img[img[..., 3] < 50, :3] = self.background[img[..., 3] < 50]
        example['instance_images'] = self.image_transforms(img[..., :3])
        instance_masks = []
        for token_id in example['token_ids']:
            current_mask_path = os.path.join(train_path, f'mask{token_id}.png')
            current_mask = imageio.imread(current_mask_path)
            current_mask = self.mask_transforms(current_mask)
            instance_masks.append(current_mask)
        example['instance_masks'] = torch.stack(instance_masks)

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]

    if with_prior_preservation:
        input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
        pixel_values = [example["class_images"] for example in examples] + pixel_values

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    masks = torch.stack(masks)
    token_ids = torch.stack(token_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_masks": masks,
        "token_ids": token_ids,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(
        model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


class SpatialDreambooth:
    def __init__(self, concepts_list, experiment_name, training_data_dir, class_prompt="a photo at the beach",
                 load_ckpt=False,
                 save_path=None,
                 device='cuda'):
        self.args = parse_args()
        self.args.concepts_list = concepts_list
        self.args.num_of_assets = len(self.args.concepts_list)
        max_size = max([c['x_size_cm'] for c in self.args.concepts_list.values()])
        self.object_loss_weight = [max_size / c['x_size_cm'] for c in self.args.concepts_list.values()]
        self.args.initializer_tokens = [c['class_prompt'] for c in self.args.concepts_list.values()]
        self.load_ckpt = load_ckpt
        self.device = device
        self.save_path = save_path
        self.args.training_data_dir = training_data_dir

        self.args.class_data_dir = 'inputs/class_data/' + experiment_name
        self.args.output_dir = 'outputs/' + experiment_name

        objects_prompt = ' next to '.join([v['class_prompt'] for k, v in self.args.concepts_list.items()])
        self.args.class_prompt = class_prompt + f' with a {objects_prompt}'

        if self.load_ckpt:
            if os.path.exists(os.path.join(self.save_path, 'text_encoder')):
                self.pipeline = DiftDiffusionPipeline.from_pretrained(
                    self.save_path,
                )
                self.pipeline.scheduler = DDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                )
                self.pipeline.to(self.device)
            else:
                self.main()
        else:
            self.main()

    def get_pipeline(self):
        return self.pipeline

    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.pipeline.unet),
                text_encoder=self.accelerator.unwrap_model(self.pipeline.text_encoder),
                tokenizer=self.pipeline.tokenizer,
                revision=self.args.revision,
            )
            pipeline.save_pretrained(path)

    def main(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            logging_dir=logging_dir,
        )

        if (
                self.args.train_text_encoder
                and self.args.gradient_accumulation_steps > 1
                and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Generate class images if prior preservation is enabled.
        if self.args.with_prior_preservation:
            class_images_dir = Path(self.args.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.args.num_class_images:
                torch_dtype = (
                    torch.float16
                    if self.accelerator.device.type == "cuda"
                    else torch.float32
                )
                if self.args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif self.args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif self.args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=self.args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=self.args.sample_batch_size
                )

                sample_dataloader = self.accelerator.prepare(sample_dataloader)
                pipeline.to(self.accelerator.device)

                for example in tqdm(
                        sample_dataloader,
                        desc="Generating class images",
                        disable=not self.accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                                class_images_dir
                                / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.pipeline = DiftDiffusionPipeline.from_pretrained(self.args.pretrained_model_name_or_path,
                                                              use_auth_token=True,
                                                              )

        # Load scheduler and models
        self.pipeline.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler",
        )

        # Load the tokenizer
        if self.args.tokenizer_name:
            self.pipeline.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False,
            )
        elif self.args.pretrained_model_name_or_path:
            self.pipeline.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # Add meshes tokens to tokenizer
        self.placeholder_tokens = [
            self.args.placeholder_token.replace(">", f"{idx}>")
            for idx in range(self.args.num_of_assets)
        ]
        num_added_tokens = self.pipeline.tokenizer.add_tokens(self.placeholder_tokens)
        assert num_added_tokens == self.args.num_of_assets
        self.placeholder_token_ids = self.pipeline.tokenizer.convert_tokens_to_ids(
            self.placeholder_tokens
        )
        self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))
        self.args.instance_prompt = "a photo of " + " next to ".join(
            self.placeholder_tokens
        )

        if len(self.args.initializer_tokens) > 0:
            # Use initializer tokens
            token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
            for tkn_idx, initializer_token in enumerate(self.args.initializer_tokens):
                curr_token_ids = self.pipeline.tokenizer.encode(
                    initializer_token, add_special_tokens=False
                )
                token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[
                    curr_token_ids[0]
                ]
        else:
            # Initialize new tokens randomly
            token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.args.num_of_assets:] = token_embeds[
                                                      -3 * self.args.num_of_assets: -2 * self.args.num_of_assets
                                                      ]

        # Set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.pipeline.text_encoder.text_model.encoder.requires_grad_(False)
        self.pipeline.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.pipeline.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.pipeline.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.pipeline.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.pipeline.text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                    self.args.learning_rate
                    * self.args.gradient_accumulation_steps
                    * self.args.train_batch_size
                    * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        params_to_optimize = self.pipeline.text_encoder.get_input_embeddings().parameters()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            concepts_list=self.args.concepts_list,
            training_data_dir=self.args.training_data_dir,
            placeholder_tokens=self.placeholder_tokens,
            class_data_root=self.args.class_data_dir
            if self.args.with_prior_preservation
            else None,
            class_prompt=self.args.class_prompt,
            tokenizer=self.pipeline.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, self.args.with_prior_preservation
            ),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                    self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
                             * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
                               * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            self.pipeline.unet,
            self.pipeline.text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.pipeline.unet, self.pipeline.text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.pipeline.vae.to(self.accelerator.device)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.pipeline.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.pipeline.unet).dtype}. {low_precision_error_string}"
            )

        if (
                self.args.train_text_encoder
                and self.accelerator.unwrap_model(self.pipeline.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.pipeline.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                    self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if len(self.args.initializer_tokens) > 0:
            # Only for logging
            self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth")  # , config=vars(self.args))

        # Train
        # total_batch_size = (
        #         self.args.train_batch_size
        #         * self.accelerator.num_processes
        #         * self.args.gradient_accumulation_steps
        # )

        global_step = 0
        first_epoch = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.pipeline.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

        # Create attention controller
        self.original_attn_processors = self.pipeline.unet.attn_processors
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.pipeline.unet.train()
            if self.args.train_text_encoder:
                self.pipeline.text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                if self.args.phase1_train_steps == global_step:
                    self.pipeline.unet.requires_grad_(True)
                    if self.args.train_text_encoder:
                        self.pipeline.text_encoder.requires_grad_(True)
                    unet_params = self.pipeline.unet.parameters()

                    params_to_optimize = (
                        itertools.chain(unet_params, self.pipeline.text_encoder.parameters())
                        if self.args.train_text_encoder
                        else itertools.chain(
                            unet_params,
                            self.pipeline.text_encoder.get_input_embeddings().parameters(),
                        )
                    )
                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.lr_warmup_steps
                                         * self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps
                                           * self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )
                    optimizer, lr_scheduler = self.accelerator.prepare(
                        optimizer, lr_scheduler
                    )

                logs = {}

                # Skip steps until we reach the resumed step
                if (
                        self.args.resume_from_checkpoint
                        and epoch == first_epoch
                        and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.pipeline.unet):
                    # Convert images to latent space
                    latents = self.pipeline.vae.encode(
                        batch["pixel_values"]  # .to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.pipeline.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.pipeline.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.pipeline.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = self.pipeline.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if self.pipeline.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.pipeline.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.pipeline.noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.pipeline.noise_scheduler.config.prediction_type}"
                        )

                    if self.args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
                        target_prior, target = torch.chunk(target, 2, dim=0)

                        if self.args.apply_masked_loss:
                            max_masks = torch.max(
                                batch["instance_masks"], axis=1
                            ).values
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(64, 64)
                            )
                            model_pred = model_pred * downsampled_mask
                            target = target * downsampled_mask

                        # Compute instance loss
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                        # Compute prior loss
                        prior_loss = F.mse_loss(
                            model_pred_prior.float(),
                            target_prior.float(),
                            reduction="mean",
                        )

                        # Add the prior loss to the instance loss.
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        if self.args.apply_masked_loss:
                            max_masks = torch.max(
                                batch["instance_masks"], axis=1
                            ).values
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(64, 64)
                            )
                            model_pred = model_pred * downsampled_mask
                            target = target * downsampled_mask
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                    # Attention loss
                    if self.args.lambda_attention != 0:
                        attn_loss = 0
                        for batch_idx in range(self.args.train_batch_size):
                            GT_masks = F.interpolate(
                                input=batch["instance_masks"][batch_idx], size=(32, 32)
                            )
                            agg_attn = self.aggregate_attention(
                                res=32,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=batch_idx,
                            )
                            curr_cond_batch_idx = self.args.train_batch_size + batch_idx

                            for mask_id in range(len(GT_masks)):
                                curr_placeholder_token_id = self.placeholder_token_ids[
                                    batch["token_ids"][batch_idx][mask_id]
                                ]

                                asset_idx = (
                                    (
                                            batch["input_ids"][curr_cond_batch_idx]
                                            == curr_placeholder_token_id
                                    )
                                    .nonzero()
                                    .item()
                                )
                                asset_attn_mask = agg_attn[..., asset_idx]
                                asset_attn_mask = (
                                        asset_attn_mask / asset_attn_mask.max()
                                )
                                attn_loss += F.mse_loss(
                                    GT_masks[mask_id, 0].float(),
                                    asset_attn_mask.float(),
                                    reduction="mean",
                                )

                        attn_loss = self.args.lambda_attention * (
                                attn_loss / self.args.train_batch_size
                        )
                        logs["attn_loss"] = attn_loss.detach().item()
                        loss += attn_loss

                    self.accelerator.backward(loss)

                    # No need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.pipeline.unet.parameters(), self.pipeline.text_encoder.parameters()
                            )
                            if self.args.train_text_encoder
                            else self.pipeline.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.args.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    if global_step < self.args.phase1_train_steps:
                        # Let's make sure we don't update any embedding weights besides the newly added token
                        with torch.no_grad():
                            self.accelerator.unwrap_model(
                                self.pipeline.text_encoder
                            ).get_input_embeddings().weight[
                            : -self.args.num_of_assets
                            ] = orig_embeds_params[
                                : -self.args.num_of_assets
                                ]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

        # clear GPU memory
        self.controller.cur_step = 0
        self.controller.attention_store = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.load_ckpt:
            self.save_pipeline(self.save_path)
        self.accelerator.end_training()

        # we don't need the attention processors anymore
        self.pipeline.unet.set_attn_processor(self.original_attn_processors)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.eval()
        self.pipeline.text_encoder.eval()

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.pipeline.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipeline.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipeline.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.pipeline.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
            self, res: int, from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    @torch.no_grad()
    def perform_full_inference(self, path, guidance_scale=7.5):
        self.pipeline.unet.eval()
        self.pipeline.text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.pipeline.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.pipeline.tokenizer(
            [self.args.instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.pipeline.text_encoder(input_ids)[0]
        uncond_embeddings = self.pipeline.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.pipeline.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.pipeline.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.pipeline.unet.train()
        if self.args.train_text_encoder:
            self.pipeline.text_encoder.train()

        Image.fromarray(images[0]).save(path)

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.pipeline.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, self.pipeline.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        vis.save(path)


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
            self,
            attn: CrossAttention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)
        torch.cuda.empty_cache()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


if __name__ == "__main__":
    SpatialDreambooth()
