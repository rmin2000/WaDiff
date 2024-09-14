"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)




def save_images(images, output_path):
    for i in range(images.shape[0]):
        image_array = np.uint8(images[i])
        image = Image.fromarray(image_array)
        image.save(os.path.join(output_path, f'image_{i}.png'))

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    if not os.path.exists(f'watermark_pool/{args.wm_length}_1e4.npy'):
        os.makedirs('watermark_pool', exist_ok=True)
        np.save(f'watermark_pool/{args.wm_length}_1e4.npy', np.random.randint(0, 2, size=(int(1e4), args.wm_length)))
        np.save(f'watermark_pool/{args.wm_length}_1e5.npy', np.random.randint(0, 2, size=(int(1e5), args.wm_length)))
        np.save(f'watermark_pool/{args.wm_length}_1e6.npy', np.random.randint(0, 2, size=(int(1e6), args.wm_length)))
        
    keys = np.load(f'watermark_pool/{args.wm_length}_1e4.npy')[:1000]
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), wm_length=args.wm_length
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    
    # while len(all_images) * args.batch_size < args.num_samples:
    for idx, key in enumerate(keys):
        if args.wm_length < 0:
            key = None
        else:
            key = th.from_numpy(key).to(dist_util.dev()).float()
            key = key.repeat(args.batch_size, 1)

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            key=key,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images = [sample.cpu().numpy() for sample in gathered_samples]
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
           
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        os.makedirs(os.path.join(args.output_path, f'{idx}/'), exist_ok=True)
        save_images(arr, os.path.join(args.output_path, f'{idx}/'))

        logger.log(f"saving to {os.path.join(args.output_path, f'{idx}/')}")
    

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=16,
        use_ddim=False,
        model_path="",
        output_path='saved_images/',
        wm_length=48,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
