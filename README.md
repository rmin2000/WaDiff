### A Watermark-Conditioned Diffusion Model for IP Protection (ECCV 2024)
This code is the official implementation of [A Watermark-Conditioned Diffusion Model for IP Protection](https://arxiv.org/abs/2403.10893).

----
<div align=center><img src=pics/framework.png  width="80%" height="60%"></div>

### Abstract

The ethical need to protect AI-generated content has been a significant concern in recent years. While existing watermarking strategies have demonstrated success in detecting synthetic content (detection), there has been limited exploration in identifying the users responsible for generating these outputs from a single model (owner identification). In this paper, we focus on both practical scenarios and propose a unified watermarking framework for content copyright protection within the context of diffusion models. Specifically, we consider two parties: the model provider, who grants public access to a diffusion model via an API, and the users, who can solely query the model API and generate images in a black-box manner. Our task is to embed hidden information into the generated contents, which facilitates further detection and owner identification. To tackle this challenge, we propose a Watermark-conditioned Diffusion model called WaDiff, which manipulates the watermark as a conditioned input and incorporates fingerprinting into the generation process. All the generative outputs from our WaDiff carry user-specific information, which can be recovered by an image extractor and further facilitate forensic identification. Extensive experiments are conducted on two popular diffusion models, and we demonstrate that our method is effective and robust in both the detection and owner identification tasks. Meanwhile, our watermarking framework only exerts a negligible impact on the original generation and is more stealthy and efficient in comparison to existing watermarking strategies.

### Pipeline

First, you need to pre-train the watermark encoder and decoder. Go to the [StegaStamp](StegaStamp) folder and simply run:
```cmd
sh train.sh
```


[Pin] I will keep on updating this repo. However, reproducing the project will require some time as the original code was accidentally deleted due to an unfortunate accident (the original server was taken back without making any backup). Nevertheless, I plan to reproduce this repo before the conference. Should you have any concerns related to our project, please contact me via rminaa@connect.ust.hk.
- [x] StegaStamp training code (coco only)
- [x] Imagenet Diffusion (mostly done)
- [ ] Tracing Code (progressing)
- [ ] Stable Diffusion
