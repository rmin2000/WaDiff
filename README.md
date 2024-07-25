### A Watermark-Conditioned Diffusion Model for IP Protection (ECCV 2024)
This code is the official implementation of [A Watermark-Conditioned Diffusion Model for IP Protection](https://arxiv.org/abs/2403.10893).

----
<div align=center><img src=pics/framework.png  width="80%" height="60%"></div>

### Abstract

The ethical need to protect AI-generated content has been a significant concern in recent years. While existing watermarking strategies have demonstrated success in detecting synthetic content (detection), there has been limited exploration in identifying the users responsible for generating these outputs from a single model (owner identification). In this paper, we focus on both practical scenarios and propose a unified watermarking framework for content copyright protection within the context of diffusion models. Specifically, we consider two parties: the model provider, who grants public access to a diffusion model via an API, and the users, who can solely query the model API and generate images in a black-box manner. Our task is to embed hidden information into the generated contents, which facilitates further detection and owner identification. To tackle this challenge, we propose a Watermark-conditioned Diffusion model called WaDiff, which manipulates the watermark as a conditioned input and incorporates fingerprinting into the generation process. All the generative outputs from our WaDiff carry user-specific information, which can be recovered by an image extractor and further facilitate forensic identification. Extensive experiments are conducted on two popular diffusion models, and we demonstrate that our method is effective and robust in both the detection and owner identification tasks. Meanwhile, our watermarking framework only exerts a negligible impact on the original generation and is more stealthy and efficient in comparison to existing watermarking strategies.

### Pipeline
#### Step 1: Pre-train Watermark Decoder

First, you need to pre-train the watermark encoder and decoder jointly. Go to the [StegaStamp](StegaStamp) folder and simply run:
```cmd
cd StegaStamp
sh train.sh
```
Note that directly running the script may not be successful as you need to specify the path of the training data ```--data_dir``` in your project. Besides, you can customize your experiments by adjusting hyperparameters such as the number of watermark bits ```--bit_length```, image resolution ```--image_resolution```, training epochs ```--num_epochs``` and GPU device ```--cuda```.

#### Step 2: Fine-tune Diffusion Model
Once you have finished the pre-training process, you can utilize the watermark decoder to guide the fine-tuning process of the diffusion model. For the ImageNet Diffusion model, you can run the following commands:
```cmd
cd ../guided-diffusion
sh train.sh
```
But before running the script, you need to configure properly, i.e. the path of the pre-trained decoder checkpoint ```--wm_decoder_path``` (from Step 1) and the path of the training data ```--data_dir``` in your project (mostly the same in Step 1), the number of watermark bits ```--wm_length```, the balance parameter $\alpha$ ```--alpha```, and the time threshold $\tau$ ```--threshold```. Besides, you need to download the pre-trained diffusion model [checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and put it into the ```models/``` folder.

#### Step 3: Generate Watermarked Images
After the fine-tuning step, you could use this watermark-conditioned diffusion model to generate watermarked images with the following commands:
```cmd
sh generate.sh
```
All generated images are saved in a default folder named ```saved_images``` (specified by ```--output_path```) and will be organized into individual subfolders indexed by the specific ID of the watermark. You could also change ```--batch_size``` to adjust the number of generated images within individual subfolders.

#### Step 4: Source Identification
Finally, run the following command to perform tracing:
```cmd
cd ..
python trace.py
```
Note that the ```--image_path``` indicates where you save watermarked images, which should be consistent with the ```--output_path``` specified in Step 3.

[Pin] I will keep on updating this repo. However, reproducing the project will require some time as the original code was accidentally deleted due to an unfortunate accident (the original server was taken back without making any backup). Nevertheless, I plan to reproduce this repo before the conference. Should you have any concerns related to our project, please contact me via rminaa@connect.ust.hk.
- [x] StegaStamp training code
- [x] Imagenet Diffusion
- [x] Generation code
- [x] Tracing Code
- [ ] Stable Diffusion

### Citation
The camera-ready version of our paper will be available soon.
```
@article{min2024watermark,
  title={A watermark-conditioned diffusion model for ip protection},
  author={Min, Rui and Li, Sen and Chen, Hongyang and Cheng, Minhao},
  journal={arXiv preprint arXiv:2403.10893},
  year={2024}
}
```

#### Our codes are heavily built upon [stable-diffusion](https://github.com/CompVis/stable-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion) and [WatermarkDM](https://github.com/yunqing-me/WatermarkDM).
