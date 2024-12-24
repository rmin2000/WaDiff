import torch
import glob
import argparse
from StegaStamp import models
from torchvision import transforms
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path", type=str, default='./guided-diffusion/saved_images/', help="Directory of watermark examples."
)
parser.add_argument(
    "--bit_length", type=int, default=48, help="Length of watermark bits."
)
parser.add_argument(
    "--model_type", type=str, default='imagenet', choices=['imagenet', 'stable'], help="ImageNet Diffusion or Stable Diffusion."
)
parser.add_argument(
    "--checkpoint", type=str, default='./', help="Checkpoint of the watermark decoder."
)
parser.add_argument(
    "--device", type=int, default=0, help="GPU index."
)
parser.add_argument(
    "--detection_thre", type=float, default=0.8, help="Bit threshold for detection."
)


args = parser.parse_args()



def trace(image_path, decoder, detection_thre, device):

    count_pool_1e4, count_pool_1e5, count_pool_1e6, count_detection = 0, 0, 0, 0
    decoder.to(args.device)

    # Load pre-defined watermarks and watermarked images
    user_pool_1e4 = np.load(f'watermark_pool/{bit_length}_1e4.npy')
    user_pool_1e5 = np.load(f'watermark_pool/{bit_length}_1e5.npy')
    user_pool_1e6 = np.load(f'watermark_pool/{bit_length}_1e6.npy')

    image_path_list = glob.glob(image_path + '*/*.png')

    
    for path in image_path_list:
        img = transforms.ToTensor()(Image.open(path)).to(device)
        user_index = int(path.split('/')[-2])
        
        fingerprints_predicted = (decoder(img) > 0).float().cpu().numpy()
        
        if 1 - np.abs(fingerprints_predicted - user_pool_1e4[user_index]).sum(0) / len(fingerprints_predicted) > detection_thre
            count_detection += 1
        if np.argmin(np.abs(fingerprints_predicted - user_pool_1e4).sum(0)) == user_index:
            count_pool_1e4 += 1
        if np.argmin(np.abs(fingerprints_predicted - user_pool_1e5).sum(0)) == user_index:
            count_pool_1e5 += 1
        if np.argmin(np.abs(fingerprints_predicted - user_pool_1e6).sum(0)) == user_index:
            count_pool_1e6 += 1
        
    return {'trace1e4': count_pool_1e4/1e4, 'trace1e5': count_pool_1e5/1e5, 'trace1e6': count_pool_1e6/1e6, 'detection_acc': count_detection/len(image_path_list)}
        
    
    

if __name__ == "__main__":

    # Create watermark decoder
    decoder = models.StegaStampDecoder(
        256 if args.model_type == 'imagenet' else 512,
        3,
        args.bit_length,
    )
    decoder.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    
    # Perform tracing
    result = trace(args.image_path, decoder, args.detection_thre, args.device)
    print(result['trace1e4'], result['trace1e5'], result['trace1e6'], result['detection_acc'])
