import torch
import glob
import argparse
from StegaStamp import models
from torchvision import transforms
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path", type=str, default='./guided-diffusion/watermark_samples/', help="Directory of watermark examples."
)
parser.add_argument(
    "--bit_length", type=int, default=48, help="Length of watermark bits."
)
parser.add_argument(
    "--model_type", type=str, default='imagenet', choices=['imagenet', 'stable'], help="ImageNet Diffusion or Stable Diffusion"
)
parser.add_argument(
    "--checkpoint", type=str, default='./', help="Checkpoint of the watermark decoder"
)
parser.add_argument(
    "--device", type=int, default=0, help="GPU index"
)


args = parser.parse_args()



def trace(image_path, decoder):
    count_pool_1e4, count_pool_1e5, count_pool_1e6 = 0, 0, 0
    
    # Load pre-defined watermarks
    user_pool_1e4 = np.load('watermark_pool/1e4.npy')
    user_pool_1e5 = np.load('watermark_pool/1e5.npy')
    user_pool_1e6 = np.load('watermark_pool/1e6.npy')
    image_path_list = glob.glob(image_path + '*/*.png')

    
    for path in image_path_list:
        img = transforms(Image.open(path))
        user_index = int(path.split('/')[-2])
        
        fingerprints_predicted = (decoder(img) > 0).float()
        if np.argmin(np.abs(fingerprints_predicted - user_pool_1e4).sum(0)) == user_index:
            count_pool_1e4 += 1
        if np.argmin(np.abs(fingerprints_predicted - user_pool_1e5).sum(0)) == user_index:
            count_pool_1e5 += 1
        if np.argmin(np.abs(fingerprints_predicted - user_pool_1e6).sum(0)) == user_index:
            count_pool_1e6 += 1
        
    return count_pool_1e4/1e4, count_pool_1e5/1e5, count_pool_1e6/1e6
        
    
    

if __name__ == "__main__":

    # Create watermark decoder
    decoder = models.StegaStampDecoder(
        256 if args.model_type == imagenet else 512,
        3,
        args.bit_length,
    )
    decoder.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    decoder.to(args.device)

    # Perform tracing
    trace(args.image_path, decoder)