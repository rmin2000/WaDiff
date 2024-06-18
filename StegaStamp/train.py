import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default='../../../../data/coco/val2014', help="Directory with image dataset."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--output_dir", type=str, default='./', help="Directory to save results."
)
parser.add_argument("--data_size", type=int, default=5000, help="Sample number for training.")

parser.add_argument(
    "--bit_length",
    type=int,
    default=48,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=512,
    help="Height and width of square images.",
)
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--dataset_name", type=str, default='coco', help="Dataset name.")
parser.add_argument("--cuda", type=int, default=0)

parser.add_argument(
    "--l2_loss_await",
    help="Train without L2 loss for the first x iterations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--l2_loss_weight",
    type=float,
    default=10,
    help="L2 loss weight for image fidelity.",
)
parser.add_argument(
    "--l2_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase L2 loss weight over x iterations.",
)

parser.add_argument(
    "--BCE_loss_weight",
    type=float,
    default=1,
    help="BCE loss weight for fingerprint reconstruction.",
)

args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam, SGD
import torch.nn.functional as F
import models
import random

now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H:%M:%S")
EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"

LOGS_PATH = os.path.join(args.output_dir, f"logs/{args.dataset_name}_{EXP_NAME}")
CHECKPOINTS_PATH = os.path.join(args.output_dir, f"logs/{args.dataset_name}_{EXP_NAME}/checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, f"logs/{args.dataset_name}_{EXP_NAME}/saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)


def generate_random_fingerprints(bit_length, batch_size=4):
    z = torch.zeros((batch_size, bit_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir    
        self.filenames = glob.glob(os.path.join(data_dir, "*.jpg"))
              
        # self.filenames = sorted(self.filenames)
        random.shuffle(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    SECRET_SIZE = args.bit_length

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_RESOLUTION),
            transforms.CenterCrop(IMAGE_RESOLUTION),
            transforms.ToTensor(),
        ]
    )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    dataset = Subset(dataset, np.random.choice(len(dataset), args.data_size, replace=False))
    print(f"Finished. Loading took {time() - s:.2f}s Datset size: {len(dataset)}")


def main():

    device = torch.device(args.cuda)

    load_data()
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = True
    for param in decoder.parameters():
        param.requires_grad = True
    
    decoder_encoder_optim = Adam(params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr)

    global_step = 0
    steps_since_l2_loss_activated = -1
    encoder.train()
    decoder.train()
    for i_epoch in range(args.num_epochs):
        
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        for images in tqdm(dataloader):
            global_step += 1

            batch_size = min(args.batch_size, images.size(0))
            fingerprints = generate_random_fingerprints(args.bit_length, batch_size)

            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)

            fingerprinted_images = encoder(fingerprints, clean_images)
            # residual = fingerprinted_images - clean_images

         
            # print(fingerprinted_images.min(), fingerprinted_images.max())
           

            decoder_output = decoder(fingerprinted_images)
            
            
            # print(l2_loss_weight, BCE_loss_weight)
            criterion = nn.MSELoss()
            l2_loss = criterion(fingerprinted_images, clean_images)

            criterion = nn.BCEWithLogitsLoss()
            # BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))
            BCE_loss = F.binary_cross_entropy_with_logits(decoder_output.view(-1)*10, fingerprints.view(-1), reduction='mean')

            loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1
            print("Total loss", loss, global_step),
            print("BCE_loss", BCE_loss, global_step),
            # Logging
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step),
                print("Bitwise accuracy {}".format(bitwise_accuracy))
                print("Total loss", loss, global_step),
                print("BCE_loss", BCE_loss, global_step),
               
                
                save_image(
                    fingerprinted_images[:15], # Fast to display
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )

                writer.add_scalar(
                    "loss_weights/l2_loss_weight", l2_loss_weight, global_step
                )
                writer.add_scalar(
                    "loss_weights/BCE_loss_weight",
                    BCE_loss_weight,
                    global_step,
                )

            # checkpointing
            if global_step % 500 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, f"step_{global_step}_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, f"step_{global_step}_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, f"step_{global_step}_decoder.pth"),
                )
                
                f = open(join(CHECKPOINTS_PATH, "variables.txt"), "w")
                f.write(str(global_step))
                f.close()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
