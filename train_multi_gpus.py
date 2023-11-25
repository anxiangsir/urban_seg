import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import distributed, nn
from torch.nn import GELU, Conv2d, ConvTranspose2d, LayerNorm, ReLU, Upsample
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import Dice

from vit import load_model_and_transform

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
rank = int(os.getenv("RANK", "0"))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--frequent', type=int, default=50)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--model', type=str, default='ViT-L/14',
                    choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
parser.add_argument('--total_steps', type=int, default=40000)

args = parser.parse_args()

os.makedirs("output", exist_ok=True)


def main():
    my_model, transform = load_model_and_transform(args.model)
    my_model.cuda().train()
    my_model: torch.nn.Module

    if args.model == "ViT-B/32":
        my_model.load_state_dict(torch.load(
            "FP16-ViT-B-32.pt", "cpu"), strict=True)
    elif args.model == "ViT-B/16":
        my_model.load_state_dict(torch.load(
            "FP16-ViT-B-16.pt", "cpu"), strict=True)
    elif args.model == "ViT-L/14":
        my_model.load_state_dict(torch.load(
            "FP16-ViT-L-14.pt", "cpu"), strict=True)
        my_model_seg = MyModelSeg(
            seq_length=196, dimension=1024).cuda().train()
    elif args.model == "ViT-L/14@336px":
        my_model.load_state_dict(torch.load(
            "FP16-ViT-L-14-336px.pt", "cpu"), strict=True)
        my_model_seg = MyModelSeg(
            seq_length=576, dimension=1024).cuda().train()
        args.image_size = 336

    my_dataset = MyDataset(transform)
    my_sampler = DistributedSampler(my_dataset, shuffle=True)
    my_dataloader = DataLoader(
        my_dataset, args.batch_size,
        num_workers=8, sampler=my_sampler, drop_last=True)

    my_optimizer = AdamW([
        {"params": my_model_seg.parameters(), "lr": 0.001},
        {"params": my_model.parameters(), "lr": 0.00001},
    ], weight_decay=0.1)
    my_linear_lr = LinearLR(
        my_optimizer, start_factor=1,
        end_factor=0, total_iters=args.total_steps)
    my_model = DistributedDataParallel(
        my_model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True, static_graph=True)
    my_model_seg = DistributedDataParallel(
        my_model_seg, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True, static_graph=True)
    my_loss = torch.nn.CrossEntropyLoss().cuda()
    my_scaler = torch.cuda.amp.GradScaler()
    step = 0
    while True:
        for image, label in my_dataloader:
            my_sampler.set_epoch(step)
            image = image.cuda()
            label = label.cuda()

            with torch.cuda.amp.autocast(True):
                predict = my_model_seg(my_model(image))
            predict = predict.float()

            loss = my_loss(predict.reshape(-1, 5), label.reshape(-1))

            my_scaler.scale(loss).backward()
            my_scaler.step(my_optimizer)
            my_scaler.update()
            my_linear_lr.step()
            my_optimizer.zero_grad()
            step += 1
            if step % args.frequent == 0:
                output, metric = evaluate(my_model, my_model_seg, image_size=args.image_size)
                output = output.cpu().numpy()
                color = np.ones([output.shape[0], output.shape[1], 3])
                color[output == 0] = [255, 255, 255]  # 其他，白色，0
                color[output == 1] = [0, 255, 0]  # 植被，绿色，1
                color[output == 2] = [0, 0, 0]  # 道路，黑色，2
                color[output == 3] = [131, 139, 139]  # 建筑，黄色，3
                color[output == 4] = [139, 69, 19]  # 水体，蓝色，4
                if rank == 0:
                    cv2.imwrite(f"output/{step :07d}.jpg", color)
                with torch.no_grad():
                    distributed.all_reduce(loss)
                    # distributed.all_reduce(metric)
                    if rank == 0:
                        print(
                            f"step: {step :07d} loss: {loss.item() :.4f} dice: {metric.item() :.4f}")
        if step > args.total_steps and rank == 0:
            torch.save(my_model_seg.state_dict(), "model.pt")
            exit()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        if os.path.exists("dataset/path_list.csv"):
            with open("dataset/path_list.csv", "r") as f:
                lines = f.readlines()
            lines = [x.strip() for x in lines]
            lines = lines[1:]
            self.lines = lines
        else:
            raise "Please run <python preprocess.py first!>"
        self.transform = transform

    def __getitem__(self, index):
        path_image, path_label = self.lines[index].split(",")
        image = self.transform(Image.open(path_image))
        return image, torch.from_numpy(cv2.imread(path_label, cv2.IMREAD_GRAYSCALE))

    def __len__(self):
        return len(self.lines)

class MyModelSeg(torch.nn.Module):
    def __init__(self, seq_length=49, dimension=768) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.dimension = dimension

        if seq_length == 49:
            self.layers = torch.nn.Sequential(
                # B x dimension x 7 x 7
                ConvTranspose2d(dimension, 768, kernel_size=2, stride=2),
                LayerNorm2d(768),
                GELU(),
                # B x 768 x 14 x 14
                ConvTranspose2d(768, 384, kernel_size=2, stride=2),
                LayerNorm2d(384),
                GELU(),
                # B x 384 x 28 x 28
                ConvTranspose2d(384, 196, kernel_size=2, stride=2),
                LayerNorm2d(196),
                GELU(),
                # B x 192 x 56 x 56
                ConvTranspose2d(196, 96, kernel_size=2, stride=2),
                LayerNorm2d(96),
                GELU(),
                # B x 96 x 112 x 112
                ConvTranspose2d(96, 48, kernel_size=2, stride=2),
                LayerNorm2d(48),
                GELU(),
                # B x 48 x 224 x 224
                Conv2d(48, 5, (1, 1), 1, 0, bias=False)
            )
        elif seq_length == 196:
            self.layers = torch.nn.Sequential(
                # B x dimension x 14 x 14
                ConvTranspose2d(dimension, 384, kernel_size=2, stride=2),
                LayerNorm2d(384),
                GELU(),
                # B x 384 x 28 x 28
                ConvTranspose2d(384, 196, kernel_size=2, stride=2),
                LayerNorm2d(196),
                GELU(),
                # B x 192 x 56 x 56
                ConvTranspose2d(196, 96, kernel_size=2, stride=2),
                LayerNorm2d(96),
                GELU(),
                # B x 96 x 112 x 112
                ConvTranspose2d(96, 48, kernel_size=2, stride=2),
                LayerNorm2d(48),
                GELU(),
                # B x 48 x 224 x 224
                Conv2d(48, 5, (1, 1), 1, 0, bias=False)
            )
        elif seq_length == 576:
            self.layers = torch.nn.Sequential(
                # B x dimension x 21 x 21
                ConvTranspose2d(dimension, 512, kernel_size=2, stride=2),
                LayerNorm2d(512),
                GELU(),
                # B x 384 x 42 x 42
                ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                LayerNorm2d(256),
                GELU(),
                # B x 192 x 84 x 84
                ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                GELU(),
                # B x 96 x 168 x 168
                ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                LayerNorm2d(64),
                GELU(),
                # B x 48 x 336 x 336
                Conv2d(64, 5, (1, 1), 1, 0, bias=False)
            )
        else:
            raise NotImplementedError




    def forward(self, x):
        B, S, D = x.size()
        if self.seq_length == 49:
            x = torch.reshape(x, (B, 7, 7, self.dimension))
        elif self.seq_length == 196:
            x = torch.reshape(x, (B, 16, 16, self.dimension))
            x = x[:, 1:15, 1:15, :]
        elif self.seq_length == 576:
            x = torch.reshape(x, (B, 24, 24, self.dimension))
            x = x[:, 0:21, 0:21, :]

        x: torch.Tensor
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.permute(0, 2, 3, 1)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def evaluate(model, model_seg, image_size=224):
    my_dice = Dice().cuda()
    model.eval()
    model_seg.eval()
    image = cv2.imread("dataset/origin/5.png")
    gt = torch.from_numpy(cv2.imread(
        "dataset/origin/5_class.png", cv2.IMREAD_GRAYSCALE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)
    output = torch.zeros([h, w]).cuda().long()

    idx_h = 0
    while idx_h * image_size < h:
        h_e = min(h, idx_h * image_size + image_size)
        h_s = h_e - image_size
        idx_w = 0
        while idx_w * image_size < w:
            w_e = min(w, idx_w * image_size + image_size)
            w_s = w_e - image_size
            idx_w += 1
            img = image[:, :, h_s: h_s + image_size,
                        w_s: w_s + image_size].cuda()
            img = img / 255
            img[:, 0, :, :] -= 0.485
            img[:, 1, :, :] -= 0.456
            img[:, 2, :, :] -= 0.406
            img[:, 0, :, :] /= 0.229
            img[:, 1, :, :] /= 0.224
            img[:, 2, :, :] /= 0.225
            with torch.cuda.amp.autocast(True):
                predict = model_seg(model(img))
            predict = predict.float()
            predict = torch.argmax(predict, dim=3).squeeze()
            output[h_s: h_s + image_size, w_s: w_s + image_size] = predict
        idx_h += 1
    model.train()
    model_seg.train()
    metric = my_dice(output, gt.cuda())
    return output, metric


if __name__ == "__main__":
    main()
