import os

import cv2
import numpy as np
import torch
from torchmetrics import Dice
from PIL import Image
from torch.nn import Conv2d, LayerNorm, ReLU, Upsample
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.dataloader import DataLoader

from vit import load_model_and_transform

os.makedirs("output", exist_ok=True)

def main():
    my_model, transform = load_model_and_transform("ViT-B/32")
    my_model.cuda().train()
    my_model: torch.nn.Module
    if os.path.exists("FP16-ViT-B-32.pt"):
        my_model.load_state_dict(torch.load("FP16-ViT-B-32.pt", "cpu"), strict=True)

    my_dataset = MyDataset(transform)
    my_dataloader = DataLoader(my_dataset, 32, True, num_workers=8, drop_last=True)
    my_model_seg = MyModelSeg().cuda().train()
    my_optimizer = AdamW([
            {"params": my_model_seg.parameters(), "lr": 0.0005},
            {"params": my_model.parameters(), "lr": 0.00001},
        ], weight_decay=0.1)
    my_linear_lr = LinearLR(my_optimizer, start_factor=1, end_factor=0, total_iters=20000)
    my_loss = torch.nn.CrossEntropyLoss().cuda()
    step = 0
    while True:
        for image, label in my_dataloader:

            image = image.cuda()
            label = label.cuda()

            predict = my_model_seg(my_model(image))
            loss = my_loss(predict.reshape(-1, 5), label.reshape(-1))

            loss.backward()
            my_optimizer.step()
            my_linear_lr.step()
            my_optimizer.zero_grad()
            step += 1
            if step % 100 == 0:
                output, metric = evaluate(my_model, my_model_seg)
                output = output.cpu().numpy()
                color = np.ones([output.shape[0], output.shape[1], 3])
                color[output==0] = [255, 255, 255] #其他，白色，0
                color[output==1] = [0, 255, 0]     #植被，绿色，1
                color[output==2] = [0, 0, 0]       #道路，黑色，2
                color[output==3] = [131, 139, 139] #建筑，黄色，3
                color[output==4] = [139, 69, 19]   #水体，蓝色，4
                cv2.imwrite(f"output/{step :07d}.jpg", color)

                with torch.no_grad():
                    print(f"step: {step :07d} loss: {loss.item() :.4f} dice: {metric.item() :.4f}")

        if step > 20000:
            torch.save(my_model_seg.state_dict(), "model.pt")
            break


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
    def __init__(self,) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            # B x 768 x 7 x 7
            Upsample(scale_factor=2),
            Conv2d(768, 768, (1, 1), 1, 0, bias=False),
            ReLU(),
            Conv2d(768, 768, (3, 3), 1, (1, 1), bias=False),
            LayerNorm([768, 14, 14]),
            ReLU(),
            # B x 768 x 14 x 14
            Upsample(scale_factor=2),
            Conv2d(768, 384, (1, 1), 1, 0, bias=False),
            ReLU(),
            Conv2d(384, 384, (3, 3), 1, (1, 1), bias=False),
            LayerNorm([384, 28, 28]),
            ReLU(),
            # B x 384 x 28 x 28
            Upsample(scale_factor=2),
            Conv2d(384, 192, (1, 1), 1, 0, bias=False),
            ReLU(),
            Conv2d(192, 192, (3, 3), 1, (1, 1), bias=False),
            LayerNorm([192, 56, 56]),
            ReLU(),
            # B x 192 x 56 x 56
            Upsample(scale_factor=2),
            Conv2d(192, 96, (1, 1), 1, 0, bias=False),
            ReLU(),
            Conv2d(96, 96, (3, 3), 1, (1, 1), bias=False),
            LayerNorm([96, 112, 112]),
            ReLU(),
            # B x 96 x 112 x 112
            Upsample(scale_factor=2),
            Conv2d(96, 48, (1, 1), 1, 0, bias=False),
            ReLU(),
            Conv2d(48, 48, (3, 3), 1, (1, 1), bias=False),
            LayerNorm([48, 224, 224]),
            ReLU(),
            # B x 48 x 224 x 224
            Conv2d(48, 5, (1, 1), 1, 0, bias=False))

    def forward(self, x):
        B, S, D = x.size()
        x = torch.reshape(x, (B, 7, 7, 768))
        x: torch.Tensor
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.permute(0, 2, 3, 1)
        return x


def evaluate(model, model_seg, image_size=224):
    my_dice = Dice().cuda()
    model.eval()
    model_seg.eval()
    image = cv2.imread("dataset/origin/5.png")
    gt = torch.from_numpy(cv2.imread("dataset/origin/5_class.png", cv2.IMREAD_GRAYSCALE))
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
            img = image[:, :, h_s: h_s + image_size, w_s: w_s + image_size].cuda()
            img = img / 255
            img[:, 0, :, :] -= 0.485
            img[:, 1, :, :] -= 0.456
            img[:, 2, :, :] -= 0.406
            img[:, 0, :, :] /= 0.229
            img[:, 1, :, :] /= 0.224
            img[:, 2, :, :] /= 0.225
            predict = model_seg(model(img))
            predict = torch.argmax(predict, dim=3).squeeze()
            output[h_s: h_s + image_size, w_s: w_s + image_size] = predict
        idx_h += 1
    model.train()
    model_seg.train()
    metric = my_dice(output, gt.cuda())
    return output, metric


if __name__ == "__main__":
    main()
