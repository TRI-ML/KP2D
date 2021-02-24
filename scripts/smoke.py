from kp2d.networks.keypoint_net import KeypointNet, get_vanilla_conv_block, SepConvBlock, MixSepConvBlock
import cv2
from time import time
import numpy as np
from torchvision.transforms import ToTensor
import torch


def main():
    img = np.random.rand(2048, 2048, 3).astype('float32')
    to_tensor = ToTensor()

    device = 'cpu'
    kp1 = KeypointNet(base_conv_block=get_vanilla_conv_block).to(device)
    kp2 = KeypointNet(base_conv_block=SepConvBlock).to(device)
    kp3 = KeypointNet(base_conv_block=MixSepConvBlock).to(device)

    n_trials = 10
    tick = time()
    for _ in range(n_trials):
        t = to_tensor(cv2.resize(img, None, fx=.25, fy=.25))
        kp1(t.unsqueeze(0).to(device))
    print(time() - tick, "model 1")
    tick = time()
    for _ in range(n_trials):
        t = to_tensor(cv2.resize(img, None, fx=.25, fy=.25))
        kp2(t.unsqueeze(0).to(device))
    print(time() - tick, "model 2")
    tick = time()
    for _ in range(n_trials):
        t = to_tensor(cv2.resize(img, None, fx=.25, fy=.25))
        kp3(t.unsqueeze(0).to(device))
    print(time() - tick, "model 3")

    torch.save(kp1, 'kp1.ckpt')
    torch.save(kp2, 'kp2.ckpt')
    torch.save(kp3, 'kp3.ckpt')


if __name__ == '__main__':
    main()
