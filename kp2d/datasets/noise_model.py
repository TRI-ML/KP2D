import cv2
import numpy as np
import scipy.signal
from PIL import Image
from kp2d.datasets.augmentations import sample_homography, warp_homography
from math import pi
import torch
from kp2d.utils.image import image_grid

#import kornia

class NoiseUtility():
    def __init__(self, shape, fov = 70):
        self.shape = shape
        self.fov = fov
        self.device = 'cuda'

        self.init_map()

        self.kernel = torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0.25, 0.5, 0.25, 0], [0.5, 1, 1, 1, 0.5], [0, 0.25, 0.5, 0.25, 0], [0, 0, 0, 0, 0]])
        self.kernel = (self.kernel / torch.sum(self.kernel)).unsqueeze(0).unsqueeze(0).to(self.device)




    def init_map(self):
        fov = 70
        r_max = 100
        r_min = 0

        h, w = self.shape
        ang = np.linspace(-fov / 2, +fov / 2, w)
        r = np.linspace(r_min, h, h)
        map = np.zeros([h, w, 2])
        map_inv = np.zeros([h, w, 2])

        for x in range(w):
            for y in range(h):
                r_n = r[y]
                ang_n = ang[x]
                map[y, x, 1] = r_n * np.cos(ang_n * np.pi / 180)
                map[y, x, 0] = r_n * np.sin(ang_n * np.pi / 180)

        for x in range(w):
            for y in range(h):
                x_n = x - w / 2
                y_n = y

                map_inv[y, x, 1] = np.sqrt(x_n * x_n + y_n * y_n)
                map_inv[y, x, 0] = np.arctan(x_n / max(y_n, 0.00000001)) / np.pi * 2

        def normalize(m):
            minimum = m.min()
            r = m.max() - minimum

            return (m - minimum - r / 2) / r * 2

        map_inv[:, :, 0] = normalize(map_inv[:, :, 0]) * 180 / 70
        map_inv[:, :, 1] = normalize(map_inv[:, :, 1])

        map[:, :, 0] = normalize(map[:, :, 0])
        map[:, :, 1] = normalize(map[:, :, 1])
        print(map_inv[:, :, 0].min(), map_inv[:, :, 1].min())
        print(map_inv[:, :, 0].max(), map_inv[:, :, 1].max())
        self.map = torch.from_numpy(map).float().unsqueeze(0).to(self.device)
        self.map_inv = torch.from_numpy(map_inv).float().unsqueeze(0).to(self.device)

    def filter(self, img):

        filtered = img

        noise = create_row_noise_torch(torch.clip(filtered, 2, 50)) * 2

        filtered = filtered + noise
        filtered = add_sparkle(filtered, self.kernel)
        # filtered = scipy.signal.convolve2d(filtered, self.kernel, boundary='symm', mode='same')
        #
        # filtered = add_sparkle(filtered, self.kernel)
        # filtered = (filtered * 0.75 + img * 0.25)
        #
        # filtered = np.clip(filtered * (1 + 0.3 * create_speckle_noise(filtered)), 0, 255)
        return filtered

    def sim_2_real_filter(self, img):
        if img.shape.__len__() == 3:
            mapped = self.pol_2_cart(img[:,:,0], self.map)
        else:
            mapped = self.pol_2_cart(img, self.map)

        mapped = self.filter(mapped)
        mapped = self.cart_2_pol(mapped, self.map)
        mapped = scipy.signal.convolve2d(mapped, self.kernel, boundary='symm', mode='same')
        if img.shape.__len__() == 3:
            return np.stack((mapped,mapped,mapped), axis=2).astype(img.dtype)
        else:
            return mapped.astype(img.dtype)

    def to_torch(self, img):
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device)
    def pol_2_cart_sample(self, sample):
        img = self.to_torch(np.array(sample['image'])[:,:,0])
        mapped = self.pol_2_cart_torch(img)
        sample['image'] = mapped.to(img.dtype)
        return sample

    def augment_sample(self, sample):
        orig_type = sample['image'].dtype
        img = sample['image']
        _,_,H, W = img.shape

        homography = sample_homography([H, W], perspective=False, scaling = False,
                                       patch_ratio=0.7,
                                       scaling_amplitude=0,
                                       max_angle=pi)
        homography = torch.from_numpy(homography).float().to(self.device)
        source_grid = image_grid(1, H, W,
                                 dtype=img.dtype,
                                 device=self.device,
                                 ones=False, normalized=True).clone().permute(0, 2, 3, 1)

        source_warped = warp_homography(source_grid, homography)
        source_img = torch.nn.functional.grid_sample(img, source_warped, align_corners=True)


        sample['image'] = img.to(orig_type)
        sample['image_aug'] = source_img.to(orig_type)
        sample['homography'] = homography


        return sample

    def filter_sample(self, sample):
        img = sample['image']
        img_aug = sample['image_aug']

        sample['image'] = self.filter(img).to(img.dtype)
        sample['image_aug'] = self.filter(img_aug).to(img_aug.dtype)
        return sample

    def cart_2_pol_sample(self, sample):
        img = sample['image']
        img_aug = sample['image_aug']
        mapped = self.cart_2_pol_torch(img).squeeze(0).squeeze(0)
        mapped_aug = self.cart_2_pol_torch(img_aug).squeeze(0).squeeze(0)
        sample['image'] = torch.stack((mapped, mapped, mapped), axis=0).to(img.dtype)
        sample['image_aug'] = torch.stack((mapped_aug, mapped_aug, mapped_aug), axis=0).to(img.dtype)

        cv2.imshow("img", mapped.to('cpu').numpy()  / 255)
        cv2.imshow("img aug", mapped_aug.to('cpu').numpy()  / 255)
        cv2.waitKey(0)
        return sample

    def pol_2_cart_torch(self, img):
        return torch.nn.functional.grid_sample(img, self.map_inv, mode='bilinear', padding_mode='zeros',
                                                align_corners=None)

    def cart_2_pol_torch(self, img):
        return torch.nn.functional.grid_sample(img, self.map, mode='bilinear', padding_mode='zeros',
                                                align_corners=None)



def create_row_noise_torch(x, amp= 20):
    noise = x.clone().to('cuda')
    for r in range(x.shape[2]):
        noise[:,:,r,:] =(torch.randn(x.shape[3])*amp-amp/2 + torch.randn(x.shape[3])*amp/2+amp/2).to('cuda')/(torch.sum(x[:,:,r,:])/3000+1)
    return noise

def create_row_noise(x):
    noise = x.clone()
    for r in range(x.shape[0]):
        noise[r,:] = np.random.normal(5,5,x.shape[1])/(np.sum(x[r,:])/500+1)
    return noise

def create_speckle_noise(x):
    noise = np.clip(np.random.normal(20,100,x.shape)-100,0,255)/255.
    return noise

def add_sparkle(x, conv_kernel):
    kernel = torch.ones(3, 3)
    sparkle = torch.clip(x-150,0,255)*2*(torch.randn(x.shape).to('cuda'))
    #sparkle = torch.clip((kornia.morphology.dilation(x, kernel, iterations=2).astype('int8')-50)*2-np.random.normal(20,100,x.shape),0,255)
    sparkle = torch.nn.functional.conv2d(sparkle, conv_kernel, bias=None, stride=[1,1], padding='same')
    x = torch.clip(x+sparkle,0,255)
    return x





