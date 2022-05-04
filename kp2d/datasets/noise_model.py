import cv2
import numpy as np
import scipy.signal
from PIL import Image
from kp2d.datasets.augmentations import sample_homography, warp_homography
from math import pi
import torch
from kp2d.utils.image import image_grid



class NoiseUtility():
    def __init__(self, shape, fov = 70):
        self.shape = shape
        self.fov = fov
        self.init_map()

        self.kernel = np.array(
            [[0, 0, 0, 0, 0], [0, 0.25, 0.5, 0.25, 0], [0.5, 1, 1, 1, 0.5], [0, 0.25, 0.5, 0.25, 0], [0, 0, 0, 0, 0]])
        self.kernel = self.kernel / np.sum(self.kernel)

    def init_map(self):
        h, w = self.shape
        ang = np.linspace(-self.fov / 2, +self.fov / 2, w)
        r = np.linspace(0, h, h)
        self.map = np.zeros([h, w, 2])

        for x in range(w):
            for y in range(h):
                r_n = r[y]
                ang_n = ang[x]
                self.map[y, x, 0] = r_n * np.cos(ang_n * np.pi / 180)
                self.map[y, x, 1] = r_n * np.sin(ang_n * np.pi / 180)




        self.map[:, :, 1] = self.map[:, :, 1] - np.floor(self.map[:, :, 1].min())

    def filter(self, img):

        filtered = img

        noise = create_row_noise(np.clip(filtered, 2, 50)) * 2

        filtered = filtered + noise
        filtered = add_sparkle(filtered, self.kernel)
        filtered = scipy.signal.convolve2d(filtered, self.kernel, boundary='symm', mode='same')

        filtered = add_sparkle(filtered, self.kernel)
        filtered = (filtered * 0.75 + img * 0.25)

        filtered = np.clip(filtered * (1 + 0.3 * create_speckle_noise(filtered)), 0, 255)
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


    def pol_2_cart_sample(self, sample):
        img = np.array(sample['image'])[:,:,0]
        mapped = self.pol_2_cart(img)
        sample['image'] = mapped.astype(img.dtype)
        return sample

    def augment_sample(self, sample):
        orig_type = sample['image'].dtype
        img = sample['image']
        H, W = img.shape
        img = np.stack((img,img,img), axis=0)
        img = torch.from_numpy(img).float()

        img = img.unsqueeze(0)
        homography = sample_homography([H, W],
                                       patch_ratio=0.7,
                                       scaling_amplitude=0,
                                       max_angle=pi)
        homography = torch.from_numpy(homography).float()
        source_grid = image_grid(1, H, W,
                                 dtype=img.dtype,
                                 device='cpu',
                                 ones=False, normalized=True).clone().permute(0, 2, 3, 1)

        source_warped = warp_homography(source_grid, homography)
        source_img = torch.nn.functional.grid_sample(img, source_warped, align_corners=True)

        sample['image'] = np.array(img.squeeze(0)[0,:,:]).astype(orig_type)
        sample['image_aug'] = np.array(source_img.squeeze(0)[0,:,:]).astype(orig_type)
        sample['homography'] = homography

        return sample

    def filter_sample(self, sample):
        img = sample['image']
        img_aug = sample['image_aug']

        sample['image'] = self.filter(img).astype(img.dtype)
        sample['image_aug'] = self.filter(img_aug).astype(img_aug.dtype)
        return sample
    def cart_2_pol_sample(self, sample):
        img = sample['image']
        img_aug = sample['image_aug']
        mapped = self.pol_2_cart(img)
        mapped_aug = self.pol_2_cart(img_aug)
        sample['image'] = np.stack((mapped, mapped, mapped), axis=0).astype(img.dtype)
        sample['image_aug'] = np.stack((mapped_aug, mapped_aug, mapped_aug), axis=0).astype(img.dtype)

        return sample



    def pol_2_cart(self, img):
        h, w = img.shape
        mapped = np.zeros([h, w])
        for x in range(w):
            for y in range(h):
                if (self.map[y, x, 0] < h) and (self.map[y, x, 1] < w):
                    mapped[int(self.map[y, x, 0]), int(self.map[y, x, 1])] = img[y, x]

        return mapped.astype(img.dtype)

    def cart_2_pol(self, img):
        h, w = img.shape

        mapped = np.zeros([h, w])
        for x in range(w):
            for y in range(h):
                if (self.map[y, x, 0] < h) and (self.map[y, x, 1] < w):
                    mapped[y, x] = img[int(self.map[y, x, 0]), int(self.map[y, x, 1])]

        return mapped.astype(img.dtype)

def create_row_noise(x):
    noise = x.copy()
    for r in range(x.shape[0]):
        noise[r,:] = np.random.normal(5,5,x.shape[1])/(np.sum(x[r,:])/300+1)
    return noise

def create_speckle_noise(x):
    noise = np.clip(np.random.normal(20,100,x.shape)-100,0,255)/255.
    return noise

def add_sparkle(x, conv_kernel):
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype= np.uint8)
    sparkle = np.clip((cv2.erode(x, kernel, iterations=2).astype('int8')-50)*2-np.random.normal(20,100,x.shape),0,255)
    sparkle = scipy.signal.convolve2d(sparkle, conv_kernel, boundary='symm', mode='same')
    x = np.clip(x + sparkle,0,255)
    return x





