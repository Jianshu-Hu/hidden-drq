import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torchvision.utils import save_image

import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_pad_crop = nn.Sequential(
           nn.ReplicationPad2d(image_pad),
           kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        # self.aug_pad_crop = torchvision.transforms.RandomCrop(size=(obs_shape[-1], obs_shape[-1]), padding=image_pad,
        #                                                       padding_mode='edge')
        self.aug_pad_crop_zoom_out = torchvision.transforms.RandomCrop(size=(obs_shape[-1], obs_shape[-1]), padding=10,
                                                 padding_mode='edge')
        self.aug_crop = torchvision.transforms.RandomCrop(size=(obs_shape[-1], obs_shape[-1]))
        self.aug_rotation = kornia.augmentation.RandomRotation(degrees=5.0)

        self.color_jitter = torchvision.transforms.ColorJitter(brightness=0.1)
        # , contrast=0.3, saturation=0.3, hue=0.3)

        self.background_aug = BackgroundDetection(device)

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, data_aug=0):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()
        # obses_aug_2 = obses.copy()
        # next_obses_aug_2 = next_obses.copy()
        # obses_aug_3 = obses.copy()
        # next_obses_aug_3 = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        # obses_aug_2 = torch.as_tensor(obses_aug_2, device=self.device).float()
        # next_obses_aug_2 = torch.as_tensor(next_obses_aug_2, device=self.device).float()
        # obses_aug_3 = torch.as_tensor(obses_aug_3, device=self.device).float()
        # next_obses_aug_3 = torch.as_tensor(next_obses_aug_3, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        # data_aug
        # 0: padding + random crop
        # 1: interpolate + random crop
        # 2: color jitter + padding, random crop
        # 3: padding random crop + rotation
        # 4: remove background
        # 5: add noise to background + padding, random crop
        # 6: zoom out + random crop
        if data_aug == 0:
            obses = self.aug_pad_crop(obses)
            next_obses = self.aug_pad_crop(next_obses)

            obses_aug = self.aug_pad_crop(obses_aug)
            next_obses_aug = self.aug_pad_crop(next_obses_aug)

            # obses_aug_2 = self.aug_pad_crop(obses_aug_2)
            # next_obses_aug_2 = self.aug_pad_crop(next_obses_aug_2)
            #
            # obses_aug_3 = self.aug_pad_crop(obses_aug_3)
            # next_obses_aug_3 = self.aug_pad_crop(next_obses_aug_3)

            # obses_aug = [obses_aug_1, obses_aug_2, obses_aug_3]
            # next_obses_aug = [next_obses_aug_1, next_obses_aug_2, next_obses_aug_3]
        elif data_aug == 1:
            obses = F.interpolate(obses, mode='bilinear', scale_factor=1.05)
            obses = self.aug_crop(obses)
            next_obses = F.interpolate(next_obses, mode='bilinear', scale_factor=1.05)
            next_obses = self.aug_crop(next_obses)
        elif data_aug == 2:
            repeat = int(obses.shape[1]/3)
            for i in range(repeat):
                obses[:, 3*i:3*(i+1)] = self.color_jitter(obses[:, 3*i:3*(i+1)])
                next_obses[:, 3*i:3*(i+1)] = self.color_jitter(next_obses[:, 3*i:3*(i+1)])
            obses = self.aug_pad_crop(obses)
            next_obses = self.aug_pad_crop(next_obses)
        elif data_aug == 3:
            obses = self.aug_rotation(obses)
            next_obses = self.aug_rotation(next_obses)

            obses_aug = self.aug_rotation(obses_aug)
            next_obses_aug = self.aug_rotation(next_obses_aug)
        elif data_aug == 4:
            obses = self.background_aug.remove_background(obses)
            next_obses = self.background_aug.remove_background(next_obses)
        elif data_aug == 5:
            obses = self.background_aug.add_noise_to_background(obses)
            next_obses = self.background_aug.add_noise_to_background(next_obses)
            obses = self.aug_pad_crop(obses)
            next_obses = self.aug_pad_crop(next_obses)
        elif data_aug == 6:
            obses = F.interpolate(obses, mode='bilinear', scale_factor=0.8)
            obses = self.aug_pad_crop_zoom_out(obses)
            next_obses = F.interpolate(next_obses, mode='bilinear', scale_factor=0.8)
            next_obses = self.aug_pad_crop_zoom_out(next_obses)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug


class BackgroundDetection():
    def __init__(self, device):
        self.threshold = 0.001
        self.device = device

        self.gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2, 5))

    def background_masks(self, images):
        # images N*C*W*H
        repeat_frames = int(images.shape[1]/3)
        mask = torch.ones((images.shape[0], images.shape[-1], images.shape[-1]), dtype=torch.bool).to(self.device)
        for i in range(repeat_frames-1):
            diff = torch.mean(torch.square(images[:, i*3:(i+1)*3] - images[:, (i+1)*3:(i+2)*3]), dim=1)
            mask = mask & (diff < self.threshold)

        return mask.unsqueeze(1).expand(images.shape)

    def remove_background(self, images):
        mask = self.background_masks(images)
        images = torch.masked_fill(images, mask, value=0)
        return images

    def add_noise_to_background(self, images):
        mask = self.background_masks(images)
        noisy_images = images * (~mask) + self.gaussian_blur(images) * mask
        return noisy_images