import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_pad_crop = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
        self.aug_crop = nn.Sequential(
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
        self.aug_rotation = kornia.augmentation.RandomRotation(degrees=5.0)

        self.color_jitter = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)

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

    def sample(self, batch_size, CBAM=False, data_aug=0):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        # data_aug
        # 0: padding + random crop
        # 1: interpolate + random crop
        # 2: color jitter
        if not CBAM:
            if data_aug == 0:
                obses = self.aug_pad_crop(obses)
                next_obses = self.aug_pad_crop(next_obses)
            elif data_aug == 1:
                obses = F.interpolate(obses, mode='bilinear', scale_factor=1.1)
                obses = self.aug_crop(obses)
                next_obses = F.interpolate(next_obses, mode='bilinear', scale_factor=1.1)
                next_obses = self.aug_crop(next_obses)
            elif data_aug == 2:
                obses = self.color_jitter(obses)
                next_obses = self.color_jitter(next_obses)

        if data_aug == 0:
            obses_aug = self.aug_pad_crop(obses_aug)
            next_obses_aug = self.aug_pad_crop(next_obses_aug)
        elif data_aug == 1:
            obses_aug = F.interpolate(obses_aug, mode='bilinear', scale_factor=1.1)
            obses_aug = self.aug_crop(obses_aug)
            next_obses_aug = F.interpolate(next_obses_aug, mode='bilinear', scale_factor=1.1)
            next_obses_aug = self.aug_crop(next_obses_aug)
        elif data_aug == 2:
            obses_aug = self.color_jitter(obses_aug)
            next_obses_aug = self.color_jitter(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
