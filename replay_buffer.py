import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torchvision
from torch import Tensor
from torchvision.transforms import functional as F

from typing import List


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, data_aug, degrees, dist_alpha):
        self.capacity = capacity
        self.device = device

        self.aug = aug(data_aug, image_pad, obs_shape, degrees, dist_alpha)

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

        self.data_aug = data_aug

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

    def sample(self, batch_size):
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
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.data_aug > 0:
            if self.data_aug in {5, 6}:
                # if we use torchvision, we need this setting to make sure different augmentations
                # are applied within the mini-batch
                obses = torchvision.transforms.Lambda(
                    lambda x: torch.stack([self.aug(x_) for x_ in x]))(obses)
                next_obses = torchvision.transforms.Lambda(
                    lambda x: torch.stack([self.aug(x_) for x_ in x]))(next_obses)

                obses_aug = torchvision.transforms.Lambda(
                    lambda x: torch.stack([self.aug(x_) for x_ in x]))(obses_aug)
                next_obses_aug = torchvision.transforms.Lambda(
                    lambda x: torch.stack([self.aug(x_) for x_ in x]))(next_obses_aug)
            else:
                obses = self.aug(obses)
                next_obses = self.aug(next_obses)

                obses_aug = self.aug(obses_aug)
                next_obses_aug = self.aug(next_obses_aug)
        else:
            # without data aug
            pass

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug


def aug(data_aug, image_pad, obs_shape, degrees, dist_alpha):
    if data_aug == 1:
        # random crop
        augmentation = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop(size=(obs_shape[-1], obs_shape[-1])))
    elif data_aug == 2:
        # random rotation
        augmentation = kornia.augmentation.RandomRotation(degrees=degrees)
    elif data_aug == 3:
        # random hflip
        augmentation = kornia.augmentation.RandomHorizontalFlip(p=0.1)
    elif data_aug == 4:
        # random rotation + random crop
        augmentation = nn.Sequential(
            kornia.augmentation.RandomRotation(degrees=degrees),
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop(size=(obs_shape[-1], obs_shape[-1]))
        )
    elif data_aug == 5:
        # random crop with beta distribution
        augmentation = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            RandomCropNew(size=(obs_shape[-1], obs_shape[-1]), alpha=dist_alpha))
    elif data_aug == 6:
        # random rotation with beta distribution
        augmentation = RandomRotationNew(degrees=degrees, alpha=dist_alpha)
    else:
        augmentation = None

    return augmentation


class RandomCropNew(torchvision.transforms.RandomCrop):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)

        self.beta = torch.distributions.Beta(alpha, alpha)

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        # beta distribution
        i = int((h - th) * self.beta.sample().item())
        j = int((w - tw) * self.beta.sample().item())
        # if np.random.rand(1) < 0.5:
        #     i = torch.randint(0, (h-th)+1, size=(1,)).item()
        #     if i == int((h - th) / 2):
        #         if np.random.rand(1) < 0.5:
        #             j = torch.randint(0, int((w - tw) / 2 - 1), size=(1,)).item()
        #         else:
        #             j = torch.randint(int((w - tw) / 2) + 2, (w - tw) + 1, size=(1,)).item()
        #     elif i == int((w - tw) / 2)+1 or i == int((w - tw) / 2)-1:
        #         if np.random.rand(1) < 0.5:
        #             j = torch.randint(0, int((w - tw) / 2), size=(1,)).item()
        #         else:
        #             j = torch.randint(int((w - tw) / 2) + 1, (w - tw) + 1, size=(1,)).item()
        #     else:
        #         j = torch.randint(0, (w-tw)+1, size=(1,)).item()
        # else:
        #     j = torch.randint(0, (w - tw) + 1, size=(1,)).item()
        #     if j == int((w - tw) / 2):
        #         if np.random.rand(1) < 0.5:
        #             i = torch.randint(0, int((h - th) / 2-1), size=(1,)).item()
        #         else:
        #             i = torch.randint(int((h - th)/2) + 2, (h - th) + 1, size=(1,)).item()
        #     elif j == int((w - tw) / 2)-1 or j == int((w - tw) / 2)+1:
        #         if np.random.rand(1) < 0.5:
        #             i = torch.randint(0, int((h - th) / 2), size=(1,)).item()
        #         else:
        #             i = torch.randint(int((h - th)/2) + 1, (h - th) + 1, size=(1,)).item()
        #     else:
        #         i = torch.randint(0, (h-th)+1, size=(1,)).item()

        return i, j, th, tw

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)


class RandomRotationNew(torchvision.transforms.RandomRotation):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)

        self.beta = torch.distributions.Beta(alpha, alpha)

    def get_params(self, degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = degrees[0] + (degrees[1] - degrees[0]) * self.beta.sample().item()
        return angle

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill)

