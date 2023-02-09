import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, data_aug, cycnn, degrees,
                 randnet):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        # avoid using small rotation
        self.aug_rotation_1 = kornia.augmentation.RandomRotation(degrees=[15.0, degrees])
        self.aug_rotation_2 = kornia.augmentation.RandomRotation(degrees=[-degrees, -15.0])

        self.aug_h_flip = kornia.augmentation.RandomHorizontalFlip(p=0.1)

        self.rand_conv = nn.Conv2d(obs_shape[0], obs_shape[0], kernel_size=3, padding='same').to(self.device)

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

        self.data_aug = data_aug
        self.cycnn = cycnn
        self.randnet = randnet

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
        # np.save('/bigdata/users/jhu/hidden-drq/outputs/obs_black.npy', obses)
        # raise ValueError('debug')
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        if self.cycnn:
            obses = torch.as_tensor(obses).float()
            next_obses = torch.as_tensor(next_obses).float()
            obses_aug = torch.as_tensor(obses_aug).float()
            next_obses_aug = torch.as_tensor(next_obses_aug).float()
        else:
            obses = torch.as_tensor(obses, device=self.device).float()
            next_obses = torch.as_tensor(next_obses, device=self.device).float()
            obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
            next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.data_aug == 1:
            obses = self.aug_trans(obses)
            next_obses = self.aug_trans(next_obses)

            obses_aug = self.aug_trans(obses_aug)
            next_obses_aug = self.aug_trans(next_obses_aug)
        elif self.data_aug == 2:
            if np.random.rand(1) < 0.5:
                aug_rotation = self.aug_rotation_1
            else:
                aug_rotation = self.aug_rotation_2
            obses = aug_rotation(obses)
            next_obses = aug_rotation(next_obses)

            obses_aug = aug_rotation(obses_aug)
            next_obses_aug = aug_rotation(next_obses_aug)
        elif self.data_aug == 3:
            obses = self.aug_h_flip(obses)
            next_obses = self.aug_h_flip(next_obses)

            obses_aug = self.aug_h_flip(obses_aug)
            next_obses_aug = self.aug_h_flip(next_obses_aug)

        if self.cycnn:
            obses = utils.polar_transform(obses)
            next_obses = utils.polar_transform(next_obses)

            obses_aug = utils.polar_transform(obses_aug)
            next_obses_aug = utils.polar_transform(next_obses_aug)

            obses = obses.to(self.device)
            next_obses = next_obses.to(self.device)

            obses_aug = obses_aug.to(self.device)
            next_obses_aug = next_obses_aug.to(self.device)

        if self.randnet:
            with torch.no_grad():
                # if np.random.rand(1) < 0.9:
                #     torch.nn.init.xavier_normal_(self.rand_conv.weight)
                #     obses = self.rand_conv(obses)
                #     next_obses = self.rand_conv(next_obses)
                # if np.random.rand(1) < 0.9:
                torch.nn.init.xavier_normal_(self.rand_conv.weight)
                obses_aug = self.rand_conv(obses_aug)
                next_obses_aug = self.rand_conv(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
