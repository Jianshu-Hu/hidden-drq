import numpy as np
import torch
import torchvision
import data_aug_new as new_aug


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, data_aug, degrees, dist_alpha):
        self.capacity = capacity
        self.device = device

        self.aug = new_aug.aug(data_aug, image_pad, obs_shape, degrees, dist_alpha)

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


