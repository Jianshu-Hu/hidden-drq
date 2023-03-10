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
        obses_aug_1 = obses.copy()
        next_obses_aug_1 = next_obses.copy()
        obses_aug_2 = obses.copy()
        next_obses_aug_2 = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug_1 = torch.as_tensor(obses_aug_1, device=self.device).float()
        next_obses_aug_1 = torch.as_tensor(next_obses_aug_1, device=self.device).float()
        obses_aug_2 = torch.as_tensor(obses_aug_2, device=self.device).float()
        next_obses_aug_2 = torch.as_tensor(next_obses_aug_2, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.data_aug > 0:
            obses = self.aug(obses)
            next_obses = self.aug(next_obses)

            obses_aug_1 = self.aug(obses_aug_1)
            next_obses_aug_1 = self.aug(next_obses_aug_1)

            obses_aug_2 = self.aug(obses_aug_2)
            next_obses_aug_2 = self.aug(next_obses_aug_2)
        else:
            # without data aug
            pass

        return obses, actions, rewards, next_obses, not_dones_no_max,\
               obses_aug_1, next_obses_aug_1, obses_aug_2, next_obses_aug_2


