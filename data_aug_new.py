from typing import Dict, Optional, Tuple, Union, List, Any

import torch
import kornia
from torch.distributions import Distribution, Uniform
from torch.distributions.beta import Beta
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.utils.helpers import _extract_device_dtype

import kornia.augmentation.random_generator as rg
from kornia.core import tensor, Tensor, as_tensor
ParameterBound = Tuple[Any, str, Optional[float], Optional[Tuple[float, float]]]
from torch import nn
from torchvision.transforms import functional as F
import torchvision
import numpy as np


class BetaCropGenerator(rg.CropGenerator):
    def __init__(self, size, alpha) -> None:
        super().__init__(size=size)
        self.alpha = alpha

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.rand_sampler = Beta(tensor(self.alpha, device=device, dtype=dtype),
                                 tensor(self.alpha, device=device, dtype=dtype))


class RandomBetaCrop(kornia.augmentation.RandomCrop):
    def __init__(self, size: Tuple[int, int], alpha) -> None:
        super().__init__(size=size)
        self._param_generator = BetaCropGenerator(size, alpha)


class BetaRotationGenerator(rg.PlainUniformGenerator):
    def __init__(self, *samplers: ParameterBound, alpha) -> None:
        super().__init__(*samplers)
        self.alpha = alpha

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.sampler_dict: Dict[str, Distribution] = {}
        for factor, name, center, bound in self.samplers:
            if center is None and bound is None:
                factor = as_tensor(factor, device=device, dtype=dtype)
            elif center is None or bound is None:
                raise ValueError(f"`center` and `bound` should be both None or provided. Got {center} and {bound}.")
            else:
                factor = _range_bound(factor, name, center=center, bounds=bound, device=device, dtype=dtype)
            self.sampler_dict.update({name: Beta(self.alpha, self.alpha)})
            self.factor = factor

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([t for t, _, _, _ in self.samplers])

        return {
            name: (self.factor[0]+_adapted_rsampling((batch_size,), dist, same_on_batch)
                   .to(device=_device, dtype=_dtype) * (self.factor[1]-self.factor[0]))
            for name, dist in self.sampler_dict.items()
        }


class RandomBetaRotation(kornia.augmentation.RandomRotation):
    def __init__(self, degrees: Union[Tensor, float, Tuple[float, float], List[float]], alpha) -> None:
        super().__init__(degrees=degrees)
        self._param_generator = BetaRotationGenerator((degrees, "degrees", 0.0, (-360.0, 360.0)), alpha=alpha)


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
            RandomBetaCrop(size=(obs_shape[-1], obs_shape[-1]), alpha=dist_alpha))
    elif data_aug == 6:
        # random rotation with beta distribution
        augmentation = RandomBetaRotation(degrees=degrees, alpha=dist_alpha)
    else:
        augmentation = None

    return augmentation

# test
# rng = torch.manual_seed(0)
# input = torch.rand((256, 3, 84, 84))
#
# aug_rot = RandomBetaRotation(degrees=180.0, alpha=0.5)
# aug_crop = RandomBetaCrop(size=(3, 3), alpha=0.5)
# out = aug_rot(input)
# print(out)


class TangentProp():
    def __init__(self, data_aug, image_pad):
        self.data_aug = data_aug
        self.image_pad = image_pad

    def tangent_vector(self, obs):
        if self.data_aug == -1:
            # tan_vec_variance = torch.zeros_like(obs)
            # pad = nn.Sequential(
            #     torch.nn.ReplicationPad2d(1))
            # # horizontal shift 1 pixel
            # obs_aug = F.crop(pad(obs), top=1, left=2, height=obs.shape[-1], width=obs.shape[-1])
            # tan_vector_1 = obs_aug - obs
            # tan_vec_variance += torch.pow(tan_vector_1, 2)
            # # vertical shift 1 pixel
            # obs_aug = F.crop(pad(obs), top=2, left=1, height=obs.shape[-1], width=obs.shape[-1])
            # tan_vector_2 = obs_aug - obs
            # tan_vec_variance += torch.pow(tan_vector_2, 2)
            # tan_vec = torch.pow(tan_vec_variance, 0.5)

            # calculate expected transformed image
            expected_T_obs = self.expected_transformed_obs(obs)
            tan_vec = expected_T_obs-obs
        else:
            tan_vec = None
        return tan_vec

    def expected_transformed_obs(self, obs):
        h, w = obs.shape[-1], obs.shape[-1]
        if self.data_aug == -1:
            expected_trans_obs = torch.zeros_like(obs)
            pad = nn.Sequential(
                torch.nn.ReplicationPad2d(self.image_pad))
            pad_obs = pad(obs)
            for i in range(self.image_pad*2+1):
                for j in range(self.image_pad*2+1):
                    expected_trans_obs += pad_obs[:, :, i:i+h, j:j+w]
            expected_trans_obs = expected_trans_obs/((self.image_pad+1)**2)
        else:
            expected_trans_obs = None
        return expected_trans_obs



# # torch implementation
# class RandomCropNew(torchvision.transforms.RandomCrop):
#     def __init__(self, alpha=0.5, **kwargs):
#         super().__init__(**kwargs)
#
#         self.beta = torch.distributions.Beta(alpha, alpha)
#
#     def get_params(self, img, output_size):
#         """Get parameters for ``crop`` for a random crop.
#
#         Args:
#             img (PIL Image or Tensor): Image to be cropped.
#             output_size (tuple): Expected output size of the crop.
#
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         _, h, w = F.get_dimensions(img)
#         th, tw = output_size
#
#         if h < th or w < tw:
#             raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
#
#         if w == tw and h == th:
#             return 0, 0, h, w
#
#         # beta distribution
#         i = int((h - th + 1) * self.beta.sample().item())
#         j = int((w - tw + 1) * self.beta.sample().item())
#         # if np.random.rand(1) < 0.5:
#         #     i = torch.randint(0, (h-th)+1, size=(1,)).item()
#         #     if i == int((h - th) / 2):
#         #         if np.random.rand(1) < 0.5:
#         #             j = torch.randint(0, int((w - tw) / 2 - 1), size=(1,)).item()
#         #         else:
#         #             j = torch.randint(int((w - tw) / 2) + 2, (w - tw) + 1, size=(1,)).item()
#         #     elif i == int((w - tw) / 2)+1 or i == int((w - tw) / 2)-1:
#         #         if np.random.rand(1) < 0.5:
#         #             j = torch.randint(0, int((w - tw) / 2), size=(1,)).item()
#         #         else:
#         #             j = torch.randint(int((w - tw) / 2) + 1, (w - tw) + 1, size=(1,)).item()
#         #     else:
#         #         j = torch.randint(0, (w-tw)+1, size=(1,)).item()
#         # else:
#         #     j = torch.randint(0, (w - tw) + 1, size=(1,)).item()
#         #     if j == int((w - tw) / 2):
#         #         if np.random.rand(1) < 0.5:
#         #             i = torch.randint(0, int((h - th) / 2-1), size=(1,)).item()
#         #         else:
#         #             i = torch.randint(int((h - th)/2) + 2, (h - th) + 1, size=(1,)).item()
#         #     elif j == int((w - tw) / 2)-1 or j == int((w - tw) / 2)+1:
#         #         if np.random.rand(1) < 0.5:
#         #             i = torch.randint(0, int((h - th) / 2), size=(1,)).item()
#         #         else:
#         #             i = torch.randint(int((h - th)/2) + 1, (h - th) + 1, size=(1,)).item()
#         #     else:
#         #         i = torch.randint(0, (h-th)+1, size=(1,)).item()
#
#         return i, j, th, tw
#
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped.
#
#         Returns:
#             PIL Image or Tensor: Cropped image.
#         """
#         if self.padding is not None:
#             img = F.pad(img, self.padding, self.fill, self.padding_mode)
#
#         _, height, width = F.get_dimensions(img)
#         # pad the width if needed
#         if self.pad_if_needed and width < self.size[1]:
#             padding = [self.size[1] - width, 0]
#             img = F.pad(img, padding, self.fill, self.padding_mode)
#         # pad the height if needed
#         if self.pad_if_needed and height < self.size[0]:
#             padding = [0, self.size[0] - height]
#             img = F.pad(img, padding, self.fill, self.padding_mode)
#
#         i, j, h, w = self.get_params(img, self.size)
#
#         return F.crop(img, i, j, h, w)
#
#
# class RandomRotationNew(torchvision.transforms.RandomRotation):
#     def __init__(self, alpha=0.5, p=0.5, **kwargs):
#         super().__init__(**kwargs)
#
#         self.interpolation = F.InterpolationMode.NEAREST
#         self.beta = torch.distributions.Beta(alpha, alpha)
#         self.p = p
#
#     def get_params(self, degrees: List[float]) -> float:
#         """Get parameters for ``rotate`` for a random rotation.
#
#         Returns:
#             float: angle parameter to be passed to ``rotate`` for random rotation.
#         """
#         # probability of applying the data aug
#         if np.random.rand(1) < 1-self.p:
#             angle = 0
#         else:
#             angle = degrees[0] + (degrees[1] - degrees[0] + 1) * self.beta.sample().item()
#         return angle
#
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be rotated.
#
#         Returns:
#             PIL Image or Tensor: Rotated image.
#         """
#         fill = self.fill
#         channels, _, _ = F.get_dimensions(img)
#         if isinstance(img, torch.Tensor):
#             if isinstance(fill, (int, float)):
#                 fill = [float(fill)] * channels
#             else:
#                 fill = [float(f) for f in fill]
#         angle = self.get_params(self.degrees)
#
#         return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill)
