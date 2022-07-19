# import torch
from torchvision import transforms

import random
# from PIL import ImageFilter


class RandomApply:

    def __init__(self, transform, p):
        self.p = p
        self.transform = transform

    def __call__(self, image):
        if random.random() < self.p:
            image = self.transform(image)
        return image


# class GaussianBlur:
#     def __init__(self, std_min, std_max):
#         self.std_min = std_min
#         self.std_max = std_max
#
#     def __call__(self, image):
#         radius = random.uniform(self.std_min, self.std_max)
#         image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         return image


class GaussianBlur(transforms.GaussianBlur):

    def __init__(self, input_h=512, input_w=1024, sigma=(0.1, 2.0)):
        k_h, k_w = input_h // 10, input_w // 10
        k_h -= (k_h + 1) % 2
        k_w -= (k_w + 1) % 2
        super().__init__(kernel_size=(k_h, k_w), sigma=sigma)
