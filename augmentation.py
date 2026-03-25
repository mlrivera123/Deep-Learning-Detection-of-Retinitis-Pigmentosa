import math
import torch

class Augment:
    def __init__(self, strength, *args, **kwargs):
        self.strength = strength

    def forward(self, z, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, x, model, *args, **kwargs):

        with torch.no_grad():
            x = x.unsqueeze(0).to(model.device, dtype=model.dtype)
            x = x * 2 - 1
            z = model.encode(x).latent_dist.sample()
            z = self.forward(z, *args, **kwargs)
            x_ = model.decode(z, return_dict=False)[0]
            x_ = (x_ + 1) / 2.0
            x_ = x_.squeeze(0).clamp(0, 1).float()
            if model.device != 'cpu':
                x_ = x_.cpu()

        return x_

class ConstantAugment(Augment):

    def forward(self, z):
        z = z + self.strength
        return z

class RandomNormalAugment(Augment):

    def forward(self, z):
        znoise = torch.randn(z.size()).to(z.device, dtype=z.dtype)
        z = z + znoise*self.strength
        return z
    
class RandomUniformAugment(Augment):

    def forward(self, z):
        znoise = torch.rand(z.size()).to(z.device, dtype=z.dtype)
        z = z + znoise*self.strength
        return z

class SinusoidalAugment(Augment):

    def forward(self, z):
        znoise = torch.zeros(z.size()).to(z.device, dtype=z.dtype)
        for i in range(z.size(1)):
            for j in range(z.size(2)):
                for k in range(z.size(3)):
                    znoise[0, i, j, k] = math.sin(i * 0.05 + j * 0.05 + k * 0.05)
        z = z + znoise*self.strength
        return z

class XAugment(Augment):

    def __init__(self, strength, xnoise):
 
        assert 0.0 <= strength <= 1.0, f'strength should in the interval [0, 1], got {strength}'
        self.strength = strength
        self.xnoise = xnoise

    def forward(self, z, model):
        xnoise = self.xnoise.unsqueeze(0).to(model.device, dtype=model.dtype)
        xnoise = xnoise * 2 - 1
        znoise = model.encode(xnoise).latent_dist.sample()
        z = z*(1-self.strength) + znoise*self.strength
        return z
