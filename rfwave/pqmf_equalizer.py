# #####################################################################################
# ##                  Crafted with <3 by Peng Liu                                    ##
# ##                                                                                 ##
# ##                  - feanorliu@tencent.com                                        ##
# ##                  - laupeng1989@gmail.com                                        ##
# #####################################################################################

import torch
from torch.amp import autocast

from torch import nn
from rfwave.pqmf import PQMF


class PQMFEqualizer(nn.Module):
    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0, update_batches=100000, eps=1e-6):
        super().__init__()
        self.pqmf = PQMF(subbands, taps, cutoff_ratio, beta)
        self.register_buffer('mean_ema', torch.zeros([subbands]))
        self.register_buffer('var_ema', torch.ones([subbands]))
        self.register_buffer('batch', torch.zeros(()))
        self.update_batches = update_batches
        self.eps = eps

    @autocast('cuda', enabled=False)
    def project_sample(self, x: torch.Tensor):
        audio_subbands = self.pqmf.analysis(x)
        if self.training and self.batch < self.update_batches:
            audio_subbands_mean = [torch.mean(x.float()) for x in torch.unbind(audio_subbands, dim=1)]
            audio_subbands_var = [torch.var(x.float()) for x in torch.unbind(audio_subbands, dim=1)]
            current_mean = torch.stack(audio_subbands_mean).detach().to(self.mean_ema.dtype)
            current_var = torch.stack(audio_subbands_var).detach().to(self.mean_ema.dtype)
            if self.batch == 0:
                self.mean_ema.copy_(current_mean)
                self.var_ema.copy_(current_var)
            else:
                self.mean_ema.lerp_(current_mean, 0.01)
                self.var_ema.lerp_(current_var, 0.01)
            self.batch += 1
        # Convert buffers once to match input dtype
        mean = self.mean_ema.to(audio_subbands.dtype).unsqueeze(-1)
        std = torch.sqrt(self.var_ema.to(audio_subbands.dtype).unsqueeze(-1) + self.eps)
        audio_subbands = (audio_subbands - mean) / std
        audio = self.pqmf.synthesis(audio_subbands)
        return audio

    @autocast('cuda', enabled=False)
    def return_sample(self, x: torch.Tensor):
        x_subbands = self.pqmf.analysis(x)
        # Convert buffers once to match input dtype
        mean = self.mean_ema.to(x_subbands.dtype).unsqueeze(-1)
        std = torch.sqrt(self.var_ema.to(x_subbands.dtype).unsqueeze(-1) + self.eps)
        x_subbands = x_subbands * std + mean
        x = self.pqmf.synthesis(x_subbands)
        return x


class MeanStdProcessor(nn.Module):
    def __init__(self, dim, update_batches=100000, eps=1e-6):
        super().__init__()
        self.register_buffer('mean_ema', torch.zeros([dim]))
        self.register_buffer('var_ema', torch.ones([dim]))
        self.register_buffer('batch', torch.zeros(()))
        self.update_batches = update_batches
        self.eps = eps

    @autocast('cuda', enabled=False)
    def project_sample(self, x: torch.Tensor):
        if self.training and self.batch < self.update_batches:
            mean = torch.mean(x.float(), dim=(0, 2)).detach().to(self.mean_ema.dtype)
            var = torch.var(x.float(), dim=(0, 2)).detach().to(self.mean_ema.dtype)
            if self.batch == 0:
                self.mean_ema.copy_(mean)
                self.var_ema.copy_(var)
            else:
                self.mean_ema.lerp_(mean, 0.01)
                self.var_ema.lerp_(var, 0.01)
            self.batch += 1
        # Convert buffers once to match input dtype
        mean = self.mean_ema.to(x.dtype)[None, :, None]
        std = torch.sqrt(self.var_ema.to(x.dtype)[None, :, None] + self.eps)
        return (x - mean) / std

    @autocast('cuda', enabled=False)
    def return_sample(self, x: torch.Tensor):
        # Convert buffers once to match input dtype
        mean = self.mean_ema.to(x.dtype)[None, :, None]
        std = torch.sqrt(self.var_ema.to(x.dtype)[None, :, None] + self.eps)
        return x * std + mean
