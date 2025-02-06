import julius
import torch
import numpy as np
from torch import Tensor
from typing import Optional


from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class Quantization(BaseWaveformTransform):
    """
    Applies a quantization transform to a signal. 
    This is a similar idea to a mu-law endoding, but the bins are linearly spaced.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = False
    requires_target = False

    def __init__(
        self,
        max_win_len: float = 1.9, 
        min_win_len: float = 1.8,
        win_len_unit: str = 'seconds', 
        n_bins: int = 10,
        min_value: float = -1.0,
        max_value: float = 1.0,
        n_bins_jitter: float = 3.7,
        n_windows: int = 1,
        mode: str = "per_channel",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param max_win_len: Minimum length of the window, in either samples, seconds or fraction
        :param min_win_len: Maximum length of the window, in either samples, seconds or fraction
        :param win_len_unit: "seconds", "samples", "fraction"
        :param n_bins: number of bins to quantize the signal
        :param min_value: Minimum value of the input signals, used to scale the quantization properly
        :param max_value: Maximum value of the input signals, used to scale the quantization properly
        :param n_bins_jitter: Adds a small offset to the number of bins, can be non integet. E.g. 1.5 with n_bins = 10, we sample bins from [8.5, 11.5]
        :param:n_windows: Number of windows to apply the quantization. NOTE this is not supported yet.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        self.max_win_len = max_win_len
        self.min_win_len = min_win_len
        self.win_len_unit = win_len_unit
        self.n_bins = n_bins
        self.min_value = min_value
        self.max_value = max_value
        self.n_bins_jitter = n_bins_jitter
        self.n_windows = n_windows
        
        if self.min_value > self.max_value:
            raise ValueError("min_value must not be greater than max_value")
        if self.n_bins <= 0:
            raise ValueError("n_bins must not be greater than 0")

        self.cached_lpf = None

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :params samples: (batch_size, num_channels, num_samples)
        """
        batch_size, channels, num_samples = samples.shape

        if self.win_len_unit == "samples":
            min_win_len_samples = self.min_win_len
            max_win_len_samples = self.max_win_len
        elif self.win_len_unit == "fraction":
            min_win_len_samples = int(round(self.min_win_len * samples.shape[-1]))
            max_win_len_samples = int(round(self.max_win_len * samples.shape[-1]))
        elif self.win_len_unit == "seconds":
            min_win_len_samples = int(round(self.min_win_len * sample_rate))
            max_win_len_samples = int(round(self.max_win_len * sample_rate))
        else:
            raise ValueError("Invalid win_len_unit")

        self.transform_parameters["win_lens"] = torch.randint(
            low=min_win_len_samples,
            high=max_win_len_samples + 1,
            size=(batch_size, channels),
            dtype=torch.int32,
            device=samples.device,)
        
        # NOTE: we sample center_id in the apply_transform so that we dont have to iterate batch and channels twice

        # n_bins with jitter if needed
        self.transform_parameters["n_bins"] = self.n_bins + (torch.rand(
            size=(batch_size, channels),
            dtype=torch.float32,
            device=samples.device,)  * (2*self.n_bins_jitter) - self.n_bins_jitter)
                
    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        #win_len = self.transform_parameters['win_lens']
        #center_id = self.transform_parameters['center_id']
        #n_bins = self.transform_parameters['n_bins']

        for b in range(batch_size):
            for c in range(num_channels):
                sample = samples[b,c,:]
                win_len = self.transform_parameters['win_lens'][b, c]
                n_bins = self.transform_parameters['n_bins'][b, c]
                center_id = torch.randint(low=(win_len.item() // 2) + 1, 
                                          high=(samples.shape[-1] - (win_len.item() // 2)) - 1,
                                          size=(1,), 
                                          dtype=torch.int32,
                                          device=samples.device,)
                
                start_id = center_id - (win_len // 2)
                end_id = center_id + (win_len // 2)
                bin_size = (self.max_value - self.min_value) / (n_bins - 1)[..., None, None]
                new_sample = (sample - self.min_value) / bin_size
                new_sample = torch.round(new_sample)
                new_sample *= bin_size
                new_sample += self.min_value
                sample[..., start_id:end_id] = new_sample[..., start_id:end_id]
                #quantized_samples.append(sample)

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

    def apply_transform_before_random(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        min_win = int(self.min_win_len_seconds * sample_rate)
        max_win = int(self.max_win_len_seconds * sample_rate)
        win_len = torch.randint(min_win, max_win, size=(1,)) 
        center_id = torch.randint((win_len // 2) + 1, (samples.shape[-1] - (win_len // 2)) - 1, size=(1,))
        start_id = center_id - (win_len // 2)
        end_id = center_id + (win_len // 2)

        # Apply jitter to bins
        n_bins = self.n_bins + (torch.rand(size=(1,)) * (2*self.n_bins_jitter) - self.n_bins_jitter)
        bin_size = (self.max_value - self.min_value) / (n_bins - 1)
        #for i in range(batch_size):
        new_samples = (samples - self.min_value) / bin_size
        new_samples = torch.round(new_samples)
        new_samples *= bin_size
        new_samples += self.min_value

        samples[..., start_id:end_id] = new_samples[..., start_id:end_id]

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
    
    def apply_transform_release_candidate(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        min_win = int(self.min_win_len_seconds * sample_rate)
        max_win = int(self.max_win_len_seconds * sample_rate)
        #center_id = torch.randint((max_win // 2) + 1, (samples.shape[-1] - (max_win // 2)) - 1, size=(1,))
        win_len = torch.randint(min_win, max_win, size=(1,)) 
        center_id = torch.randint((win_len // 2) + 1, (samples.shape[-1] - (win_len // 2)) - 1, size=(1,))
        start_id = center_id - (win_len // 2)
        end_id = center_id + (win_len // 2)

        #n_bins = self.n_bins + torch.randint(-self.n_bins_jitter, self.n_bins_jitter, size=(1,))
        n_bins = self.n_bins + (torch.rand(size=(1,)) * (2*self.n_bins_jitter) - self.n_bins_jitter)
        print(f'n_bins: {n_bins}')
        print(f'win_len: {win_len}, \t center_id = {center_id}')
        print(f'start_id: {start_id}, \t end_id = {end_id}')
        #start_id = 0
        #end_id = 90000
        bin_size = (self.max_value - self.min_value) / (n_bins - 1)
        #for i in range(batch_size):
        new_samples = (samples - self.min_value) / bin_size
        print(f'min: {new_samples.min()}  mean:{new_samples.mean()} max: {new_samples.max()}')
        new_samples = torch.round(new_samples)
        print(f'min: {new_samples.min()}  mean:{new_samples.mean()} max: {new_samples.max()}')
        new_samples *= bin_size #+ self.min_value
        print(f'min: {new_samples.min()}  mean:{new_samples.mean()} max: {new_samples.max()}')
        new_samples += self.min_value

        samples[..., start_id:end_id] = new_samples[..., start_id:end_id]
        #samples = new_samples

        if False:
            start_id = 0
            end_id = 90000
            bin_size = (self.max_value - self.min_value) / (self.n_bins - 1)
            #for i in range(batch_size):
            samples = (samples - self.min_value) / bin_size
            samples = torch.round(samples)
            samples *= bin_size #+ self.min_value
            samples += self.min_value

            #samples[..., start_id:end_id] = new_samples[..., start_id:end_id]
            #samples = new_samples

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
    
    def apply_transform_basic(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        start_id = torch.randint(0, samples.shape[-1], size=(1,))
        end_id = torch.randint(start_id, samples.shape[-1], size=(1,))

        #start_id = 0
        #end_id = 90000
        bin_size = (self.max_value - self.min_value) / (self.n_bins - 1)
        #for i in range(batch_size):
        new_samples = (samples - self.min_value) / bin_size
        new_samples = torch.round(new_samples)
        new_samples *= bin_size #+ self.min_value
        new_samples += self.min_value

        samples[..., start_id:end_id] = new_samples[..., start_id:end_id]
        #samples = new_samples

        if False:
            start_id = 0
            end_id = 90000
            bin_size = (self.max_value - self.min_value) / (self.n_bins - 1)
            #for i in range(batch_size):
            samples = (samples - self.min_value) / bin_size
            samples = torch.round(samples)
            samples *= bin_size #+ self.min_value
            samples += self.min_value

            #samples[..., start_id:end_id] = new_samples[..., start_id:end_id]
            #samples = new_samples

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
