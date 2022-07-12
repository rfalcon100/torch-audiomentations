import logging
import torch
import torchaudio
from typing import Optional
from torch import Tensor
from torch.nn.functional import pad

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio
from ..utils.object_dict import ObjectDict


class SpecTimeMasking(BaseWaveformTransform):

    """
    Wrapper for the time_masking augmentation from torchaudio. This is so that it can be used as part of an augmentation set.

    Reference:
    https://pytorch.org/audio/stable/transforms.html#timemasking
    """

    supported_modes = {"per_batch", "per_example"}
    requires_sample_rate = False

    def __init__(
        self,
        time_mask_param: int,
        iid_masks: bool = False,
        p_proportion: float = 1.0,
        mask_value: float = 0.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        """
        param time_mask_param (int) – maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param).
        param iid_masks (bool, optional) – whether to apply different masks to each example/channel in the batch. (Default: False) This option is applicable only when the input tensor is 4D.
        param p (float, optional) – maximum proportion of time steps that can be masked. Must be within range [0.0, 1.0]. (Default: 1.0)
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.time_mask_param = time_mask_param
        self.iid_masks = iid_masks
        self.p_proportion = p_proportion
        self.mask_value = mask_value

        self.tform = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param,
                                                       iid_masks=self.iid_masks,
                                                       p=p_proportion)

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        pass

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        output_samples = []

        #for i in range(samples.shape[0]):
            #sample = samples[i][:, :]
            #output_samples.append(self.tform(sample, mask_value=self.mask_value))
        output_samples = (self.tform(samples, mask_value=self.mask_value))

        return ObjectDict(
            #samples=torch.cat(output_samples, dim=0),
            samples=output_samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
