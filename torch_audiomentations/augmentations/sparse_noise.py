import torch
from torch import Tensor
from typing import Optional
from math import ceil

from torch_audiomentations.utils.fft import rfft, irfft
from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import calculate_rms
from ..utils.io import Audio
from ..utils.object_dict import ObjectDict


def _gen_noise(f_decay, num_samples, sample_rate, device):
    """
    Generate colored noise with f_decay decay using torch.fft
    """
    noise = torch.normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        (sample_rate,),
        device=device,
    )
    spec = rfft(noise)
    mask = 1 / (
        torch.linspace(1, (sample_rate / 2) ** 0.5, spec.shape[0], device=device)
        ** f_decay
    )
    spec *= mask
    noise = Audio.rms_normalize(irfft(spec).unsqueeze(0)).squeeze()
    noise = torch.cat([noise] * int(ceil(num_samples / noise.shape[0])))
    return noise[:num_samples]


class AddSparseNoise(BaseWaveformTransform):
    """
    Add sparse colored noise to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = False
    requires_target = False

    def __init__(
        self,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        min_noise_ratio: float = 0.1,
        max_noise_ratio: float = 1.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param min_f_decay:
            defines the minimum frequency power decay (1/f**f_decay).
            Typical values are "white noise" (f_decay=0), "pink noise" (f_decay=1),
            "brown noise" (f_decay=2), "blue noise (f_decay=-1)" and "violet noise"
            (f_decay=-2)
        :param max_f_decay:
            defines the maximum power decay (1/f**f_decay) for non-white noises.
        :param min_noise_ratio:
            defines how much of the signal will have added noise.
            So if noise_ratio = 1.0, the whole signal will have noise. And if
            noise_Ratio = 0.0, the signal will have no noise added.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        :param target_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        if self.min_f_decay > self.max_f_decay:
            raise ValueError("min_f_decay must not be greater than max_f_decay")
        
        self.min_noise_ratio = min_noise_ratio
        self.max_noise_ratio = max_noise_ratio
        if self.min_noise_ratio > self.max_noise_ratio:
            raise ValueError("min_noise_ratio must not be greater than max_noise_ratio")
        

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = samples.shape

        # (batch_size, ) SNRs
        for param, mini, maxi in [
            ("snr_in_db", self.min_snr_in_db, self.max_snr_in_db),
            ("f_decay", self.min_f_decay, self.max_f_decay),
            ("noise_ratio", self.min_noise_ratio, self.max_noise_ratio),
        ]:
            if mini == maxi:
                self.transform_parameters[param] = torch.full(
                    size=(batch_size,),
                    fill_value=mini,
                    dtype=torch.float32,
                    device=samples.device,
                )
            else:
                dist = torch.distributions.Uniform(
                    low=torch.tensor(mini, dtype=torch.float32, device=samples.device),
                    high=torch.tensor(maxi, dtype=torch.float32, device=samples.device),
                    validate_args=True,
                )
                self.transform_parameters[param] = dist.sample(sample_shape=(batch_size,))

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        noise_base = torch.stack(
            [
                _gen_noise(
                    self.transform_parameters["f_decay"][i],
                    num_samples,
                    sample_rate,
                    samples.device,
                )
                for i in range(batch_size)
            ]
        )

        # (batch_size, num_channels)
        noise_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        #print(self.transform_parameters["sparsity"])
        num_nonzero = torch.max(torch.ones_like(self.transform_parameters["noise_ratio"], dtype=torch.int), 
                                torch.round(self.transform_parameters["noise_ratio"] * num_samples).to(torch.int))
        #print(num_nonzero)

        #print(f'noise_base: {noise_base.shape}')
        #noise_values = torch.randn(batch_size * num_channels, num_samples, device=samples.device) * noise_rms  # white noise
        noise_values = noise_base * noise_rms  # colored noise
        noise = torch.zeros_like(samples)
        for b in range(batch_size):
            indices = torch.randperm(num_samples)[:num_nonzero[b]]  # random permutation of indices, up to selected sparsity proportion
            dist = torch.distributions.Uniform( low=0, high=num_samples)

            #print(f'noise_rms: {noise_rms.shape}')
            #print(f'indices: {indices.shape}')
            #print(f'noise: {noise.shape}')
            #print(f'noise_values: {noise_values.shape}')
            noise[b, :,  indices] = noise_values[b, None, indices]
        samples = samples + noise

        # samples = samples + noise_rms.unsqueeze(-1 * noise.view(batch_size, 1, num_samples).expand(-1, num_channels, -1)
        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

def trash():

    if len(signal.shape) == 2:
        signal = signal[None, ...]
    batch, num_channels, num_timesteps = signal.shape
    num_nonzero = max(1, int(sparsity * num_timesteps))
    indices = torch.randperm(num_timesteps)[:num_nonzero]
    noise_values = torch.randn(batch, num_channels, num_nonzero, device=signal.device) * sigma
    #noise_values = torch.randn_like(signal) * sigma
    noise = torch.zeros_like(signal)
    noise[..., indices] = noise_values
    return signal + noise