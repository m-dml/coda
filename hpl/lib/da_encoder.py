from dataclasses import dataclass
from typing import Any

from hydra.conf import MISSING


@dataclass
class Unet:
    _target_: str = "unet.model.asymmetric_unet.Unet"
    in_channels: int = 2
    out_channels: int = 1

    encoder_dim: int = 2
    encoder_layers: Any = MISSING
    encoder_kwargs: Any = MISSING
    pooling_kwargs: Any = MISSING

    decoder_dim: int = 1
    decoder_layers: Any = MISSING
    decoder_kwargs: Any = MISSING
    upconv_kwargs: Any = MISSING

    output_block_kwargs: Any = MISSING
    global_pooling_type: str = "avg"
    global_pooling_dim: int = -2
