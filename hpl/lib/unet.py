from dataclasses import dataclass
from typing import Any


@dataclass
class GlobalMaxPool:
    _target_: str = "hpl.model.unet.GlobalMaxPool"
    dim: Any = -1


@dataclass
class GlobalAvgPool:
    _target_: str = "hpl.model.unet.GlobalAvgPool"
    dim: Any = -1


@dataclass
class ConvolutionalEncodingBlock:
    _target_: str = "hpl.model.unet.ConvolutionalEncodingBlock"
    _recursive_: bool = False
    convolution: Any = None
    activation: Any = None
    layers: Any = None
    pooling: Any = None
    batch_norm: Any = None
    dropout: Any = None


@dataclass
class ConvolutionalDecodingBlock:
    _target_: str = "hpl.model.unet.ConvolutionalDecodingBlock"
    _recursive_: bool = False
    convolution: Any = None
    activation: Any = None
    layers: Any = None
    upscale: Any = None
    batch_norm: Any = None
    dropout: Any = None


@dataclass
class ConvolutionalEncoder:
    _target_: str = "hpl.model.unet.encoder_builder"
    _recursive_: bool = False
    levels: Any = None
    block: Any = None


@dataclass
class ConvolutionalDecoder:
    _target_: str = "hpl.model.unet.decoder_builder"
    _recursive_: bool = False
    levels: Any = None
    block: Any = None


@dataclass
class Unet:
    _target_: str = "hpl.model.unet.Unet"
    _recursive_: bool = False
    encoder: Any = None
    decoder: Any = None
    output_convolution: Any = None
    global_pool: Any = None
