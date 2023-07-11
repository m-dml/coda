from dataclasses import dataclass
from typing import Any, Union


@dataclass
class PeriodicConv1d:
    _target_: str = "hpl.model.periodic.PeriodicConv1d"
    _recursive_: bool = True
    _convert_: Any = None
    _partial_: bool = False

    # dataclass args:
    in_channels: Union[int, None] = None
    out_channels: Union[int, None] = None
    kernel_size: Any = None
    stride: Any = 1
    padding: Any = 0
    dilation: Any = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"
    device: Any = None
    dtype: Any = None


@dataclass
class PeriodicConv2d:
    _target_: str = "hpl.model.periodic.PeriodicConv2d"
    _recursive_: bool = True
    _convert_: Any = None
    _partial_: bool = False

    # dataclass args:
    in_channels: Union[int, None] = None
    out_channels: Union[int, None] = None
    kernel_size: Union[int, None] = None
    stride: int = 1
    padding: Union[str, int] = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"
    device: Any = None
    dtype: Any = None


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
