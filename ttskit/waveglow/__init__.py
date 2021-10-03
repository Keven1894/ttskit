#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/10
"""
### waveglow
#### 声码器（vocoder）
- 把语音频谱数据转为语音信号。
- 语音信号和语音频谱数据并不是简单可以相互转换的数据，语音转为频谱是有信息丢失的，但是频谱记录了语音最主要的信息，可以通过其他技术手段把语音频谱尽可能逼真地逆变为语音信号，声码器就是这样的技术。
- 声码器是把声音特征转为语音信号的技术。
- 在语音合成任务中，声码器是负责把语音频谱转为语音信号。
- 通常语音频谱记录的语音信息并不是全面的，例如mel频谱只是记录了部分频段的幅度信息，缺失了相位信息，而且许多频率的信息也丢失了。
- 而声码器模型就是要从这样的频谱中尽可能准确全面地还原出语音信号。
- 现在通常的方案是用深度学习的方法来解决，针对语音特征和语音信号的关系进行建模。

#### Waveglow声码器
- WaveGlow是英伟达团队提出的一种依靠流的从梅尔频谱图合成高质量语音的模型。
- Waveglow贡献是基于流的网络，结合了Glow和WaveNet的想法，因此网络称为WaveGlow 。
- WaveGlow是一个生成模型，通过从分布采样中生成音频。
- WaveGlow易于实施，仅使用单个网络进行训练，仅使用似然损失函数进行训练。
- WaveGlow是兼顾生成速度快、生成质量高、稳定性强的模型。
"""
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).absolute().parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
