# author: kuangdd
# date: 2021/4/25
"""
### mellotron
#### 语音合成器（synthesizer）
- 把文本转为语音频谱数据。
- 语音合成器接收声音编码向量和文本数据，然后结合这些信息把文本转为语音频谱。
- 语音合成器的任务是把文本转为语音频谱，本质上是序列到序列的任务。
- 可以把文本看做一个一个字组成的序列，把语音频谱看做是由一个一个语音特征组成的序列，语音合成器就是把文字序列转为语音特征序列的桥梁。
- 语音合成器的关键就是怎样建立模型让文字准确的转为正确的读音，而且放在正确的位置，同时读音前后应当衔接自然，而且整个语音听起来也应当自然。
- 要实现这样的目标，应当做很有针对性的模型。

#### Mellotron语音合成器
- Mellotron是英伟达团队提出的语音合成模型，主要目标是做韵律风格转换和音乐生成。
- Mellotron可以更加精细化的调整韵律和音调，将基频信息引入模型刻画声调信息，基频是区别音高的主要元素。
- Mellotron模型的训练完全端到端化，不需要在数据集中含有音乐数据也可以生成音乐。
- Mellotron不需要对音调和文本进行人为的对齐就可以学到两者之间的对齐关系。
- Mellotron可以使用外部输入的注意力映射表，从而实现韵律迁移。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
