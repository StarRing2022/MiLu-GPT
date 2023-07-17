# MiLu-Uni
基于GPT2+BERT的语言模型，以少量的纯中文语料从头训练，验证小模型在ChatGPT类似友好能力

GPT2+BERTokenizer从头训练模型（50W闲聊等语料）

环境：<br>
WIN10+Torch1.31+Cuda11.6 <br>
transformer4.29<br>

主要代码说明：<br>
generate_dialogue_subset.py：产生小的子数据集<br>
preprocess.py：将txt格式数据集作序列化，得到pkl格式数据集<br>
train.py：从头训练模型<br>
interact.py: 使用从头训练模型给出回复<br>
gptbot.py：给出一个对话机器人<br>
generatedialogpt2.py：几种基于指令式或非指令式的回复格式<br>
