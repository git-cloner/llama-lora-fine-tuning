# 在单块16G的推理卡上微调复现vicuna-7b

## 1、概述

对FaceBook/LLaMA的微调一般有两种方案，一种是斯坦福的alpaca系列，一种是基于shareGPT语料的Vicuna方向。Vicuna采用的是多轮对话的语料，训练效果要好于默认为单轮对话的alpaca，所以要在Llama基础上微调，建议首选Vicuna的方式。

关于两种微调方式，在以下项目中都有详细描述（FastChat中对lora模式的说明比较简单）。

https://github.com/tloen/alpaca-lora

https://github.com/lm-sys/FastChat

Alpaca-lora的训练对内存要求不高，大概12G的2080Ti，就可以支持，但训练多轮会话的类Vicuna的模型，则对显存的要求比较高，Vicuna模式的训练至少需要24G显存【官方建议是4 * V100(32G)】。

如果有足够高端的显卡，只要跟随文件训练即可，如果只有16G的显卡，又想自己定制语料复现Vicuna模型，那就得想很多办法，不断地降精度，从32位降到半精度16位，再从16位降成8位，再辅以一些加速训练方法才能达到目的。

## 2、微调方案

- 采用lora方式只训练一部分参数
- 基础模型采用半精度llama-7b-hf
- 使用load_in_8bit装载基础模型
- 采用peft技术微调
- 采用bitsandbytes加速

所以本文在FastChat的基础上，修改lora训练代码，使用shareGPT语料，在16G显存的推理卡进行微调，大概占用显存13G左右。

- 操作系统 centos或ubuntu

- NVIDA P100或T4：16G显存或以上

- CUDA、conda：https://zhuanlan.zhihu.com/p/597063490

## 3、微调过程

### 3.1、安装依赖环境

#### 3.1.1 下载源码

```bash
git clone https://github.com/git-cloner/llama-lora-fine-tuning
cd llama-lora-fine-tuning
```

#### 3.1.2  安装微调依赖环境

##### 3.1.2.1 install pkg-config

```bash
wget https://pkg-config.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar -zxvf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2
./configure --with-internal-glib
make -j4
make check
sudo make install
```

##### 3.1.2.2 install libicu

```bash
wget https://mirrors.aliyun.com/blfs/conglomeration/icu/icu4c-73_1-src.tgz
tar xf icu4c-73_1-src.tgz
cd icu/source
./configure
make
make check
sudo make install
sudo ldconfig
```

##### 3.1.2.3 安装驱动及conda

https://zhuanlan.zhihu.com/p/597063490

##### 3.1.2.4 install packages

```bash
conda create -n llama-lora python=3.10
conda activate llama-lora
pip3 install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

### 3.2、准备Llama模型

可以用以下办法下载原始模型后转换为半精度，也可以从https://huggingface.co/decapoda-research/llama-7b-hf直接下载转换好的半精度模型。

#### 3.2.1 下载Llama模型

```bash
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1
pip3 install git+https://github.com/juncongmoo/pyllama -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
python -m llama.download --model_size 7B
```

#### 3.2.2 转换模型为huggingface格式

```bash
CUDA_VISIBLE_DEVICES=1 python3 ./convert_llama_weights_to_hf.py --input_dir ./pyllama_data --model_size 7B --output_dir ./pyllama_data/output/7B
```

### 3.3、整理语料

#### 3.3.1 语料下载

下载52k的ShareGPT：https://huggingface.co/datasets/RyokoAI/ShareGPT52K

其他语料参见：https://github.com/Zjh-819/LLMDataHub

下载的sg_90k_part1.json和sg_90k_part2.json放到data下

#### 3.3.2 合并语料文件

```bash
python3 fastchat/data/merge.py --in ./data/sg_90k_part1.json ./data/sg_90k_part2.json ./data/dummy_cn.json ./data/dummy_en.json --out ./data/sg_90k.json
```

#### 3.3.3 html转markdown

```bash
python3 fastchat/data/clean_sharegpt.py --in ./data/sg_90k.json --out ./data/sharegpt_clean.json
```

#### 3.3.4 去掉一些用不到的语言（可选）

```bash
python3 fastchat/data/optional_clean.py --in ./data/sharegpt_clean.json --out ./data/sharegpt_clean_1.json --skip-lang SOME_LANGUAGE_CODE
其中SOME_LANGUAGE_CODE的取值举例如下：
en - 英语
es - 西班牙语
fr - 法语
de - 德语
it - 意大利语
ja - 日语
ko - 朝鲜语
zh - 中文
ar - 阿拉伯语
ru - 俄语
pt - 葡萄牙语
nl - 荷兰语
```

#### 3.3.5 将长会话切分成短对话

```shell
CUDA_VISIBLE_DEVICES=1 python3 fastchat/data/split_long_conversation.py --in ./data/sharegpt_clean.json --out ./data/sharegpt_clean_split.json --model-name ./pyllama_data/output/7B
```

### 3.4、微调

#### 3.4.1 微调命令

```bash
# 禁用wandb
wandb disabled
# 为了防止ssh终端断开导致训练中止，训练可在后台运行（去掉#三处注释即可在后台运行）
# 如果有多颗GPU，可以用--num_gpus参数
CUDA_VISIBLE_DEVICES=0,1 \ #nohup \
deepspeed --num_gpus=2 fastchat/train/train_lora.py \
    --deepspeed ./deepspeed-config.json \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path ./pyllama_data/output/7B \
    --data_path ./data/sharegpt_clean_split.json \
    --fp16 True \
    --output_dir ./output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 14 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --gradient_checkpointing True #>> lora.log  2>&1 &
  # 如果在后台运行，则tail lora.log查看训练进度
  tail -f lora.log
```

#### 3.4.2 微调性能

在P100（16G）上进行微调，占用内存13.5G，在训练一轮的情况下，需要120个小时，大约5天时间，还是非常耗时时，形成的模型效果也有待验证。

model_max_length会影响到训练的时长，如果设成1024，比2048的时长减少一半，但会影响到推理效果。

#### 3.4.3 A100微调命令

单块A100微调的参数如下，大约需要16小时。

```bash
deepspeed fastchat/train/train_lora.py \
    --deepspeed ./deepspeed-config.json \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path ./pyllama_data/output/7B \
    --data_path ./data/sharegpt_clean_split.json \
    --fp16 True \
    --output_dir ./output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 56 \
    --per_device_eval_batch_size 56 \
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True
```

## 4、测试模型

### 4.1 模型位置

训练好的lora peft模型由adapter_config.json、adapter_model.bin和trainer_state.json组成。下面是peft和原模型的目录结构。

```bash
model
───llama-peft
│      adapter_config.json
│      adapter_model.bin
│      trainer_state.json
│
└──llama_7b
        config.json
        generation_config.json
        pytorch_model-00001-of-00002.bin
        pytorch_model-00002-of-00002.bin
        pytorch_model.bin.index.json
        special_tokens_map.json
        tokenizer.json
        tokenizer.model
        tokenizer_config.json
```

### 4.2 测试生成

```bash
CUDA_VISIBLE_DEVICES=0  python generate.py  --base_model ./model/llama-7b --lora_weights ./model/llama-peft
```

