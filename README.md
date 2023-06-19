[中文](https://github.com/git-cloner/llama-lora-fine-tuning/blob/main/README_cn.md)

# Fine-tuning vicuna-7b on a single 16G GPU

## 1. Overview

There are generally two schemes for fine-tuning FaceBook/LLaMA. One is Stanford's alpaca series, and the other is Vicuna based on shareGPT corpus. Vicuna uses multi-round dialogue corpus, and the training effect is better than alpaca which is defaulted to single-round dialogue. Therefore, it is recommended to fine-tune Llama based on Vicuna.
The two fine-tuning ways are described in detail in the following projects (the description of lora mode in FastChat is relatively simple). <br>
https://github.com/tloen/alpaca-lora <br>
https://github.com/lm-sys/FastChat <br>
Alpaca-lora has low memory requirements, about 12G 2080Ti can support, but training multi-round session models like Vicuna requires high GPU memory. Vicuna model training requires at least 24G GPU memory [official recommendation is 4 * V100 (32G)].
If you have a high-end graphics card, just follow the file to train. If you only have a 16G graphics card but want to customize the corpus to reproduce the Vicuna model, you have to think of many ways to continuously reduce the precision from 32 bits to half precision 16 bits, then from 16 bits to 8 bits, and accelerate the training method to achieve the goal.

## 2. Fine-tuning method

• Use lora method to train only part of the parameters <br>
• The basic model adopts half-precision llama-7b-hf <br>
• Use load_in_8bit to load the basic model <br>
• Use peft technology for fine-tuning <br>
• Use bitsandbytes to accelerate <br>
Then we based on FastChat, this article modifies the lora training code, uses the shareGPT corpus, and fine-tunes on a 16G card, occupying about 13G of GPU memory. <br>
• Operating system: centos or ubuntu <br>
• NVIDA P100 or T4: 16G GPU memory or above <br>
• CUDA, conda

## 3.Fine-tuning process

### 3.1 Install dependency environment 

#### 3.1.1 Download source code

```bash
git clone https://github.com/git-cloner/llama-lora-fine-tuning
cd llama-lora-fine-tuning
```

#### 3.1.2 Install fine-tuning dependency environment

##### 3.1.2.1 Install pkg-config

```bash
wget https://pkg-config.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar -zxvf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2
./configure --with-internal-glib  
make -j4
make check  
sudo make install
```

##### 3.1.2.2 Install libicu

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

##### 3.1.2.3 Install packages

```bash
conda create -n llama-lora python=3.10
conda activate llama-lora
pip3 install -r requirements.txt
```

### 3.2 Prepare the Llama model

You can download the original model and convert it to half precision, or download the converted half precision model directly from https://huggingface.co/decapoda-research/llama-7b-hf.

#### 3.2.1 Download the Llama model

```bash
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1
pip3 install git+https://github.com/juncongmoo/pyllama -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
python -m llama.download --model_size 7B
```

#### 3.2.2 Convert the model to huggingface format

```bash
CUDA_VISIBLE_DEVICES=1 python3 ./convert_llama_weights_to_hf.py --input_dir ./pyllama_data --model_size 7B --output_dir ./pyllama_data/output/7B
```

### 3.3 Organize the corpus 

#### 3.3.1 Corpus download

```ash
Download 52k ShareGPT: https://huggingface.co/datasets/RyokoAI/ShareGPT52K
Other corpora refer to: https://github.com/Zjh-819/LLMDataHub
Download sg_90k_part1.json and sg_90k_part2.json into the data directory
```

#### 3.3.2 Merge corpus files

```bash
python3 fastchat/data/merge.py --in ./data/sg_90k_part1.json ./data/sg_90k_part2.json ./data/dummy_cn.json ./data/dummy_en.json --out ./data/sg_90k.json
```

#### 3.3.3 Html to Markdown

```bash
python3 fastchat/data/clean_sharegpt.py --in ./data/sg_90k.json --out ./data/sharegpt_clean.json
```

#### 3.3.4 Remove some unused languages (optional)

```bash
python3 fastchat/data/optional_clean.py --in ./data/sharegpt_clean.json --out ./data/sharegpt_clean_1.json --skip-lang SOME_LANGUAGE_CODE 
The values of SOME_LANGUAGE_CODE are as follows:
en - English
es - Spanish 
fr - French
de - German
it - Italian
ja - Japanese
ko - Korean 
zh - Chinese
ar - Arabic
ru - Russian
pt - Portuguese
nl - Dutch
```

#### 3.3.5 Split long conversations into short dialogues

```bash
CUDA_VISIBLE_DEVICES=1 python3 fastchat/data/split_long_conversation.py --in ./data/sharegpt_clean.json --out ./data/sharegpt_clean_split.json --model-name ./pyllama_data/output/7B
```

### 3.4 Fine-tuning

#### 3.4.1 Fine-tuning command

 

```bash
# Disable wandb 
wandb disabled 
# In order to prevent the SSH terminal from disconnecting and stopping the training, the training can run in the background (remove the # in three places to run in the background)
# If you have multiple GPUs,using --num_gpus parameter
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
  --gradient_checkpointing True #>> lora.log 2>&1 &
# If running in the background, tail lora.log to check the training progress 
tail -f lora.log
```

#### 3.4.2 Fine-tuning performance

Fine-tuning on P100 (16G) occupies 13.5G of memory. In the case of one round of training, it takes 120 hours, about 5 days, which is still very time-consuming. The effect of the resulting model needs to be verified. 
model_max_length will affect the training time. If set to 1024, the time will be halved compared to 2048, but it will affect the inference effect. 

#### 3.4.3 Fine-tuning on A100

fine-tuning on single A100 and  take about 16 hours.

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

## 4、Test trained model

### 4.1 model file structure

The trained LoRa peft model consists of adapter_config.json, adapter_model.bin, and trainer_state.json. Below is the file structure of peft and the original llama model.

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

### 4.2 test generate

```bash
CUDA_VISIBLE_DEVICES=0  python generate.py  --base_model ./model/llama-7b --lora_weights ./model/llama-peft
```

