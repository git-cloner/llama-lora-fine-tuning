import json
from transformers import pipeline
import torch

translator_en2zh = None


def load_model():
    global translator_en2zh
    if translator_en2zh is None:
        translator_en2zh = pipeline(
            "translation", model="Helsinki-NLP/opus-mt-en-zh", device=0)
    print("load model ok")


def trans_en2zh(input):
    global translator_en2zh
    return translator_en2zh(input)


def trans_dummy():
    with open('./data/dummy_en.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
    j = len(data)
    i = 0
    for items in data:
        i = i + 1
        print(str(i) + ' of ' + str(j))
        items['id'] = 'cn_' + items['id']
        for item in items["conversations"] :
            item["value"] = trans_en2zh(item['value'])[0]['translation_text']
    with open('./data/dummy_cn.json', 'w', encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False,indent=2)


if __name__ == "__main__":
    print("torch.cuda.is_available:", torch.cuda.is_available())
    load_model()
    trans_dummy()
