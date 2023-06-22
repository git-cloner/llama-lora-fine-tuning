import json
import argparse


def change_format(args):
    with open(args["in_file"], 'r+', encoding='utf-8') as f:
        data = json.load(f)
    j = len(data)
    i = 0
    new_data = []
    temp_instruction = ""
    for items in data:
        i = i + 1
        print(str(i) + ' of ' + str(j))
        for item in items["conversations"]:
            if item["from"] == "human":
                temp_instruction = item["value"]
            else:
                new_item = {}
                new_item["instruction"] = temp_instruction
                new_item["input"] = ""
                new_item["output"] = item["value"]
                new_data.append(new_item)
    with open(args["out_file"], 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str,
                        default="./data/alpaca_sharegpt.json")
    args = parser.parse_args()
    change_format(vars(args))
