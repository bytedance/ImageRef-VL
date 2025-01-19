# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jsonlines
import re
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rslt_file", default="", type=str)
    args = parser.parse_args()
    return args

args = parse_args()


def extract_rating(text):
    pattern = r'\[\[(\d+)\]\]'
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    else:
        return None
    
with jsonlines.open(args.rslt_file) as reader:
    rslt = list(reader)

scores = [extract_rating(i["response"]) for i in rslt]
scores = [i for i in scores if i is not None]

print(np.mean(scores))
