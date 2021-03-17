import time
import json
import codecs
import os
import numpy as np
import random

import torch


def iterative_support(func, query):
    if isinstance(query, (list, tuple, set)):
        return [iterative_support(func, i) for i in query]
    return func(query)


def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)


def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list


def f1_score(sent_list, pred_list, gold_list, script_path):
    fn_out = 'eval_%04d.txt' % random.randint(0, 10000)
    if os.path.isfile(fn_out):
        os.remove(fn_out)

    text_file = open(fn_out, mode='w')
    for i, words in enumerate(sent_list):
        tags_1 = gold_list[i]
        tags_2 = pred_list[i]
        for j, word in enumerate(words):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
        text_file.write('\n')
    text_file.close()

    cmd = 'perl %s < %s' % (script_path, fn_out)
    msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    msg += ''.join(os.popen(cmd).readlines())
    time.sleep(1.0)
    if fn_out.startswith('eval_') and os.path.exists(fn_out):
        os.remove(fn_out)
    return float(msg.split('\n')[3].split(':')[-1].strip())


def iob_tagging(entities, s_len):
    tags = ["O"] * s_len

    for el, er, et in entities:
        for i in range(el, er + 1):
            if i == el:
                tags[i] = "B-" + et
            else:
                tags[i] = "I-" + et
    return tags


def conflict_judge(line_x, line_y):
    if line_x[0] == line_y[0]:
        return True
    if line_x[0] < line_y[0]:
        if line_x[1] >= line_y[0]:
            return True
    if line_x[0] > line_y[0]:
        if line_x[0] <= line_y[1]:
            return True
    return False


def extract_json_data(file_path):
    with codecs.open(file_path, "r", "utf-8") as fr:
        dataset = json.load(fr)
    return dataset
