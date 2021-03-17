from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer

from misc import extract_json_data
from misc import iob_tagging, f1_score


class UnitAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, UNK_SIGN = "[PAD]", "[UNK]"

    def __init__(self, source_path):
        self._tokenizer = BertTokenizer.from_pretrained(source_path, do_lower_case=False)

    def tokenize(self, item):
        return self._tokenizer.tokenize(item)

    def index(self, items):
        return self._tokenizer.convert_tokens_to_ids(items)


class LabelAlphabet(object):

    def __init__(self):
        super(LabelAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


def corpus_to_iterator(file_path, batch_size, if_shuffle, label_vocab=None):
    material = extract_json_data(file_path)
    instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in material]

    if label_vocab is not None:
        label_vocab.add("O")
        for _, u in instances:
            for _, _, l in u:
                label_vocab.add(l)

    class _DataSet(Dataset):

        def __init__(self, elements):
            self._elements = elements

        def __getitem__(self, item):
            return self._elements[item]

        def __len__(self):
            return len(self._elements)

    def distribute(elements):
        sentences, entities = [], []
        for s, e in elements:
            sentences.append(s)
            entities.append(e)
        return sentences, entities

    wrap_data = _DataSet(instances)
    return DataLoader(wrap_data, batch_size, if_shuffle, collate_fn=distribute)


class Procedure(object):

    @staticmethod
    def train(model, dataset, optimizer):
        model.train()
        time_start, total_penalties = time.time(), 0.0

        for batch in tqdm(dataset, ncols=50):
            loss = model.estimate(*batch)
            total_penalties += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_con = time.time() - time_start
        return total_penalties, time_con

    @staticmethod
    def test(model, dataset, eval_path):
        model.eval()
        time_start = time.time()
        seqs, outputs, oracles = [], [], []

        for sentences, segments in tqdm(dataset, ncols=50):
            with torch.no_grad():
                predictions = model.inference(sentences)

            seqs.extend(sentences)
            outputs.extend([iob_tagging(e, len(u)) for e, u in zip(predictions, sentences)])
            oracles.extend([iob_tagging(e, len(u)) for e, u in zip(segments, sentences)])

        out_f1 = f1_score(seqs, outputs, oracles, eval_path)
        return out_f1, time.time() - time_start
