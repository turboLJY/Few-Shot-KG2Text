import codecs
import json
import torch
import random
import spacy
import math
import pickle
import os
import numpy as np
from torch.utils.data import Dataset


class Vocab(object):
    def __init__(self, filename):
        self._token2idx = pickle.load(open(filename, "rb"))
        self._idx2token = {v: k for k, v in self._token2idx.items()}

    def size(self):
        return len(self._idx2token)

    def convert_ids_to_tokens(self, x):
        if isinstance(x, list):
            return [self.convert_ids_to_tokens(i) for i in x]
        return self._idx2token[x]

    def convert_tokens_to_ids(self, x):
        if isinstance(x, list):
            return [self.convert_tokens_to_ids(i) for i in x]
        return self._token2idx[x]


class NLP:
    def __init__(self):
        self.nlp = spacy.load('../../en_core_web_sm-2.3.1', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        text = ' '.join(text.split())
        if lower:
            text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


class S2SDataset(Dataset):
    """Dataset for sequence-to-sequence generative models, i.e., BART"""

    def __init__(self, data_dir, dataset, tokenizer, node_vocab, relation_vocab, num_samples, usage):
        self.data_dir = data_dir
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.node_vocab = node_vocab
        self.relation_vocab = relation_vocab
        self.num_samples = num_samples
        self.usage = usage
        self.input_nodes, self.input_edges, self.input_types, self.output_ids, self.pointer, \
            self.pairs, self.relations, self.positions, self.descriptions = self.prepare_data()

    def __len__(self):
        return len(self.input_nodes)

    def __getitem__(self, idx):
        return self.input_nodes[idx], self.input_edges[idx], self.input_types[idx], self.output_ids[idx], \
                    self.pointer[idx], self.pairs[idx], self.relations[idx], self.positions[idx], self.descriptions[idx]

    def prepare_data(self):
        """
        read corpus file
        """
        try:
            data = torch.load(os.path.join(self.data_dir, self.dataset, '{}_{}.tar'.format(self.usage, self.num_samples)))
            input_nodes, input_edges, input_types, output_ids, pointer, input_pairs, relations, positions, descriptions = \
                data["nodes"], data["edges"], data["types"], data["outputs"], data["pointer"], data["pairs"], \
                data["relations"], data["positions"], data["descriptions"]
        except FileNotFoundError:
            all_data = []
            data_file = os.path.join(self.data_dir, self.dataset, '{}_processed.json'.format(self.usage))
            with codecs.open(data_file, "r") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    sentence = data["target_txt"]
                    if sentence.strip() == "":
                        continue
                    desc = data["description"]
                    if len(desc) > 512:
                        continue
                    all_data.append(data)

            if self.num_samples != "all":
                all_data = random.sample(all_data, int(self.num_samples))

            input_nodes, input_edges, input_types, output_ids, pointer, \
                input_pairs, relations, positions, descriptions = [], [], [], [], [], [], [], [], []
            for data in all_data:
                nodes = self.node_vocab.convert_tokens_to_ids(data["split_nodes"])
                edges = data["split_edges"]
                types = self.relation_vocab.convert_tokens_to_ids(data["split_types"])

                outputs = self.tokenizer.convert_tokens_to_ids(["<s>"] + data["plm_output"] + ["</s>"])
                copy_pointer = [0] + data["pointer"] + [0]
                assert len(outputs) == len(copy_pointer), "The length of outputs and pointer should be matched."

                pairs = data["pairs"]
                rela = self.relation_vocab.convert_tokens_to_ids(data["relations"])

                pos = data["positions"]
                desc = self.tokenizer.convert_tokens_to_ids(data["description"])
                assert len(types) == len(edges[0]), "The length of edges and types should be matched."
                input_nodes.append(nodes)
                input_edges.append(edges)
                input_types.append(types)
                output_ids.append(outputs)
                pointer.append(copy_pointer)
                input_pairs.append(pairs)
                relations.append(rela)
                positions.append(pos)
                descriptions.append(desc)
            data = {"nodes": input_nodes, "edges": input_edges, "types": input_types, "outputs": output_ids,
                    "pointer": pointer, "pairs": input_pairs, "relations": relations, "positions": positions,
                    "descriptions": descriptions}

            torch.save(data, os.path.join(self.data_dir, self.dataset, '{}_{}.tar'.format(self.usage, self.num_samples)))

        return input_nodes, input_edges, input_types, output_ids, pointer, input_pairs, relations, positions, descriptions
