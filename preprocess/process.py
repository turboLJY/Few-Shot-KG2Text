import json
import re
import unidecode
import random
import spacy
from transformers import RobertaTokenizer, BartTokenizer, BertTokenizer


class NLP:
    def __init__(self):
        self.nlp = spacy.load('../../../en_core_web_sm-2.3.1', disable=['ner', 'parser', 'tagger'])
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


nlp = NLP()


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t)
    new_d = " ".join(new_d)
    return new_d


def get_nodes(n):
    n = unidecode.unidecode(n.strip().lower())
    n = n.replace('-', ' ')
    n = n.replace('_', ' ')
    n = nlp.word_tokenize(n)

    return n


def get_relation(n):
    n = unidecode.unidecode(n.strip().lower())
    n = n.replace('-', ' ')
    n = n.replace('_', ' ')
    n = nlp.word_tokenize(n)

    return n


def get_text(txt, lower=True):
    if lower:
        txt = txt.lower()
    txt = unidecode.unidecode(txt.strip())
    txt = txt.replace('-', ' ')
    txt = nlp.word_tokenize(txt)

    return txt


def BFS(graph, s):
    queue = [s]
    seen = [s]
    node_seq = []
    while queue:
        vertex = queue.pop(0)
        adj_nodes = graph[vertex]
        for w in adj_nodes:
            if w not in seen:
                queue.append(w)
                seen.append(w)
        node_seq.append(vertex)
    return node_seq


def DFS(graph, s):
    stack = [s]
    seen = [s]
    node_seq = []
    while stack:
        vertex = stack.pop()
        adj_nodes = graph[vertex]
        for w in adj_nodes:
            if w not in seen:
                stack.append(w)
                seen.append(w)
        node_seq.append(vertex)
    return node_seq


bert_tokenizer = BartTokenizer.from_pretrained('../../pretrained_model/bart-large')
bart_tokenizer = BartTokenizer.from_pretrained('../../pretrained_model/bart-large')
# print(tokenizer.decoder_start_token_id)
# exit(0)

filename = ['train.json', 'valid.json', 'test.json']

for fn in filename:
    fin = open(fn, "r")
    data = json.load(fin)
    fin.close()

    fout = open(fn[:-5] + "_processed.json", "w")
    for d in data:
        new_dict = dict()

        # -------WebNLG dataset------
        valid = True
        ner_dict = {}
        ren_dict = {}
        for k, v in d['ner2ent'].items():
            en = get_nodes(v)
            if en == "":
                valid = False
            ner_dict[k] = en
            ren_dict[en] = k
        new_dict['ner2ent'] = ner_dict
        new_dict['ent2ner'] = ren_dict
        # -------WebNLG dataset------

        # -------Agenda dataset------
        # new_dict['title'] = get_text(d['title'])
        # types = d['types'].split()
        # valid = True
        # ner_dict = {}
        # ren_dict = {}
        # for idx in range(len(types)):
        #     en = get_nodes(d['entities'][idx])
        #     if en == "":
        #         valid = False
        #     ner = types[idx][1:-1].upper()
        #     ner_dict[ner + "_" + str(idx)] = en
        #     ren_dict[en] = ner + "_" + str(idx)
        # new_dict['ner2ent'] = ner_dict
        # new_dict['ent2ner'] = ren_dict
        # -------WebNLG dataset------

        # -------Genwiki dataset------
        # valid = True
        # ner_dict = {}
        # ren_dict = {}
        # for idx, ent in enumerate(d['entities']):
        #     ner = "ENT_" + str(idx)
        #     en = get_nodes(ent)
        #     if en == "":
        #         valid = False
        #     ner_dict[ner] = en
        #     ren_dict[en] = ner
        # new_dict['ner2ent'] = ner_dict
        # new_dict['ent2ner'] = ren_dict
        # -------WebNLG dataset------

        if not valid:
            continue

        temp = []
        serialization = []
        for tri in d['triples']:
            h = get_nodes(tri[0])
            t = get_nodes(tri[2])
            r = camel_case_split(get_relation(tri[1]))
            new_t = [h, r, t]
            temp.append(new_t)
            serialization.extend(["<Head>", h, "<Relation>", r, "<Tail>", t])
        new_dict['triples'] = temp
        new_dict['triples_serialization'] = serialization

        tokens = []
        for token in d['target'].split():
            if token.isupper() and '_' in token:
                tokens.append(token)
            else:
                tokens.append(token.lower())
        new_dict['target'] = get_text(' '.join(tokens), lower=False)

        try:
            tokens = []
            nodes = []
            for token in new_dict['target'].split():
                if token.isupper():
                    tokens.append(new_dict['ner2ent'][token])
                    if new_dict['ner2ent'][token] not in nodes:
                        nodes.append(new_dict['ner2ent'][token])
                else:
                    tokens.append(token)
            new_dict['target_txt'] = (' '.join(tokens)).lower()
        except KeyError:
            continue

        new_dict['plm_output'] = bart_tokenizer.tokenize(new_dict['target_txt'])

        test_output = []
        pointer = []
        idx = 0
        for tok in new_dict['target'].split():
            if idx == 0:
                if tok.isupper():
                    ent = bart_tokenizer.tokenize(new_dict['ner2ent'][tok])
                    test_output.extend(ent)
                    pointer.extend([1] * len(ent))
                else:
                    word = bart_tokenizer.tokenize(tok)
                    test_output.extend(word)
                    pointer.extend([0] * len(word))
            else:
                if tok.isupper():
                    ent = bart_tokenizer.tokenize(" " + new_dict['ner2ent'][tok])
                    test_output.extend(ent)
                    pointer.extend([1] * len(ent))
                else:
                    word = bart_tokenizer.tokenize(" " + tok)
                    test_output.extend(word)
                    pointer.extend([0] * len(word))
            idx += 1

        assert len(pointer) == len(new_dict['plm_output']), "The length of pointer and output are not equal!"
        assert test_output == new_dict['plm_output'], "The test output and plm output are not equal!"

        new_dict['pointer'] = pointer

        adject = dict()
        for t in new_dict['triples']:
            if t[0] not in nodes:
                nodes.append(t[0])
            if t[2] not in nodes:
                nodes.append(t[2])

            if t[0] not in adject:
                adject[t[0]] = []
            adject[t[0]].append(t[2])
            if t[2] not in adject:
                adject[t[2]] = []
            adject[t[2]].append(t[0])

        # for en in d['entities']:
        #     case_en = get_nodes(en)
        #     if case_en not in nodes:
        #         nodes.append(case_en)

        new_dict['nodes'] = nodes

        edges = [[], []]
        types = []
        for t in new_dict['triples']:
            hid = new_dict['nodes'].index(t[0])
            tid = new_dict['nodes'].index(t[2])
            edges[0].append(hid)
            edges[1].append(tid)
            types.append(t[1])
            edges[1].append(hid)
            edges[0].append(tid)
            types.append(t[1])
        new_dict['edges'] = edges
        new_dict['types'] = types

        #
        # bfs_edges = [[], []]
        # bfs_types = []
        # for t in new_dict['triples']:
        #     hid = new_dict['bfs_nodes'].index(t[0])
        #     tid = new_dict['bfs_nodes'].index(t[2])
        #     bfs_edges[0].append(hid)
        #     bfs_edges[1].append(tid)
        #     bfs_types.append(t[1])
        #     bfs_edges[1].append(hid)
        #     bfs_edges[0].append(tid)
        #     bfs_types.append(t[1])
        # new_dict['bfs_edges'] = bfs_edges
        # new_dict['bfs_types'] = bfs_types
        #
        # new_dict['dfs_nodes'] = new_dict['nodes']
        #
        # dfs_edges = [[], []]
        # dfs_types = []
        # for t in new_dict['triples']:
        #     hid = new_dict['dfs_nodes'].index(t[0])
        #     tid = new_dict['dfs_nodes'].index(t[2])
        #     dfs_edges[0].append(hid)
        #     dfs_edges[1].append(tid)
        #     dfs_types.append(t[1])
        #     dfs_edges[1].append(hid)
        #     dfs_edges[0].append(tid)
        #     dfs_types.append(t[1])
        # new_dict['dfs_edges'] = dfs_edges
        # new_dict['dfs_types'] = dfs_types
        #
        # new_dict['shuffle_nodes'] = nodes
        # random.shuffle(new_dict['shuffle_nodes'])
        #
        # shuffle_edges = [[], []]
        # shuffle_types = []
        # for t in new_dict['triples']:
        #     hid = new_dict['shuffle_nodes'].index(t[0])
        #     tid = new_dict['shuffle_nodes'].index(t[2])
        #     shuffle_edges[0].append(hid)
        #     shuffle_edges[1].append(tid)
        #     shuffle_types.append(t[1])
        #     shuffle_edges[1].append(hid)
        #     shuffle_edges[0].append(tid)
        #     shuffle_types.append(t[1])
        # new_dict['shuffle_edges'] = shuffle_edges
        # new_dict['shuffle_types'] = shuffle_types

        word_nodes = [bert_tokenizer.tokenize(node) for node in new_dict['nodes']]
        new_dict['split_nodes'] = [nd for nodes in word_nodes for nd in nodes]

        start = 0
        split2start = {}
        for idx in range(len(word_nodes)):
            split2start[idx] = start
            start += len(word_nodes[idx])

        split_edges = [[], []]
        split_types = []
        pairs = []
        relations = []
        for tri in new_dict['triples']:
            h, r, t = bert_tokenizer.tokenize(tri[0]), tri[1], bert_tokenizer.tokenize(tri[2])
            hidx = word_nodes.index(h)
            tidx = word_nodes.index(t)
            pairs.append([[split2start[hidx], split2start[hidx] + len(h) - 1],
                          [split2start[tidx], split2start[tidx] + len(t) - 1]])
            relations.append(r)
            for i, hn in enumerate(word_nodes[hidx]):
                for j, tn in enumerate(word_nodes[tidx]):
                    split_edges[0].append(split2start[hidx] + i)
                    split_edges[1].append(split2start[tidx] + j)
                    split_types.append(r)
                    split_edges[1].append(split2start[hidx] + i)
                    split_edges[0].append(split2start[tidx] + j)
                    split_types.append(r)
        new_dict['split_edges'] = split_edges
        new_dict['split_types'] = split_types
        new_dict['pairs'] = pairs
        new_dict['relations'] = relations

        assert len(new_dict['pairs']) == len(new_dict['relations']), "the length of pairs and relations are not equal"

        target_tokens = new_dict['target'].split()

        order2ent = {}
        used_ner = set()
        new_target_tokens = []
        order = 1
        for idx, token in enumerate(target_tokens):
            if token.isupper():
                if token not in used_ner:
                    new_target_tokens.append('<mask>')
                    ent = new_dict['ner2ent'][token]
                    used_ner.add(token)
                    order2ent[order] = ent
                    order += 1
                else:
                    ent = new_dict['ner2ent'][token]
                    new_target_tokens.append(ent)
            else:
                new_target_tokens.append(token)

        target_tokens = ["<s>"] + bert_tokenizer.tokenize(' '.join(new_target_tokens)) + ["</s>"]

        positions = [[0] * len(bert_tokenizer.tokenize(ent)) for ent in new_dict['nodes']]
        masked_target_tokens = []
        new_target_tokens = []
        order = 1
        for idx, token in enumerate(target_tokens):
            if token == '<mask>':
                ent = order2ent[order]
                ent_len = len(bert_tokenizer.tokenize(ent))
                start = len(masked_target_tokens)
                ent_idx = new_dict['nodes'].index(ent)
                positions[ent_idx] = list(range(start, start + ent_len))
                masked_target_tokens.extend(['<mask>'] * ent_len)
                new_target_tokens.extend(bert_tokenizer.tokenize(ent))
                order += 1
            else:
                masked_target_tokens.append(token)
                new_target_tokens.append(token)
        positions = [p for pos in positions for p in pos]

        new_dict['positions'] = positions
        new_dict['description'] = new_target_tokens
        new_dict['masked_description'] = masked_target_tokens

        assert len(new_dict['split_nodes']) == len(new_dict['positions'])

        fout.write(json.dumps(new_dict, ensure_ascii=False) + "\n")
    fout.close()
