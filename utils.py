import json
import torch
import random
import logging
import os
import datetime
import yaml
import re
import numpy as np
from torch import optim
from optim import CosineSchedule, TransformerSchedule


def build_optimizer(parameters, learner, learning_rate, config):
    if learner.lower() == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif learner.lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif learner.lower() == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=learning_rate)
    elif learner.lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=learning_rate)
    elif learner.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=learning_rate)
    elif learner.lower() == 'cosine_warmup':
        optimizer = CosineSchedule(
            optim.AdamW(parameters, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01),
            learning_rate, config["warmup_steps"], config["training_steps"]
        )
    elif learner.lower() == 'transformer_warmup':
        optimizer = TransformerSchedule(
            optim.AdamW(parameters, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01),
            learning_rate, config["embedding_size"], config["warmup_steps"]
        )
    else:
        raise ValueError('Received unrecognized optimizer {}.'.format(learner))
    return optimizer


def init_seed(seed, reproducibility):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def init_device(config):
    use_gpu = config["use_gpu"]
    device = torch.device("cuda:" + str(config["gpu_id"]) if torch.cuda.is_available() and use_gpu else "cpu")
    return device


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def init_logger(config):
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    logfilename = '{}-{}-{}.log'.format(config["dataset"], config["num_samples"], get_local_time())
    logfilepath = os.path.join(config["log_dir"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config["state"] is None or config["state"].lower() == 'info':
        level = logging.INFO
    elif config["state"].lower() == 'debug':
        level = logging.DEBUG
    elif config["state"].lower() == 'error':
        level = logging.ERROR
    elif config["state"].lower() == 'warning':
        level = logging.WARNING
    elif config["state"].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[fh, sh]
    )


def read_configuration(config_file):
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        config_dict = yaml.load(f.read(), Loader=yaml_loader)

    return config_dict


def collate_fn_graph_text(batch):
    nodes, edges, types, outputs, pointer, pairs, relations, positions, descriptions = [], [], [], [], [], [], [], [], []
    for b in batch:
        nodes.append(b[0])
        edges.append(b[1])
        types.append(b[2])
        outputs.append(b[3])
        pointer.append(b[4])
        pairs.append(b[5])
        relations.append(b[6])
        positions.append(b[7])
        descriptions.append(b[8])

    nodes, node_masks = padding(nodes, pad_idx=0)
    outputs, output_masks = padding(outputs, pad_idx=1)  # tokenizer.pad_token_id
    pointer, pointer_masks = padding(pointer, pad_idx=0)  # tokenizer.pad_token_id
    pairs, pair_masks = padding(pairs, pad_idx=[[0, 0], [0, 0]])
    relations, _ = padding(relations, pad_idx=0)
    positions, _ = padding(positions, pad_idx=0)
    descriptions, description_masks = padding(descriptions, pad_idx=1)  # tokenizer.pad_token_id

    return nodes, edges, types, node_masks, descriptions, description_masks, positions, relations, pairs, pair_masks, \
           outputs, output_masks, pointer, pointer_masks


def padding(inputs, pad_idx):
    lengths = [len(inp) for inp in inputs]
    max_len = max(lengths)
    padded_inputs = torch.as_tensor([inp + [pad_idx] * (max_len - len(inp)) for inp in inputs], dtype=torch.long)
    masks = torch.as_tensor([[1.] * len(inp) + [0.] * (max_len - len(inp)) for inp in inputs], dtype=torch.bool)
    return padded_inputs, masks


def edge_padding(edges, types, pad_idx):
    new_edges = []
    for edg in edges:
        heads = [edg[0][i] for i in range(0, len(edg[0]), 2)]
        tails = [edg[1][i] for i in range(0, len(edg[1]), 2)]
        new_edges.append([heads, tails])

    new_types = []
    for typ in types:
        new_types.append([typ[i] for i in range(0, len(typ), 2)])

    lengths = [len(typ) for typ in new_types]
    max_len = max(lengths)
    padded_edges = torch.as_tensor([[edg[0] + [pad_idx] * (max_len - len(edg[0])),
                                     edg[1] + [pad_idx] * (max_len - len(edg[1]))] for edg in new_edges],
                                   dtype=torch.long)
    padded_types = torch.as_tensor([typ + [pad_idx] * (max_len - len(typ)) for typ in new_types], dtype=torch.long)
    masks = torch.as_tensor([[1.] * len(typ) + [0.] * (max_len - len(typ)) for typ in new_types], dtype=torch.bool)
    return padded_edges, padded_types, masks
