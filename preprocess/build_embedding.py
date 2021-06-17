import torch
import pickle
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('../../pretrained_model/bart-large')
tokenizer = BartTokenizer.from_pretrained('../../pretrained_model/bart-large')
embedding = model.get_input_embeddings().weight

vocab = pickle.load(open("node.pkl", "rb"))

my_embedding = []
my_iddx = set()
for token, idx in vocab.items():
    iddx = tokenizer.convert_tokens_to_ids([token])[0]
    my_iddx.add(iddx)
    my_embedding.append(embedding[iddx])

my_embedding = torch.stack(my_embedding, dim=0).detach().numpy()
np.save("node_embeddings.npy", my_embedding)



