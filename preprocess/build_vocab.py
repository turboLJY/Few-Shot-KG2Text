# import json
#
# train_rel = set()
# fin = open("train_processed.json", "r")
# for line in fin:
#     data = json.loads(line.strip())
#     for t in data["triples"]:
#         train_rel.add(t[1])
#         # train_rel.add(t[0])
#         # train_rel.add(t[2])
# fin.close()
# print(len(train_rel))
#
# unseen_valid = set()
# fin = open("valid_processed.json", "r")
# for line in fin:
#     data = json.loads(line.strip())
#     for t in data["triples"]:
#         # if t[0] not in train_rel:
#         #     unseen_valid.add(t[0])
#         # if t[2] not in train_rel:
#         #     unseen_valid.add(t[2])
#         if t[1] not in train_rel:
#             unseen_valid.add(t[1])
# fin.close()
# print(unseen_valid)
#
# unseen_test = set()
# fin = open("test_processed.json", "r")
# for line in fin:
#     data = json.loads(line.strip())
#     for t in data["triples"]:
#         # if t[0] not in train_rel:
#         #     unseen_test.add(t[0])
#         # if t[2] not in train_rel:
#         #     unseen_valid.add(t[2])
#         if t[1] not in train_rel:
#             unseen_valid.add(t[1])
# fin.close()
# print(unseen_test)
#
import pickle
import json

entities = set()
relations = set()

filename = ['train_processed.json', 'valid_processed.json', 'test_processed.json']

for fn in filename:
    fin = open(fn, "r")
    data = []
    for line in fin:
        data.append(json.loads(line.strip()))
    fin.close()

    for d in data:
        for r in d["relations"]:
            relations.add(r)

        for n in d["split_nodes"]:
            entities.add(n)

print(len(entities))
print(len(relations))

entities = ["<pad>"] + list(entities)

en2idx = dict()
for idx, en in enumerate(entities):
    en2idx[en] = idx

relations = ["<pad>"] + list(relations)

rel2idx = dict()
for idx, rel in enumerate(relations):
    rel2idx[rel] = idx

pickle.dump(en2idx, open("node.pkl", "wb"))
pickle.dump(rel2idx, open("relation.pkl", "wb"))

