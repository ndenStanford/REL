import json
import os
import sqlite3

import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def binary_to_dict(the_binary):
    jsn = "".join(chr(int(x, 2)) for x in the_binary.split())
    d = json.loads(jsn)
    return d


fname = "/home/ubuntu/REL/data/wiki_2019/generated/entity_word_embedding.db"
table_name = "wiki"
column = "p_e_m"
candidate_set = set()

read_dictionary = np.load("../BLINK/title2id.npy", allow_pickle="TRUE").item()

db = sqlite3.connect(fname, isolation_level=None)

c = db.cursor()
# q = c.execute('select emb from embeddings where word = :word', {'word': w}).fetchone()
# return array('f', q[0]).tolist() if q else None
e = c.execute("select * from wiki")  # limit 10000"

for row in e:
    the_binary = row[1]
    candidates = binary_to_dict(the_binary)
    for candidate, score in candidates:
        candidate_cleaned = candidate.replace("_", " ")
        candidate_set.add(candidate_cleaned)

print(len(candidate_set))
valid_candidate_count = 0
invalid_candidate_count = 0

for c in candidate_set:
    if c in read_dictionary.keys():
        valid_candidate_count += 1
    else:
        invalid_candidate_count += 1

print("valid percent = ", str(valid_candidate_count / len(candidate_set) * 100))
print("invalid percent = ", str(invalid_candidate_count / len(candidate_set) * 100))
