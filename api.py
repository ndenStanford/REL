import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner
from REL.server import make_handler

wiki_version = "wiki_2014"

config = {
    "mode": "eval",
    "model_path": "/home/ubuntu/REL/data/ed-wiki-2014/model",  # or alias, see also tutorial 7: custom models
}

base_url = "/home/ubuntu/REL/data"

model = EntityDisambiguation(base_url, wiki_version, config)

# Using Flair:
tagger_ner = load_flair_ner("ner-fast")

# Alternatively, using n-grams:
tagger_ngram = Cmns(base_url, wiki_version, n=5)

handler = make_handler(
        base_url, wiki_version, model, tagger_ner
    )

text_doc = "If you're going to try, go all the way - Charles Bukowski"
post_data = {
    "text": text_doc,
    "spans": [],  # in case of ED only, this can also be left out when using the API
}

text, spans = handler.read_json(post_data)
response = handler.generate_response(text, spans)
response

'''
server_address = ("127.0.0.1", 1235)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_version, model, tagger_ner
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
'''