import argparse
import os
import time

import spacy
import torch
from nltk.tokenize import sent_tokenize, word_tokenize

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def _sentence_indexes(match, texts):
    sentence_indexes = []
    entities_location_indexes = []
    ltok = len(match)
    for i, text in enumerate(texts):
        for j in range(len(text)):
            if match == text[j : (j + ltok)]:
                sentence_indexes.append(i)
                entities_location_indexes.append(j)
                break
    return (sentence_indexes, entities_location_indexes)


def update_entities(entities, sentences):

    for i in range(len(entities["entities"])):
        match = entities["entities"][i]["entity_text"]
        (sentence_indexes, entities_location_indexes) = _sentence_indexes(
            match, sentences
        )
        entities["entities"][i]["sentence_indexes"] = sentence_indexes
        entities["entities"][i]["entities_location_indexes"] = entities_location_indexes

    return entities


def generate_sample(entities, sentences):
    entities = update_entities(entities, sentences)

    sample = dict()

    for entity in entities["entities"]:
        mention = entity["entity_text"]
        sentence_indexes = entity["sentence_indexes"]
        entities_location_indexes = entity["entities_location_indexes"]
        _id = 0

        for sentence_index, entities_location_index in zip(
            sentence_indexes, entities_location_indexes
        ):
            sample[str(_id)] = [
                sentences[sentence_index],
                [[entities_location_index, len(mention)]],
            ]
            _id += 1

    return sample
