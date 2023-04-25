from typing import Iterable, List

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMBEDDING_MODEL = 'sentence-transformers/distiluse-base-multilingual-cased-v1'


class EmbeddingCreator:
    """
    Simple Class to embed stuff
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Calculate
        self.pipe = pipeline(model="Davlan/bert-base-multilingual-cased-ner-hrl")

    def encode(self, sentences: List[str]):
        """
        Method to encode sentences
        :param sentences:
        :return:
        """
        return self.model.encode(sentences)
