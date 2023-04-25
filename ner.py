from typing import List
from transformers import pipeline


class NERDetector:
    def __init__(self):
        self.pipeline = pipeline(model="Davlan/bert-base-multilingual-cased-ner-hrl")

    def detect_ner(self, sentences: List[str]):
        self.pipeline(sentences)
