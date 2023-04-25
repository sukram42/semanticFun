from typing import List, Iterable

from pydantic import BaseModel
import numpy as np

class Sentence(BaseModel):
    embedding: np.ndarray
    text: str

    class Config:
        arbitrary_types_allowed = True


class SentenceStore(BaseModel):
    sentences: List[Sentence] = []

    def append(self, sentence: Sentence):
        self.sentences.append(sentence)

    def extend(self, sentence: Iterable[Sentence]):
        self.sentences.extend(sentence)

    def to_texts(self):
        return map(lambda sent: sent.text, self.sentences)

    def get_embeddings(self):
        return np.vstack([*map(lambda sent: sent.embedding, self.sentences)])

