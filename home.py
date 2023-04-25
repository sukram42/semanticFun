import streamlit as st

from embeddings import EmbeddingCreator
from models import SentenceStore

st.write("")


# Initialize the Streamlit state
def reset_sentences():
    st.session_state.sentences = SentenceStore()


with st.spinner(text='Modelle werden geladen und der Bumps hier vorbereitet 🐸'):
    if 'sentenceCreator' not in st.session_state:
        st.session_state.sentenceCreator = EmbeddingCreator()

    if 'sentences' not in st.session_state:
        reset_sentences()

    if 'increment_sentences' not in st.session_state:
        st.session_state.increment_sentences = ""

    st.success("Alles vorbereitet 🥳")
