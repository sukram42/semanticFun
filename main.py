import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import pandas as pd
import plotly.express as px
from transformers import pipeline

from embeddings import EmbeddingCreator
from ner import NERDetector

embedder = EmbeddingCreator()
nerCreator = NERDetector()


def submit_form():
    st.session_state.sentences.extend([st.session_state.increment_sentences])
    st.session_state.increment_sentences = ""


with st.form("my_form"):
   st.write("This is the note input. Here you can input your notes. These notes will afterwards be combined by "
            "semantic analysis.")
   st.text_area("Add a new Note", key="increment_sentences")



   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit", on_click=submit_form)




# Calculate
def reset_sentences():
    st.session_state.sentences = []

# Init State
if 'sentences' not in st.session_state:
    reset_sentences()
if 'perplexity' not in st.session_state:
    st.session_state.perplexity = 3

n_sentences = len(st.session_state.sentences)
#
st.write(st.session_state.sentences)
# # Calculate Embeddings
embeddings = embedder.encode(st.session_state.sentences)

# # Calculate NER
# ner_results = nerCreator.detect_ner(st.session_state.sentences)
#
# for se in ner_results:
#     print("---------------------")
#     for ent in se:
#         print(f"+++{ent['word']:<15} {ent['entity']:<15} | {ent}")
#
#
st.slider("Perplexity of the plot", key="perplexity", value=3)
#
#
try:
    distance_matrix = pairwise_distances(embeddings, embeddings, metric='cosine')
    tsne_model = TSNE(metric="precomputed", perplexity=st.session_state.perplexity, init="random", n_components=2)
    Xpr = tsne_model.fit_transform(distance_matrix)

    distances = pd.DataFrame(distance_matrix)
    st.dataframe(distances)

    df = pd.DataFrame(Xpr)
    df['text'] = st.session_state.sentences

    fig = px.scatter(df, x=0, y=1, hover_data=['text'])

    st.plotly_chart(fig)

except ValueError as e:
    st.write("We need atleast 3 sentences")
    print(e)



# st.button("RESET", on_click=reset_sentences)
print(submitted)
