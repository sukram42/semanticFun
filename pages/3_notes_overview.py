import streamlit as st

from home import reset_sentences

for sentence in st.session_state.sentences.sentences:
    st.text(sentence.text)

st.button("Reset", on_click=reset_sentences)