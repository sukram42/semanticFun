import streamlit as st

from models import Sentence

for sentence in st.session_state.sentences.sentences:
    st.text(sentence.text)


def submit_form():
    st.session_state.sentences.append(
        Sentence(text=st.session_state.increment_sentences,
                 embedding=st.session_state.sentenceCreator.encode([st.session_state.increment_sentences])))

    st.session_state.increment_sentences = ""


with st.form("my_form"):
    st.write("This is the note input. Here you can input your notes. These notes will afterwards be combined by "
             "semantic analysis.")
    st.text_area("Add a new Note", key="increment_sentences")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit", on_click=submit_form)
