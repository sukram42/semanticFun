import streamlit as st
import plotly.express as px
import pandas as pd

import umap.umap_ as umap
with st.spinner(text='Die ganzen Dimensionen werden etwas kleiner geschnitten ✂'):
    reducer = umap.UMAP(random_state=42)

    transformed = reducer.fit_transform(st.session_state.sentences.get_embeddings())

    df = pd.DataFrame(transformed)

    df['text'] = [*st.session_state.sentences.to_texts()]
    fig = px.scatter(df, x=0, y=1, hover_data=['text'])

    st.success("VOILÁ!")
st.plotly_chart(fig)
