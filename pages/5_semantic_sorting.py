import streamlit as st

from sklearn.metrics import pairwise_distances

from sko.GA import GA_TSP

def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

num_points = len(st.session_state.sentences.sentences)

with st.spinner(text='Wir suchen den kleinsten Weg zwischen den Notizen 🔎'):
    embeddings = st.session_state.sentences.get_embeddings()
    distance_matrix = pairwise_distances(embeddings, embeddings, metric='cosine')

    print(distance_matrix)

    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
    best_points, best_distance = ga_tsp.run()
    print(best_points)
    print("-"*20)
    print(best_distance)

st.markdown("## original Ordering")
for sentence in st.session_state.sentences.sentences:
    st.write(sentence.text)

st.markdown("## semantic Ordering")
for point in best_points:
    st.write(st.session_state.sentences.sentences[point].text)
