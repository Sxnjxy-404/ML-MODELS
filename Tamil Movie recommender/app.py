import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Data --------------------
data = {
    'Title': [
        'Vikram', 'Master', 'Kaithi', 'Jailer', 'Beast',
        'Leo', 'Asuran', 'Viduthalai Part 1', 'Thunivu', 'Maamannan'
    ],
    'Year': [2022, 2021, 2019, 2023, 2022, 2023, 2019, 2023, 2023, 2023],
    'Director': [
        'Lokesh Kanagaraj', 'Lokesh Kanagaraj', 'Lokesh Kanagaraj', 'Nelson Dilipkumar', 'Nelson Dilipkumar',
        'Lokesh Kanagaraj', 'Vetrimaaran', 'Vetrimaaran', 'H. Vinoth', 'Mari Selvaraj'
    ],
    'Writer': [
        'Lokesh Kanagaraj', 'Lokesh Kanagaraj', 'Lokesh Kanagaraj', 'Nelson Dilipkumar', 'Nelson Dilipkumar',
        'Lokesh Kanagaraj', 'Vetrimaaran', 'Vetrimaaran', 'H. Vinoth', 'Mari Selvaraj'
    ],
    'Stars': [
        'Kamal Haasan, Vijay Sethupathi, Fahadh Faasil',
        'Thalapathy Vijay, Vijay Sethupathi, Malavika Mohanan',
        'Karthi, Narain, Dheena',
        'Rajinikanth, Vinayakan, Ramya Krishnan',
        'Vijay, Pooja Hegde, Selvaraghavan',
        'Vijay, Trisha, Arjun Sarja',
        'Dhanush, Manju Warrier, Pasupathy',
        'Soori, Vijay Sethupathi, Gautham Vasudev Menon',
        'Ajith Kumar, Manju Warrier, Samuthirakani',
        'Udhayanidhi Stalin, Vadivelu, Fahadh Faasil'
    ],
    'Rating': [8.3, 7.9, 8.5, 7.5, 6.2, 7.8, 8.6, 8.0, 6.5, 7.2],
    'Votes': [95000, 88000, 72000, 67000, 49000, 81000, 89000, 50000, 46000, 40000]
}

df = pd.DataFrame(data)

# -------------------- Content-Based Filtering --------------------
df['Features'] = df['Director'] + ' ' + df['Writer'] + ' ' + df['Stars']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Helper to get recommendations
def content_recommend(title):
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['Title', 'Rating']]

# -------------------- Collaborative Filtering (Rating-Based Approximation) --------------------
def collaborative_recommend(selected_title):
    high_rated = df[df['Rating'] >= 7.5]
    selected_votes = df[df['Title'] == selected_title]['Votes'].values[0]

    similar_vote_range = df[
        (df['Votes'] >= selected_votes - 10000) &
        (df['Votes'] <= selected_votes + 10000)
    ]

    merged = pd.merge(high_rated, similar_vote_range, how='inner')
    return merged[['Title', 'Rating']].drop_duplicates().head(5)


# -------------------- Hybrid Recommender --------------------
def hybrid_recommend(title):
    content_df = content_recommend(title)
    collab_df = collaborative_recommend(title)
    hybrid_df = pd.concat([content_df, collab_df]).drop_duplicates().reset_index(drop=True)
    return hybrid_df.head(5)

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Tamil Movie Recommender", layout="centered")
st.title("🎬 Tamil Movie Recommendation System")

movie_choice = st.selectbox("Choose a Movie You Like", df['Title'].tolist())

method = st.radio("Select Recommendation Type:", ['Content-Based', 'Collaborative-Based', 'Hybrid'])

if st.button("Get Recommendations"):
    if method == 'Content-Based':
        st.subheader("🎯 Content-Based Recommendations")
        recs = content_recommend(movie_choice)
    elif method == 'Collaborative-Based':
        st.subheader("👥 Collaborative Recommendations")
        recs = collaborative_recommend(movie_choice)
    else:
        st.subheader("🔀 Hybrid Recommendations")
        recs = hybrid_recommend(movie_choice)

    if recs.empty:
        st.info("No recommendations found.")
    else:
        st.table(recs)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("✅ Made with Streamlit | Dataset: Tamil Movie Ratings")