import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Book Recommender", layout="centered", page_icon="📚")

# --- Custom CSS for Dark Theme ---
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stSelectbox > div {
        color: black !important;
    }
    .stButton > button {
        background-color: #00ADB5;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #393E46;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3771/3771491.png", width=100)
    st.title("📚 Book Recommender")
    st.markdown("**Built with using Streamlit**")
    st.markdown("---")

# --- Animated Title ---
st.markdown("<h1 style='text-align: center;'>✨ Smart Book Recommender 📘</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Get personalized book suggestions based on what you read and love!</p>", unsafe_allow_html=True)
st.markdown("---")

# Load data
books = pd.read_csv(r"C:\Users\sanja\Downloads\book rating\books.csv")
ratings = pd.read_csv(r"C:\Users\sanja\Downloads\book rating\ratings.csv")
books.fillna('', inplace=True)

# Content-Based
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

# Collaborative Filtering
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Inputs
book_titles = books['Title'].tolist()
book_selected = st.selectbox("📖 Choose a Book", book_titles)
user_ids = ratings['User_ID'].unique().tolist()
user_selected = st.selectbox("👤 Choose a User ID", user_ids)
rec_type = st.radio("📌 Recommendation Type", ["🎯 Content-Based", "🧠 Collaborative-Based", "⚡ Hybrid-Based"])

book_index = books[books['Title'] == book_selected].index[0]
book_id = books.loc[book_index, 'Book_ID']

# Recommendation button
if st.button("🔍 Recommend Now"):
    st.success(f"Recommendations based on: **{book_selected}**")

    # --- Content-Based ---
    if rec_type == "🎯 Content-Based":
        similar_books = content_similarity[book_index].argsort()[::-1][1:4]
        st.markdown("#### ✨ Top Content-Based Suggestions:")
        for i in similar_books:
            st.markdown(f"📘 **{books.iloc[i]['Title']}** by _{books.iloc[i]['Author']}_")

    # --- Collaborative-Based ---
    elif rec_type == "🧠 Collaborative-Based":
        if user_selected in user_sim_df.index:
            sim_users = user_sim_df[user_selected].sort_values(ascending=False)[1:]
            top_user = sim_users.index[0]
            top_user_ratings = user_item_matrix.loc[top_user]
            top_books = top_user_ratings.sort_values(ascending=False).index.tolist()
            recommended_ids = [b for b in top_books if b != book_id][:3]

            st.markdown("#### 🤝 Similar Users Recommend:")
            for bid in recommended_ids:
                title = books[books['Book_ID'] == bid]['Title'].values
                if len(title) > 0:
                    st.markdown(f"📗 **{title[0]}**")
                else:
                    st.write(f"Unknown Book ID: {bid}")
        else:
            st.error("⚠️ User not found in matrix.")

    # --- Hybrid-Based ---
    elif rec_type == "⚡ Hybrid-Based":
        content_scores = content_similarity[book_index]
        user_ratings = user_item_matrix.loc[user_selected]
        aligned_ratings = user_ratings.reindex(books['Book_ID']).fillna(0).values
        hybrid_score = 0.6 * content_scores + 0.4 * aligned_ratings
        top_indices = np.argsort(hybrid_score)[::-1]
        recommended_indices = [i for i in top_indices if i != book_index][:3]

        st.markdown("#### 🔄 Hybrid Suggestions:")
        for i in recommended_indices:
            st.markdown(f"📙 **{books.iloc[i]['Title']}** by _{books.iloc[i]['Author']}_")


# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Minimal interface. Maximum recommendations. 🖥️📖</p>", unsafe_allow_html=True)
