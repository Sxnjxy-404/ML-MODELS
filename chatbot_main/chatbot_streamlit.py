#======================================================
# chatbot_streamlit.py — Streamlit chat UI
# Run: streamlit run chatbot_streamlit.py
#======================================================

import streamlit as st
import os
from inference_utils import load_artifacts, greedy_decode, beam_search_decode

MODEL_DIR = 'artifacts'

@st.cache_resource
def _load():
    """Load model artifacts with caching"""
    try:
        model, tok, meta = load_artifacts(MODEL_DIR)
        return model, tok, meta, None
    except Exception as e:
        return None, None, None, str(e)

def main():
    st.set_page_config(
        page_title="GRU Chatbot", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 GRU Chatbot")
    st.caption("Seq2Seq with GRU — trained on your CSV (prompt,reply)")
    
    # Load model
    model, tok, meta, error = _load()
    
    if error:
        st.error(f"Error loading model: {error}")
        st.info("Make sure you have:")
        st.write("1. Trained the model by running: `python train_gru_chatbot.py`")
        st.write("2. The 'artifacts' directory exists with model files")
        return
    
    if model is None:
        st.error("Model not loaded properly")
        return
    
    # Sidebar with settings
    with st.sidebar:
        st.header("Settings")
        
        decode_method = st.selectbox(
            "Decoding Method",
            ["Greedy", "Beam Search"],
            help="Greedy is faster, Beam Search may give better results"
        )
        
        if decode_method == "Beam Search":
            beam_width = st.slider("Beam Width", 1, 5, 3)
        
        max_tokens = st.slider("Max Response Length", 10, 50, 30)
        
        st.divider()
        
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        
        st.subheader("Model Info")
        if meta:
            st.write(f"**Vocab Size:** {meta.get('vocab_size', 'N/A')}")
            st.write(f"**Max Length:** {meta.get('max_len', 'N/A')}")
            st.write(f"**Embedding Dim:** {meta.get('emb_dim', 'N/A')}")
            st.write(f"**Hidden Dim:** {meta.get('hid_dim', 'N/A')}")
    
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Chat input
    user_input = st.chat_input("Type your message…")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for role, text in st.session_state.history:
            with st.chat_message(role):
                st.markdown(text)
        
        # Process new user input
        if user_input:
            # Add user message to history
            st.session_state.history.append(("user", user_input))
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        if decode_method == "Greedy":
                            reply = greedy_decode(model, tok, meta, user_input, max_tokens)
                        else:
                            reply = beam_search_decode(model, tok, meta, user_input, beam_width, max_tokens)
                        
                        st.markdown(reply)
                        
                        # Add bot response to history
                        st.session_state.history.append(("assistant", reply))
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.history.append(("assistant", error_msg))

if __name__ == '__main__':
    main()