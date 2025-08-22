import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import warnings

warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_translation_model(target_lang):
    """Load MarianMT model and tokenizer for the given target language"""
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate_text(text, tokenizer, model):
    """Translate English text into target language"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def main():
    """Main Streamlit application"""

    st.title("🌍 AI Language Translator")
    st.markdown("### Powered by Hugging Face MarianMT models")
    st.markdown("---")

    # Sidebar language list
    languages = {
        'fr': '🇫🇷 French',
        'es': '🇪🇸 Spanish',
        'de': '🇩🇪 German',
        'it': '🇮🇹 Italian',
        'pt': '🇵🇹 Portuguese',
        'ru': '🇷🇺 Russian',
        'zh': '🇨🇳 Chinese',
        'ja': '🇯🇵 Japanese',
        'ko': '🇰🇷 Korean',
        'nl': '🇳🇱 Dutch'
    }

    selected_lang = st.sidebar.selectbox(
        "Choose target language:",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0
    )

    # Input
    col1, col2 = st.columns([1, 1])
    with col1:
        input_text = st.text_area(
            "🇬🇧 English Input",
            height=200,
            placeholder="Type some English text..."
        )

    with col2:
        st.markdown(f"### {languages[selected_lang]} Translation")

        if input_text.strip():
            try:
                tokenizer, model = load_translation_model(selected_lang)
                translation = translate_text(input_text, tokenizer, model)

                st.text_area("Translation:", value=translation, height=200, disabled=True)

                st.markdown("---")
                col_a, col_b = st.columns(2)
                col_a.metric("Input Length", len(input_text))
                col_b.metric("Output Length", len(translation))

            except Exception as e:
                st.error(f"⚠️ Failed to load model: {e}")

        else:
            st.info("✍️ Enter some English text to see the translation.")


# ✅ Run app
if __name__ == "__main__":
    main()
