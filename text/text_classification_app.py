import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import Dataset

# Set environment variable to use PyTorch backend
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import transformers after setting environment
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        pipeline
    )
except ImportError as e:
    st.error(f"Transformers import error: {e}")
    st.info("Please run: pip install tf-keras")
    st.stop()

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="AI Text Classifier Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Text Classifier Pro</h1>
    <p>Professional Text Classification with Hugging Face Transformers</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model Selection
    model_options = {
        "DistilBERT (Recommended)": "distilbert-base-uncased",
        "BERT Base": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "ALBERT": "albert-base-v2"
    }
    
    selected_model = st.selectbox(
        "Choose Base Model",
        list(model_options.keys()),
        help="DistilBERT is faster and lighter while maintaining good performance"
    )
    model_name = model_options[selected_model]
    
    # Training Parameters
    st.markdown("### 🎛️ Training Parameters")
    
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-5, 2e-5, 3e-5, 5e-5],
        value=2e-5,
        format_func=lambda x: f"{x:.0e}"
    )
    
    num_epochs = st.slider("Number of Epochs", 1, 20, 3)
    batch_size = st.selectbox("Batch Size", [8, 16, 32], index=0)
    
    # Advanced Settings
    with st.expander("Advanced Settings"):
        weight_decay = st.slider("Weight Decay", 0.0, 0.1, 0.01, 0.001)
        warmup_steps = st.number_input("Warmup Steps", 0, 1000, 100)
        max_length = st.slider("Max Sequence Length", 64, 512, 128)

# Main Content Area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Upload", "🚀 Training", "🔮 Predictions", "📈 Analytics"])

with tab1:
    st.markdown("### 📁 Dataset Upload & Preprocessing")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)",
        type=["csv"],
        help="Your CSV should contain 'text' and 'label' columns"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate dataset
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.markdown(f"""
                <div class="status-error">
                    <strong>❌ Error:</strong> Missing required columns: {', '.join(missing_columns)}
                    <br>Please ensure your CSV has 'text' and 'label' columns.
                </div>
                """, unsafe_allow_html=True)
            else:
                # Dataset Statistics
                st.markdown("### 📊 Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#667eea;">Total Samples</h3>
                        <h2 style="margin:0; color:#667eea;">{len(df):,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#667eea;">Unique Labels</h3>
                        <h2 style="margin:0; color:#667eea;">{df['label'].nunique()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_length = df['text'].str.len().mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#667eea;">Avg Text Length</h3>
                        <h2 style="margin:0; color:#667eea;">{avg_length:.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    missing_data = df.isnull().sum().sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#667eea;">Missing Values</h3>
                        <h2 style="margin:0; color:#667eea;">{missing_data}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data Preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Label Distribution
                st.markdown("### 🏷️ Label Distribution")
                label_counts = df['label'].value_counts()
                
                fig = px.bar(
                    x=label_counts.index,
                    y=label_counts.values,
                    labels={'x': 'Labels', 'y': 'Count'},
                    title="Distribution of Labels in Dataset",
                    color=label_counts.values,
                    color_continuous_scale="viridis"
                )
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Text Length Distribution
                st.markdown("### 📏 Text Length Analysis")
                text_lengths = df['text'].str.len()
                
                fig = px.histogram(
                    x=text_lengths,
                    nbins=50,
                    title="Distribution of Text Lengths",
                    labels={'x': 'Text Length (characters)', 'y': 'Frequency'}
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store dataset info in session state
                st.session_state.dataset_info = {
                    'total_samples': len(df),
                    'num_labels': df['label'].nunique(),
                    'label_names': df['label'].unique().tolist(),
                    'avg_text_length': avg_length,
                    'label_distribution': label_counts.to_dict()
                }
                
                st.session_state.df = df
                
                st.markdown("""
                <div class="status-success">
                    <strong>✅ Dataset loaded successfully!</strong> Ready for training.
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="status-error">
                <strong>❌ Error loading dataset:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🚀 Model Training")
    
    if 'df' not in st.session_state:
        st.markdown("""
        <div class="status-warning">
            <strong>⚠️ No dataset loaded.</strong> Please upload a dataset in the Data Upload tab first.
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df
        
        # Training Configuration Summary
        st.markdown("### ⚙️ Training Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.info(f"""
            **Model:** {selected_model}
            **Learning Rate:** {learning_rate}
            **Epochs:** {num_epochs}
            **Batch Size:** {batch_size}
            """)
        
        with config_col2:
            st.info(f"""
            **Weight Decay:** {weight_decay}
            **Warmup Steps:** {warmup_steps}
            **Max Length:** {max_length}
            **Dataset Size:** {len(df):,} samples
            """)
        
        # Train Button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            try:
                with st.spinner("🔄 Preparing dataset and model..."):
                    # Convert to HF dataset
                    dataset = Dataset.from_pandas(df)
                    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
                    
                    # Encode labels
                    label_names = list(df["label"].unique())
                    label2id = {l: i for i, l in enumerate(label_names)}
                    id2label = {i: l for i, l in enumerate(label_names)}
                    
                    def encode_labels(example):
                        example["label"] = label2id[example["label"]]
                        return example
                    
                    dataset = dataset.map(encode_labels)
                    
                    # Tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    def tokenize(batch):
                        return tokenizer(
                            batch["text"], 
                            padding="max_length", 
                            truncation=True,
                            max_length=max_length
                        )
                    
                    dataset = dataset.map(tokenize, batched=True)
                    dataset = dataset.remove_columns(["text"])
                    dataset.set_format("torch")
                    
                    # Model
                    num_labels = len(label_names)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id=label2id
                    )
                    
                    # Metrics
                    def compute_metrics(eval_pred):
                        logits, labels = eval_pred
                        preds = np.argmax(logits, axis=-1)
                        return {
                            "accuracy": accuracy_score(labels, preds),
                            "f1": f1_score(labels, preds, average="weighted"),
                        }
                    
                    # Training setup
                    training_args = TrainingArguments(
                        output_dir="./results",
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        logging_dir="./logs",
                        logging_steps=10,
                        learning_rate=learning_rate,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        num_train_epochs=num_epochs,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps,
                        load_best_model_at_end=True,
                        metric_for_best_model="f1",
                        greater_is_better=True,
                        report_to=None,
                    )
                
                # Training Progress
                progress_container = st.container()
                status_container = st.container()
                
                with status_container:
                    st.markdown("### 📊 Training Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                with st.spinner("🎯 Training model... This may take several minutes."):
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=dataset["train"],
                        eval_dataset=dataset["test"],
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                    )
                    
                    # Custom callback for progress tracking
                    class ProgressCallback:
                        def __init__(self, progress_bar, status_text, num_epochs):
                            self.progress_bar = progress_bar
                            self.status_text = status_text
                            self.num_epochs = num_epochs
                            
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / self.num_epochs
                            self.progress_bar.progress(progress)
                            self.status_text.text(f"Epoch {epoch + 1}/{self.num_epochs} completed")
                    
                    # Train the model
                    trainer.train()
                    
                    progress_bar.progress(1.0)
                    status_text.text("Training completed! 🎉")
                
                # Save trained model
                model.save_pretrained("./saved_model")
                tokenizer.save_pretrained("./saved_model")
                
                # Save training metadata
                training_metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'num_labels': num_labels,
                    'label_names': label_names,
                    'label2id': label2id,
                    'id2label': id2label,
                    'training_args': {
                        'learning_rate': learning_rate,
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'weight_decay': weight_decay
                    }
                }
                
                with open('./saved_model/training_metadata.json', 'w') as f:
                    json.dump(training_metadata, f)
                
                st.session_state.model_trained = True
                st.session_state.training_history = training_metadata
                
                st.markdown("""
                <div class="status-success">
                    <strong>🎉 Training completed successfully!</strong><br>
                    Your model has been saved and is ready for predictions.
                </div>
                """, unsafe_allow_html=True)
                
                # Display final metrics
                st.markdown("### 📊 Final Training Metrics")
                final_metrics = trainer.evaluate()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Final Accuracy", f"{final_metrics.get('eval_accuracy', 0):.4f}")
                with metric_col2:
                    st.metric("Final F1-Score", f"{final_metrics.get('eval_f1', 0):.4f}")
                
            except Exception as e:
                st.markdown(f"""
                <div class="status-error">
                    <strong>❌ Training failed:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🔮 Real-time Predictions")
    
    if not os.path.exists("./saved_model"):
        st.markdown("""
        <div class="status-warning">
            <strong>⚠️ No trained model found.</strong> Please train a model first in the Training tab.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load model info
        if os.path.exists('./saved_model/training_metadata.json'):
            with open('./saved_model/training_metadata.json', 'r') as f:
                metadata = json.load(f)
                
            st.markdown("### ℹ️ Model Information")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.info(f"""
                **Base Model:** {metadata.get('model_name', 'Unknown')}
                **Number of Labels:** {metadata.get('num_labels', 'Unknown')}
                **Training Date:** {metadata.get('timestamp', 'Unknown')[:10]}
                """)
            
            with info_col2:
                labels = metadata.get('label_names', [])
                st.info(f"""
                **Available Labels:**
                {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}
                """)
        
        # Prediction Interface
        st.markdown("### 💬 Try Your Model")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Single Text", "Batch Texts", "Example Texts"],
            horizontal=True
        )
        
        if input_method == "Single Text":
            user_input = st.text_area(
                "Enter text for classification:",
                placeholder="Type your text here...",
                height=100
            )
            
            if user_input.strip():
                with st.spinner("🧠 Analyzing..."):
                    try:
                        clf = pipeline(
                            "text-classification",
                            model="./saved_model",
                            tokenizer="./saved_model"
                        )
                        
                        prediction = clf(user_input)[0]
                        confidence = prediction['score']
                        predicted_label = prediction['label']
                        
                        # Display prediction with confidence
                        if confidence > 0.8:
                            confidence_color = "#28a745"  # Green
                            confidence_text = "High"
                        elif confidence > 0.6:
                            confidence_color = "#ffc107"  # Yellow
                            confidence_text = "Medium"
                        else:
                            confidence_color = "#dc3545"  # Red
                            confidence_text = "Low"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🎯 Prediction Result</h2>
                            <h1 style="margin: 1rem 0;">{predicted_label}</h1>
                            <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 1rem;">
                                <div>
                                    <h4 style="margin: 0;">Confidence</h4>
                                    <h2 style="margin: 0; color: {confidence_color};">{confidence:.2%}</h2>
                                </div>
                                <div>
                                    <h4 style="margin: 0;">Reliability</h4>
                                    <h2 style="margin: 0; color: {confidence_color};">{confidence_text}</h2>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
        
        elif input_method == "Batch Texts":
            st.markdown("**Enter multiple texts (one per line):**")
            batch_input = st.text_area(
                "Batch texts:",
                placeholder="Text 1\nText 2\nText 3\n...",
                height=150
            )
            
            if batch_input.strip():
                texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
                
                if st.button("🚀 Classify All", type="primary"):
                    with st.spinner(f"🧠 Analyzing {len(texts)} texts..."):
                        try:
                            clf = pipeline(
                                "text-classification",
                                model="./saved_model",
                                tokenizer="./saved_model"
                            )
                            
                            predictions = clf(texts)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame({
                                'Text': [text[:100] + '...' if len(text) > 100 else text for text in texts],
                                'Full_Text': texts,
                                'Predicted_Label': [pred['label'] for pred in predictions],
                                'Confidence': [pred['score'] for pred in predictions]
                            })
                            
                            st.markdown("### 📊 Batch Prediction Results")
                            st.dataframe(
                                results_df[['Text', 'Predicted_Label', 'Confidence']].round(4),
                                use_container_width=True
                            )
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "batch_predictions.csv",
                                "text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Batch prediction failed: {str(e)}")
        
        else:  # Example Texts
            st.markdown("**Try some example texts:**")
            
            # You can customize these examples based on your use case
            examples = [
                "This product is amazing! I love it so much.",
                "The service was terrible and slow.",
                "It's an okay product, nothing special.",
                "Outstanding quality and fast delivery!",
                "I'm not sure about this purchase."
            ]
            
            for i, example in enumerate(examples):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Example {i+1}:** {example}")
                with col2:
                    if st.button(f"Classify", key=f"example_{i}"):
                        with st.spinner("🧠 Analyzing..."):
                            try:
                                clf = pipeline(
                                    "text-classification",
                                    model="./saved_model",
                                    tokenizer="./saved_model"
                                )
                                
                                prediction = clf(example)[0]
                                st.success(f"**{prediction['label']}** ({prediction['score']:.2%})")
                                
                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")

with tab4:
    st.markdown("### 📈 Model Analytics & Insights")
    
    if not st.session_state.get('model_trained', False):
        st.markdown("""
        <div class="status-warning">
            <strong>⚠️ No training history available.</strong> Train a model first to see analytics.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Model Performance Overview
        if st.session_state.dataset_info:
            info = st.session_state.dataset_info
            
            st.markdown("### 📊 Dataset & Model Overview")
            
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            
            with overview_col1:
                st.metric("Total Samples", f"{info['total_samples']:,}")
            with overview_col2:
                st.metric("Number of Classes", info['num_labels'])
            with overview_col3:
                st.metric("Average Text Length", f"{info['avg_text_length']:.0f}")
            with overview_col4:
                training_time = st.session_state.training_history.get('timestamp', '')
                if training_time:
                    st.metric("Last Trained", training_time[:10])
        
        # Training Configuration
        if st.session_state.training_history:
            st.markdown("### ⚙️ Training Configuration")
            config = st.session_state.training_history.get('training_args', {})
            
            config_df = pd.DataFrame([
                {'Parameter': 'Learning Rate', 'Value': config.get('learning_rate', 'N/A')},
                {'Parameter': 'Number of Epochs', 'Value': config.get('num_epochs', 'N/A')},
                {'Parameter': 'Batch Size', 'Value': config.get('batch_size', 'N/A')},
                {'Parameter': 'Weight Decay', 'Value': config.get('weight_decay', 'N/A')},
            ])
            
            st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        # Model Comparison Section
        st.markdown("### 🔄 Model Performance Comparison")
        st.info("💡 **Tip:** Train multiple models with different configurations to compare their performance here.")
        
        # Placeholder for future enhancements
        st.markdown("""
        **Future Features:**
        - Training loss curves
        - Validation metrics over epochs  
        - Confusion matrix visualization
        - Class-wise performance metrics
        - Model size and inference speed comparison
        - Error analysis and misclassified examples
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 <strong>AI Text Classifier Pro</strong> | Built with Streamlit & Hugging Face Transformers</p>
    <p style="font-size: 0.8em;">For support and documentation, visit the project repository</p>
</div>
""", unsafe_allow_html=True)