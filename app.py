import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# NLTK downloads
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data packages"""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    packages = {
        'punkt_tab': 'tokenizers/punkt_tab',
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    
    for package_name, package_path in packages.items():
        try:
            nltk.data.find(package_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except:
                pass

download_nltk_data()

# Load pre-trained models
@st.cache_resource
def load_pretrained_models():
    """Load all pre-trained models from pickle files"""
    models_dir = 'models'
    
    try:
        # Load vectorizer
        with open(f'{models_dir}/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load label encoder
        with open(f'{models_dir}/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load categories
        with open(f'{models_dir}/categories.pkl', 'rb') as f:
            categories = pickle.load(f)
        
        # Load all models
        model_files = {
            'Logistic Regression': 'LogisticRegression.pkl',
            'Random Forest': 'RandomForest.pkl',
            'SVM': 'SVM.pkl',
            'Naive Bayes': 'NaiveBayes.pkl',
            'XGBoost': 'XGBoost.pkl'
        }
        
        models = {}
        for name, filename in model_files.items():
            with open(f'{models_dir}/{filename}', 'rb') as f:
                models[name] = pickle.load(f)
        
        return vectorizer, label_encoder, categories, models, None
    
    except FileNotFoundError as e:
        return None, None, None, None, f"Model files not found: {e}"
    except Exception as e:
        return None, None, None, None, f"Error loading models: {e}"

# Text preprocessing
@st.cache_data
def clean_text(text):
    """Clean and preprocess resume text"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# PDF extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Main App
def main():
    # Load models on startup
    vectorizer, label_encoder, categories, models, error = load_pretrained_models()
    
    # Header
    st.markdown('<div class="main-header">🎯 AI-Powered Resume Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automatically categorize resumes using Machine Learning</div>', unsafe_allow_html=True)
    
    # Check if models loaded
    if error:
        st.error(f"❌ {error}")
        st.info("""
        **Missing model files!** Please ensure the following files are in the `models/` directory:
        - LogisticRegression.pkl
        - RandomForest.pkl
        - SVM.pkl
        - NaiveBayes.pkl
        - XGBoost.pkl
        - vectorizer.pkl
        - label_encoder.pkl
        - categories.pkl
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/resume.png", width=100)
        st.title("Navigation")
        
        page = st.radio("Select Page", [
            "🏠 Home",
            "🔍 Resume Classification",
            "📊 Model Information",
            "ℹ️ About"
        ])
        
        st.markdown("---")
        st.markdown("### 📌 Quick Info")
        st.success(f"""
        **Status:** ✅ Models Loaded
        
        **Categories:** {len(categories)}
        
        **Models:** {len(models)} classifiers
        """)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(categories)
    elif page == "🔍 Resume Classification":
        show_classification_page(vectorizer, label_encoder, models)
    elif page == "📊 Model Information":
        show_model_info_page(categories, models)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(categories):
    """Home page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to Resume Classifier! 👋
        
        This application uses **pre-trained Machine Learning models** to automatically 
        classify resumes into different job categories.
        
        ### 🎯 Key Features:
        
        - **5 Pre-trained Models**: Logistic Regression, Random Forest, SVM, Naive Bayes, XGBoost
        - **Fast Predictions**: No training required - instant results
        - **PDF Support**: Upload resumes in PDF format
        - **Consensus Prediction**: See agreement across all models
        - **High Accuracy**: Models trained on comprehensive dataset
        
        ### 🚀 How to Use:
        
        1. Navigate to **Resume Classification** page
        2. Upload a PDF resume
        3. Get instant predictions from all 5 models
        4. View consensus result and individual predictions
        
        ### 📋 Supported Categories:
        """)
        
        # Display categories in columns
        num_cols = 3
        cols = st.columns(num_cols)
        for idx, category in enumerate(sorted(categories)):
            with cols[idx % num_cols]:
                st.markdown(f"✅ {category}")

def show_classification_page(vectorizer, label_encoder, models):
    """Resume classification page"""
    
    st.header("🔍 Resume Classification")
    
    st.markdown("""
    <div class="info-box">
    <b>📄 Upload a Resume:</b><br>
    Upload a PDF resume to classify it into one of the job categories.
    The system will analyze the resume using all 5 pre-trained models.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a resume PDF file",
        type=['pdf'],
        help="Upload a PDF file containing the resume text"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Filename:** {uploaded_file.name}")
        with col2:
            st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.info(f"**Type:** {uploaded_file.type}")
        
        # Classify button
        if st.button("🎯 Classify Resume", use_container_width=True, type="primary"):
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_file)
            
            if resume_text:
                # Display extracted text
                with st.expander("📄 View Extracted Text"):
                    st.text_area("Resume Content", resume_text, height=200)
                
                # Preprocess
                with st.spinner("Preprocessing text..."):
                    cleaned_text = clean_text(resume_text)
                    text_vec = vectorizer.transform([cleaned_text])
                
                # Predictions
                st.subheader("🎯 Classification Results")
                
                predictions = {}
                for model_name, model in models.items():
                    pred_enc = model.predict(text_vec)[0]
                    pred_label = label_encoder.inverse_transform([pred_enc])[0]
                    
                    if hasattr(model, "predict_proba"):
                        prob = np.max(model.predict_proba(text_vec))
                    else:
                        prob = None
                    
                    predictions[model_name] = {
                        'category': pred_label,
                        'confidence': prob
                    }
                
                # Consensus prediction
                pred_categories = [pred['category'] for pred in predictions.values()]
                most_common = Counter(pred_categories).most_common(1)[0]
                consensus_category = most_common[0]
                consensus_count = most_common[1]
                
                # Display consensus
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">📊 Consensus Prediction</div>
                    <div style="font-size: 2.5rem; font-weight: bold;">{consensus_category}</div>
                    <div style="font-size: 1rem; margin-top: 0.5rem;">
                        {consensus_count} out of {len(predictions)} models agree
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual predictions
                st.subheader("🤖 Individual Model Predictions")
                
                cols = st.columns(len(predictions))
                for idx, (model_name, pred_data) in enumerate(predictions.items()):
                    with cols[idx]:
                        confidence_str = f"{pred_data['confidence']:.1%}" if pred_data['confidence'] else "N/A"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-weight: bold; color: #1f77b4;">{model_name}</div>
                            <div style="font-size: 1.2rem; margin: 0.5rem 0;">{pred_data['category']}</div>
                            <div style="color: #666;">Confidence: {confidence_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed table
                st.subheader("📋 Detailed Predictions")
                
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted Category': [pred['category'] for pred in predictions.values()],
                    'Confidence': [f"{pred['confidence']:.2%}" if pred['confidence'] else "N/A" 
                                  for pred in predictions.values()]
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=pred_df.to_csv(index=False),
                    file_name=f"resume_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_model_info_page(categories, models):
    """Model information page"""
    
    st.header("📊 Model Information")
    
    st.subheader("🤖 Loaded Models")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", len(models))
    with col2:
        st.metric("Categories", len(categories))
    with col3:
        st.metric("Status", "✅ Ready")
    
    st.markdown("### Model Details")
    
    model_info = pd.DataFrame({
        'Model Name': list(models.keys()),
        'Type': ['Linear', 'Ensemble', 'SVM', 'Probabilistic', 'Boosting'],
        'Supports Probability': ['✅' if hasattr(model, 'predict_proba') else '❌' 
                                 for model in models.values()]
    })
    
    st.dataframe(model_info, use_container_width=True)
    
    st.markdown("### 📋 Supported Categories")
    
    num_cols = 3
    cols = st.columns(num_cols)
    for idx, category in enumerate(sorted(categories)):
        with cols[idx % num_cols]:
            st.markdown(f"✅ {category}")

def show_about_page():
    """About page"""
    
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎓 Resume Classification System
    
    This application uses **pre-trained Machine Learning models** to automatically 
    classify resumes into different job categories.
    
    ### 🔬 Technical Details
    
    #### **Models Used:**
    - **Logistic Regression**: Linear classification with regularization
    - **Random Forest**: Ensemble of 100 decision trees
    - **SVM**: Support Vector Machine with linear kernel
    - **Naive Bayes**: Multinomial probabilistic classifier
    - **XGBoost**: Gradient boosting with depth 6
    
    #### **Features:**
    - TF-IDF vectorization (3000 features)
    - N-gram range: 1-2
    - Text preprocessing: lemmatization, stopword removal
    
    ### 🛠️ Technologies
    
    - **Frontend**: Streamlit
    - **ML Libraries**: scikit-learn, XGBoost
    - **NLP**: NLTK
    - **PDF Processing**: PyPDF2
    - **Visualization**: Plotly
    
    ### 📝 Deployment
    
    Models are pre-trained and stored as pickle files for fast loading and inference.
    
    ---
    
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Made with ❤️ using Streamlit and Machine Learning</p>
        <p>© 2025 Resume Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
