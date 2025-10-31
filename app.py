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
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'categories' not in st.session_state:
    st.session_state.categories = []

# Text preprocessing function
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

# PDF text extraction
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

# Load dataset from GitHub
@st.cache_data
def load_dataset_from_github(github_url):
    """Load dataset from GitHub raw URL"""
    try:
        df = pd.read_csv(github_url)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Train models
def train_models(dataset, progress_bar=None):
    """Train all classification models"""
    
    # Preprocess data
    if progress_bar:
        progress_bar.progress(10, text="Preprocessing text data...")
    
    dataset['cleaned'] = dataset['Resume'].apply(clean_text)
    
    # TF-IDF Vectorization
    if progress_bar:
        progress_bar.progress(20, text="Creating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1,2),
        stop_words='english'
    )
    
    X = dataset['cleaned']
    y = dataset['Category']
    X_tfidf = vectorizer.fit_transform(X)
    
    # Train-test split
    if progress_bar:
        progress_bar.progress(30, text="Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'SVM': SVC(C=1.0, kernel='linear', probability=True, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'XGBoost': XGBClassifier(max_depth=6, learning_rate=0.1, random_state=42, 
                                 use_label_encoder=False, eval_metric='mlogloss')
    }
    
    # Train models
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    progress_step = 50 / len(models)
    current_progress = 30
    
    for name, model in models.items():
        if progress_bar:
            current_progress += progress_step
            progress_bar.progress(int(current_progress), text=f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train_enc, cv=cv, scoring='accuracy')
        
        # Train model
        model.fit(X_train, y_train_enc)
        
        # Predictions
        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    if progress_bar:
        progress_bar.progress(100, text="Training complete!")
    
    return vectorizer, le, models, results, (X_test, y_test)

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🎯 AI-Powered Resume Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automatically categorize resumes using Machine Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/resume.png", width=100)
        st.title("Navigation")
        
        page = st.radio("Select Page", [
            "🏠 Home",
            "📊 Model Training",
            "🔍 Resume Classification",
            "📈 Model Performance",
            "ℹ️ About"
        ])
        
        st.markdown("---")
        st.markdown("### 📌 Quick Info")
        st.info(f"""
        **Training Status:** {'✅ Trained' if st.session_state.models_trained else '❌ Not Trained'}
        
        **Categories:** {len(st.session_state.categories) if st.session_state.categories else 0}
        
        **Models:** 5 classifiers
        """)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Model Training":
        show_training_page()
    elif page == "🔍 Resume Classification":
        show_classification_page()
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Home page with introduction"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to Resume Classifier! 👋
        
        This application uses advanced **Machine Learning algorithms** to automatically 
        classify resumes into different job categories.
        
        ### 🎯 Key Features:
        
        - **Multi-Model Approach**: Uses 5 different ML algorithms
        - **High Accuracy**: Trained on comprehensive resume dataset
        - **PDF Support**: Upload resumes in PDF format
        - **Real-time Prediction**: Get instant classification results
        - **Detailed Analytics**: View model performance metrics
        
        ### 🚀 How to Use:
        
        1. **Train Models**: Go to Model Training page and load dataset
        2. **Upload Resume**: Navigate to Resume Classification page
        3. **Get Results**: View predictions from all models
        4. **Analyze Performance**: Check model metrics
        
        ### 📊 Supported Models:
        
        - Logistic Regression
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - Naive Bayes
        - XGBoost Classifier
        """)
        
        if not st.session_state.models_trained:
            st.warning("⚠️ **Models not trained yet!** Please go to the Model Training page.")
            if st.button("Go to Training Page"):
                st.rerun()

def show_training_page():
    """Model training page"""
    
    st.header("📊 Model Training Dashboard")
    
    st.markdown("""
    <div class="info-box">
    <b>ℹ️ Training Information:</b><br>
    This process will download the dataset from GitHub and train 5 different ML models.
    Training may take a few minutes depending on dataset size.
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset URL input
    st.subheader("1️⃣ Dataset Configuration")
    
    default_url = "https://github.com/namannb07/ResumeClassifier/blob/main/UpdatedResumeDataSet.csv"
    github_url = st.text_input(
        "Enter GitHub Raw URL for Dataset",
        value=default_url,
        help="Paste the raw GitHub URL of your UpdatedResumeDataSet.csv file"
    )
    
    st.markdown("**Example URL format:**")
    st.code("https://raw.githubusercontent.com/username/repository/branch/UpdatedResumeDataSet.csv")
    
    # Load dataset button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 Load Dataset", use_container_width=True):
            with st.spinner("Loading dataset from GitHub..."):
                dataset = load_dataset_from_github(github_url)
                
                if dataset is not None:
                    st.session_state.dataset = dataset
                    st.session_state.categories = dataset['Category'].unique().tolist()
                    st.success(f"✅ Dataset loaded successfully! ({len(dataset)} resumes)")
                else:
                    st.error("❌ Failed to load dataset. Please check the URL.")
    
    # Display dataset info
    if st.session_state.dataset is not None:
        st.subheader("2️⃣ Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", len(st.session_state.dataset))
        with col2:
            st.metric("Categories", st.session_state.dataset['Category'].nunique())
        with col3:
            st.metric("Columns", len(st.session_state.dataset.columns))
        with col4:
            st.metric("Status", "✅ Ready")
        
        # Category distribution
        st.subheader("📊 Category Distribution")
        
        category_counts = st.session_state.dataset['Category'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={'x': 'Category', 'y': 'Count'},
            title='Resume Distribution by Category',
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        with st.expander("🔍 View Sample Data"):
            st.dataframe(st.session_state.dataset.head(10))
        
        # Train models button
        st.subheader("3️⃣ Model Training")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Train All Models", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="Initializing training...")
                
                try:
                    vectorizer, le, models, results, test_data = train_models(
                        st.session_state.dataset,
                        progress_bar
                    )
                    
                    # Store in session state
                    st.session_state.vectorizer = vectorizer
                    st.session_state.label_encoder = le
                    st.session_state.models = models
                    st.session_state.model_results = results
                    st.session_state.models_trained = True
                    
                    st.success("🎉 All models trained successfully!")
                    
                    # Display results
                    st.subheader("📈 Training Results")
                    
                    results_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Test Accuracy': [results[name]['accuracy'] for name in results.keys()],
                        'CV Mean': [results[name]['cv_mean'] for name in results.keys()],
                        'CV Std': [results[name]['cv_std'] for name in results.keys()]
                    }).sort_values('Test Accuracy', ascending=False)
                    
                    st.dataframe(
                        results_df.style.highlight_max(subset=['Test Accuracy', 'CV Mean'], color='lightgreen')
                        .format({'Test Accuracy': '{:.4f}', 'CV Mean': '{:.4f}', 'CV Std': '{:.4f}'}),
                        use_container_width=True
                    )
                    
                    # Best model
                    best_model = results_df.iloc[0]['Model']
                    best_accuracy = results_df.iloc[0]['Test Accuracy']
                    
                    st.balloons()
                    st.success(f"🏆 **Best Model:** {best_model} with {best_accuracy:.2%} accuracy")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
    else:
        st.info("👆 Please load the dataset first to proceed with training.")

def show_classification_page():
    """Resume classification page"""
    
    st.header("🔍 Resume Classification")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models are not trained yet. Please train the models first!")
        if st.button("Go to Training Page"):
            st.rerun()
        return
    
    st.markdown("""
    <div class="info-box">
    <b>📄 Upload a Resume:</b><br>
    Upload a PDF resume to classify it into one of the job categories.
    The system will analyze the resume using all 5 trained models.
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
                    text_vec = st.session_state.vectorizer.transform([cleaned_text])
                
                # Predictions
                st.subheader("🎯 Classification Results")
                
                predictions = {}
                for model_name, model in st.session_state.models.items():
                    pred_enc = model.predict(text_vec)[0]
                    pred_label = st.session_state.label_encoder.inverse_transform([pred_enc])[0]
                    
                    if hasattr(model, "predict_proba"):
                        prob = np.max(model.predict_proba(text_vec))
                    else:
                        prob = None
                    
                    predictions[model_name] = {
                        'category': pred_label,
                        'confidence': prob
                    }
                
                # Find consensus prediction
                pred_categories = [pred['category'] for pred in predictions.values()]
                from collections import Counter
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
                
                # Individual model predictions
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
                
                # Detailed predictions table
                st.subheader("📋 Detailed Predictions")
                
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted Category': [pred['category'] for pred in predictions.values()],
                    'Confidence': [f"{pred['confidence']:.2%}" if pred['confidence'] else "N/A" 
                                  for pred in predictions.values()]
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                # Download results
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=pred_df.to_csv(index=False),
                    file_name=f"resume_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_performance_page():
    """Model performance page"""
    
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models are not trained yet. Please train the models first!")
        return
    
    results = st.session_state.model_results
    
    # Overall metrics
    st.subheader("🎯 Overall Model Comparison")
    
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'CV Mean Accuracy': [results[name]['cv_mean'] for name in results.keys()],
        'CV Std': [results[name]['cv_std'] for name in results.keys()]
    }).sort_values('Test Accuracy', ascending=False)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", results_df.iloc[0]['Model'])
    with col2:
        st.metric("Best Accuracy", f"{results_df.iloc[0]['Test Accuracy']:.2%}")
    with col3:
        st.metric("Average Accuracy", f"{results_df['Test Accuracy'].mean():.2%}")
    with col4:
        st.metric("Models Trained", len(results))
    
    # Accuracy comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Test Accuracy',
        x=results_df['Model'],
        y=results_df['Test Accuracy'],
        marker_color='skyblue'
    ))
    
    fig.add_trace(go.Bar(
        name='CV Mean Accuracy',
        x=results_df['Model'],
        y=results_df['CV Mean Accuracy'],
        marker_color='salmon',
        error_y=dict(
            type='data',
            array=results_df['CV Std'],
            visible=True
        )
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        barmode='group',
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📊 Detailed Metrics")
    
    st.dataframe(
        results_df.style.highlight_max(subset=['Test Accuracy', 'CV Mean Accuracy'], color='lightgreen')
        .format({'Test Accuracy': '{:.4f}', 'CV Mean Accuracy': '{:.4f}', 'CV Std': '{:.4f}'}),
        use_container_width=True
    )
    
    # Categories
    st.subheader("📋 Supported Categories")
    
    if st.session_state.categories:
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, category in enumerate(sorted(st.session_state.categories)):
            with cols[idx % num_cols]:
                st.markdown(f"✅ {category}")

def show_about_page():
    """About page with project information"""
    
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎓 Resume Classification System
    
    This application demonstrates the power of **Machine Learning** in automating 
    the resume screening process. It uses multiple classification algorithms to 
    categorize resumes into different job domains.
    
    ### 🔬 Technical Details
    
    #### **Machine Learning Pipeline:**
    
    1. **Text Preprocessing**
       - Lowercasing
       - Special character removal
       - Tokenization
       - Stopword removal
       - Lemmatization
    
    2. **Feature Extraction**
       - TF-IDF Vectorization
       - N-gram features (1-gram and 2-gram)
       - Max features: 3000
    
    3. **Model Training**
       - 5 different algorithms
       - 5-fold cross-validation
       - Stratified train-test split (80-20)
    
    #### **Models Used:**
    
    - **Logistic Regression**: Linear model with regularization
    - **Random Forest**: Ensemble of decision trees
    - **SVM**: Support Vector Machine with linear kernel
    - **Naive Bayes**: Probabilistic classifier
    - **XGBoost**: Gradient boosting algorithm
    
    ### 📊 Dataset Information
    
    The model is trained on a comprehensive dataset containing resumes from 
    multiple job categories including:
    
    - Data Science
    - Web Development
    - Database Administration
    - DevOps Engineering
    - Business Analysis
    - And many more...
    
    ### 🛠️ Technologies Used
    
    - **Frontend**: Streamlit
    - **ML Libraries**: scikit-learn, XGBoost
    - **NLP**: NLTK
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    - **PDF Processing**: PyPDF2
    
    ### 👨‍💻 Developer Information
    
    **Project**: Resume Classification System  
    **Purpose**: Internal Demonstration / Academic Project  
    **Technology Stack**: Python, Streamlit, Machine Learning  
    **Dataset Source**: GitHub Repository
    
    ### 📝 Usage Instructions
    
    1. Navigate to **Model Training** page
    2. Enter GitHub raw URL for the dataset
    3. Click **Load Dataset** to fetch data
    4. Click **Train All Models** to train classifiers
    5. Go to **Resume Classification** page
    6. Upload a resume PDF
    7. View predictions from all models
    
    ### 🔒 Privacy & Security
    
    - All processing is done locally
    - No data is stored permanently
    - Resume content is processed in-memory only
    - Models are trained on publicly available datasets
    
    ### 📧 Contact & Feedback
    
    For questions, suggestions, or feedback about this project, please contact 
    the development team.
    
    ---
    
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Made with ❤️ using Streamlit and Machine Learning</p>
        <p>© 2025 Resume Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
