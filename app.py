import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import PyPDF2
import docx
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# File paths for saved models
MODEL_PATH = 'resume_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# Text preprocessing function
def clean_text(text):
    """Clean and preprocess text data"""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error in text preprocessing: {e}")
        return ""

# Extract text from uploaded files
def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files"""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
            return text.strip()

        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            text = " ".join([para.text for para in doc.paragraphs])
            return text.strip()

        elif file_type == "txt":
            text = uploaded_file.read().decode('utf-8')
            return text.strip()

        else:
            st.error("Unsupported file format! Please upload PDF, DOCX, or TXT files.")
            return None

    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return None

# Train and save model
def train_and_save_model():
    """Train the model and save it along with vectorizer and encoder"""
    try:
        with st.spinner("Training model... This may take a few minutes."):
            # Load dataset
            if not os.path.exists("UpdatedResumeDataSet.csv"):
                st.error("Dataset file 'UpdatedResumeDataSet.csv' not found! Please upload it to the app directory.")
                return False

            df = pd.read_csv("UpdatedResumeDataSet.csv")

            # Check required columns
            if 'Resume' not in df.columns or 'Category' not in df.columns:
                st.error("Dataset must contain 'Resume' and 'Category' columns!")
                return False

            st.info(f"Dataset loaded: {df.shape[0]} resumes, {df['Category'].nunique()} categories")

            # Preprocess text
            st.info("Preprocessing text data...")
            df['cleaned'] = df['Resume'].apply(clean_text)

            # TF-IDF Vectorization
            st.info("Creating TF-IDF features...")
            vectorizer = TfidfVectorizer(
                max_features=3000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )

            X = vectorizer.fit_transform(df['cleaned'])
            y = df['Category']

            # Label encoding
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Train Random Forest model
            st.info("Training Random Forest classifier...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Calculate accuracy
            accuracy = model.score(X_test, y_test)
            st.success(f"Model trained successfully! Test Accuracy: {accuracy:.2%}")

            # Save model, vectorizer, and encoder
            st.info("Saving model, vectorizer, and label encoder...")
            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)
            joblib.dump(le, ENCODER_PATH)

            st.success("✅ Model, vectorizer, and encoder saved successfully!")
            return True

    except Exception as e:
        st.error(f"Error during training: {e}")
        return False

# Load saved model
@st.cache_resource
def load_model():
    """Load the saved model, vectorizer, and encoder"""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            le = joblib.load(ENCODER_PATH)
            return model, vectorizer, le
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Main Streamlit App
def main():
    st.set_page_config(page_title="Resume Category Classifier", page_icon="📄", layout="wide")

    # Title and description
    st.title("📄 Resume Category Classifier")
    st.markdown("""
    This application classifies resumes into job categories using Machine Learning.

    **Features:**
    - Upload resume files (PDF, DOCX, TXT)
    - Get predicted job category with confidence score
    - Retrain model on new data

    **Instructions:**
    1. First time? Click "Retrain Model" to train the classifier
    2. Upload your resume file
    3. Get instant predictions!
    """)

    st.markdown("---")

    # Check if model exists
    model, vectorizer, le = load_model()

    if model is None:
        st.warning("⚠️ Model not found! Please train the model first.")
        st.info("Click the 'Retrain Model' button below to train a new model.")
    else:
        st.success("✅ Model loaded successfully!")

    # Sidebar for model retraining
    st.sidebar.header("🔧 Model Management")

    if st.sidebar.button("🔄 Retrain Model", help="Train a new model on the dataset"):
        success = train_and_save_model()
        if success:
            st.sidebar.success("Model retrained! Please refresh the page.")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About:**

    This app uses:
    - TF-IDF Vectorization
    - Random Forest Classifier
    - NLTK for text preprocessing

    Dataset: UpdatedResumeDataSet.csv
    """)

    # Main prediction interface
    st.header("📤 Upload Resume for Classification")

    uploaded_file = st.file_uploader(
        "Choose a resume file (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload your resume in PDF, DOCX, or TXT format"
    )

    if uploaded_file is not None:
        st.info(f"File uploaded: **{uploaded_file.name}**")

        # Extract text
        resume_text = extract_text_from_file(uploaded_file)

        if resume_text and model is not None:
            # Show extracted text (optional)
            with st.expander("📝 View Extracted Text"):
                st.text_area("Resume Content", resume_text, height=200)

            # Predict category
            if st.button("🔍 Predict Category", type="primary"):
                with st.spinner("Analyzing resume..."):
                    try:
                        # Preprocess and vectorize
                        cleaned_text = clean_text(resume_text)
                        text_vectorized = vectorizer.transform([cleaned_text])

                        # Predict
                        prediction = model.predict(text_vectorized)[0]
                        prediction_proba = model.predict_proba(text_vectorized)[0]

                        # Get category name and confidence
                        category = le.inverse_transform([prediction])[0]
                        confidence = np.max(prediction_proba) * 100

                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Predicted Category", category)

                        with col2:
                            st.metric("Confidence", f"{confidence:.2f}%")

                        # Show top 3 predictions
                        st.markdown("### Top 3 Predicted Categories")
                        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]

                        for idx in top_3_indices:
                            cat_name = le.inverse_transform([idx])[0]
                            cat_prob = prediction_proba[idx] * 100
                            st.write(f"**{cat_name}**: {cat_prob:.2f}%")

                        st.success("✅ Classification complete!")

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

        elif resume_text is None:
            st.error("Could not extract text from the file. Please try another file.")

        elif model is None:
            st.error("Model not loaded. Please train the model first using the 'Retrain Model' button.")

    else:
        st.info("👆 Upload a resume file to get started")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit 🎈 | Powered by Machine Learning 🤖</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
