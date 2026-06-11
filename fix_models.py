import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import nltk
import re

print("Loading data...")
resume_data = pd.read_csv('UpdatedResumeDataSet.csv')

def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub('#\S+', '', resume_text)
    resume_text = re.sub('@\S+', '  ', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)
    return resume_text.lower()

resume_data['cleaned_resume'] = resume_data['Resume'].apply(lambda x: clean_resume(x))

le = LabelEncoder()
y = le.fit_transform(resume_data['Category'])

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = tfidf.fit_transform(resume_data['cleaned_resume'])

models = {
    'LogisticRegression.pkl': LogisticRegression(),
    'RandomForest.pkl': RandomForestClassifier(),
    'SVM.pkl': SVC(probability=True),
    'NaiveBayes.pkl': MultinomialNB(),
    'XGBoost.pkl': XGBClassifier()
}

print("Training and saving models...")
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

categories = list(le.classes_)
with open('models/categories.pkl', 'wb') as f:
    pickle.dump(categories, f)

for filename, model in models.items():
    model.fit(X, y)
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model, f)

print("Done!")
