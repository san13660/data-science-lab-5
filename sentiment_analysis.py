# Load EDA packages
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en')

# Build a list of stopwords to use to filter
stopwords = list(STOP_WORDS)

# Use the punctuations of string module
import string
punctuations = string.punctuation

# Creating a Spacy Parser
from spacy.lang.en import English
parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    return mytokens

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

#Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

# Vectorization
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
classifier = LinearSVC()

# Using Tfidf
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)

# Splitting Data Set
from sklearn.model_selection import train_test_split

# Features and Labels
df = pd.read_csv('preprocesado_GrammarandProductReviews.csv')
X = df['reviews.text']
ylabels = df['reviews.rating']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

# Create the  pipeline to clean, tokenize, vectorize, and classify
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# Fit our data
pipe.fit(X_train,y_train)

# Predicting with a test dataset
sample_prediction = pipe.predict(X_test)

# Prediction Results
# 1 = Positive review
# 0 = Negative review
for (sample,pred) in zip(X_test,sample_prediction):
    print(sample,"Prediction=>",pred)

# Accuracy
print("Accuracy: ",pipe.score(X_test,y_test))
print("Accuracy: ",pipe.score(X_test,sample_prediction))

# Accuracy
print("Accuracy: ",pipe.score(X_train,y_train))

# Another random review
pipe.predict(["This was a great movie"])


#### Using Tfid
# Create the  pipeline to clean, tokenize, vectorize, and classify
pipe_tfid = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfvectorizer),
                 ('classifier', classifier)])
pipe_tfid.fit(X_train,y_train)
sample_prediction1 = pipe_tfid.predict(X_test)
for (sample,pred) in zip(X_test,sample_prediction1):
    print(sample,"Prediction=>", pred)

print("Accuracy: ", pipe_tfid.score(X_test, y_test))
print("Accuracy: ", pipe_tfid.score(X_test, sample_prediction1))