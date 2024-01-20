from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the 20 newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train')# Convert the text data into word embeddings using Word2Vec
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroups_train.data)
model = LogisticRegression().fit(X, newsgroups_train.target)# Use the model to classify new documents
newsgroups_test = fetch_20newsgroups(subset='test')
X_test = vectorizer.transform(newsgroups_test.data)
predicted = model.predict(X_test)