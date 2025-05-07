from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import spacy
from nltk.stem import PorterStemmer
import csv
import re


nlp = spacy.load("en_core_web_sm") # Load the English NLP model

stemmer = PorterStemmer() # Initialize the stemmer
def create_corpus_and_labels(file_path):
    corpus = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus.append(row["text"])
            labels.append(int(row["spam"]))  # spam column is either 0 or 1
    return corpus, labels


def custom_preprocessor(text):
    text = re.sub(r'\d+', '', text)  # Remove digits
    doc = nlp(text)
    lemmatized_words = []

    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            lemmatized_words.append(token.lemma_)

    stemmed = []
    for word in lemmatized_words:
        stemmed.append(stemmer.stem(word))
    return ' '.join(stemmed)



corpus, labels = create_corpus_and_labels('L06_NLP_LLM_emails.csv')

vectorizer = CountVectorizer(
    stop_words='english',
    preprocessor=custom_preprocessor,
    lowercase=True
)

X = vectorizer.fit_transform(corpus)


print(X.shape)  # (number of emails, number of features)
print(vectorizer.get_feature_names_out()[:10])  # Print first 10 words


# Comparing MLP VS NB
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)


mlp_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', preprocessor=custom_preprocessor)),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])

nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', preprocessor=custom_preprocessor)),
    ('classifier', MultinomialNB())
])



mlp_pipeline.fit(X_train, y_train)
y_pred_mlp = mlp_pipeline.predict(X_test)

nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)

# Evaluate the model
accuracy_mlp = metrics.accuracy_score(y_test, y_pred_mlp)
print(f'MLP Accuracy: {accuracy_mlp:.2f}')
print(metrics.classification_report(y_test, y_pred_mlp))

accuracy_nb = metrics.accuracy_score(y_test, y_pred_nb)
print(f'NB Accuracy: {accuracy_nb:.2f}')
print(metrics.classification_report(y_test, y_pred_nb))
