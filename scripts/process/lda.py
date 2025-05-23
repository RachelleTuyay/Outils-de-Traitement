import json
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

with open("../../data/clean/augmented_corpus.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

stop_words = set(stopwords.words('french'))
def preprocess(text):
    tokens = word_tokenize(text.lower(), language='french')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

texts = [preprocess(doc) for doc in documents]

#Création du dictionnaire et corpus BoW
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

#Entraînement du modèle LDA
lda_model = gensim.models.LdaModel(
    corpus,
    num_topics=25,
    id2word=dictionary,
    passes=15,
    random_state=42
)

#Sauvegarde des sujets dans un fichier texte
with open("lda_topics.txt", "w", encoding="utf-8") as f:
    for idx, topic in lda_model.print_topics(-1):
        f.write(f"Sujet {idx}: {topic}\n\n")
