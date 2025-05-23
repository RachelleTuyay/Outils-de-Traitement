import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

nltk.download('punkt')
nltk.download('stopwords')

with open("../../data/clean/augmented_corpus.json", "r", encoding="utf-8") as f:
    list_of_unpreprocessed_documents = json.load(f)

# Prétraitement
stop_words = set(stopwords.words('french'))

def preprocess(text):
    tokens = word_tokenize(text.lower(), language='french')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

list_of_preprocessed_documents = [preprocess(doc) for doc in list_of_unpreprocessed_documents]

# Préparation des données pour CTM
qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")

import os
folder_path = "../data/raw"  #pour avoir un contexte, on utilise les données brutes
file_contents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            file_contents.append(f.read())

list_of_unpreprocessed_documents = file_contents
list_of_preprocessed_documents = [preprocess(doc) for doc in file_contents]

# Création du jeu de données pour CTM
training_dataset = qt.fit(
    text_for_contextual=list_of_unpreprocessed_documents,
    text_for_bow=list_of_preprocessed_documents
)

# Entraînement du modèle CTM
ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50)  # 50 topics
ctm.fit(training_dataset)

# Sauvegarde des sujets dans un fichier texte
with open("ctm_topics.txt", "w", encoding="utf-8") as f:
    for i, topic in enumerate(ctm.get_topic_lists(10)):  # top 10 mots par sujet
        f.write(f"Sujet {i}: {', '.join(topic)}\n")
