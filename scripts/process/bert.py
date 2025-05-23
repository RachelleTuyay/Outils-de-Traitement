from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
import json

# Step 1 - Embedding model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Step 2 - Dimensionality reduction
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# Step 3 - Clustering model
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Vectorizer with French stopwords
french_stopwords = stopwords.words("french")
vectorizer_model = CountVectorizer(stop_words=french_stopwords)

# Step 5 - TF-IDF weighting
ctfidf_model = ClassTfidfTransformer()

# Step 6 - Topic representation model
representation_model = KeyBERTInspired()

with open("../../data/clean/augmented_corpus.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Remove empty docs or short texts
docs = [doc.strip() for doc in docs if isinstance(doc, str) and len(doc.strip()) > 20]

#BERTopic model
topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model
)

embeddings = embedding_model.encode(docs, show_progress_bar=True)
topics, probs = topic_model.fit_transform(docs, embeddings)

# === Display topic info ===
print(topic_model.get_topic_info())
print(topic_model.get_document_info(docs))

# === Sauvegarde des résultats dans un fichier texte ===
with open("bertopic_topics.txt", "w", encoding="utf-8") as f:
    f.write("=== Résumé des topics ===\n\n")
    for topic in topic_model.get_topic_info().iterrows():
        row = topic[1]
        f.write(f"Sujet {row['Topic']} ({row['Count']} documents): {row['Name']}\n")

    f.write("\n=== Attribution des documents ===\n\n")
    doc_info = topic_model.get_document_info(docs)
    for i, row in doc_info.iterrows():
        f.write(f"Document {i} → Topic {row['Topic']} (Probabilité: {row['Probability']:.2f})\n")
        f.write(f"Texte: {docs[i][:200]}...\n\n")  # Sauvegarde les 200 premiers caractères du texte
