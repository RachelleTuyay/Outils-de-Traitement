import json
import spacy
import gensim
from gensim import corpora
from gensim.models import CoherenceModel, TfidfModel
import nltk
from nltk.corpus import stopwords
import logging
import pyLDAvis.gensim_models

# Pour voir la progression et les logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load('fr_core_news_sm')
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

# Charger les documents
with open("../../data/clean/augmented_corpus.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Nettoyage + Lemmatisation + suppression des stopwords
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in stop_words and len(token) > 2
    ]
    return tokens

texts = [preprocess(doc) for doc in documents]

# Création du dictionnaire
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Corpus BoW
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Diversité des sujets
def topic_diversity(model, topn=10):
    topic_words = [set([word for word, _ in model.show_topic(i, topn=topn)]) for i in range(model.num_topics)]
    num_overlaps = sum(len(a & b) for i, a in enumerate(topic_words) for j, b in enumerate(topic_words) if i < j)
    max_pairs = model.num_topics * (model.num_topics - 1) / 2
    return 1 - num_overlaps / (topn * max_pairs)

# Entraîner LDA et calcul des métriques
def train_lda(corpus_input, num_topics=25, passes=30, alpha='auto', eta='auto'):
    lda = gensim.models.LdaModel(
        corpus=corpus_input,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha=alpha,
        eta=eta,
        random_state=42
    )
    coherence_c_v = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
    coherence_umass = CoherenceModel(model=lda, corpus=corpus_input, dictionary=dictionary, coherence='u_mass').get_coherence()
    perplexity = lda.log_perplexity(corpus_input)
    diversity = topic_diversity(lda)
    return lda, coherence_c_v, coherence_umass, perplexity, diversity

# Entraînement sur BoW
lda_bow, coh_bow_cv, coh_bow_umass, perp_bow, div_bow = train_lda(corpus)

# Entraînement sur TF-IDF
lda_tfidf, coh_tfidf_cv, coh_tfidf_umass, perp_tfidf, div_tfidf = train_lda(corpus_tfidf)

# Choix du meilleur modèle basé sur cohérence c_v
best_lda = lda_tfidf if coh_tfidf_cv > coh_bow_cv else lda_bow
best_metrics = (coh_tfidf_cv, coh_tfidf_umass, perp_tfidf, div_tfidf) if best_lda == lda_tfidf else (coh_bow_cv, coh_bow_umass, perp_bow, div_bow)

# Sauvegarde des sujets et mesures
with open("lda_topics_fined.txt", "w", encoding="utf-8") as f:
    for idx, topic in best_lda.print_topics(-1):
        f.write(f"Sujet {idx}: {topic}\n\n")
    f.write("\n===== EVALUATIONS ====\n")
    f.write(f"Cohérence c_v LDA BoW: {coh_bow_cv:.4f}\n")
    f.write(f"Cohérence u_mass LDA BoW: {coh_bow_umass:.4f}\n")
    f.write(f"Perplexité LDA BoW: {perp_bow:.4f}\n")
    f.write(f"Diversité des sujets LDA BoW: {div_bow:.4f}\n\n")

    f.write(f"Cohérence c_v LDA TF-IDF: {coh_tfidf_cv:.4f}\n")
    f.write(f"Cohérence u_mass LDA TF-IDF: {coh_tfidf_umass:.4f}\n")
    f.write(f"Perplexité LDA TF-IDF: {perp_tfidf:.4f}\n")
    f.write(f"Diversité des sujets LDA TF-IDF: {div_tfidf:.4f}\n")

# Visualisation des topics (figure interactive)
fig = pyLDAvis.gensim_models.prepare(best_lda, corpus, dictionary)
pyLDAvis.save_html(fig, "lda_visualization.html")
print("Visualisation LDA sauvegardée dans lda_visualization.html")
