# Outils de Traitement :
Dépôt Github pour le cours "Outils de traitement de corpus" en Master 1 TAL (à INALCO, Université Sorbonne Nouvelle (P3), Université Paris Nanterre (P10) )

---

## PROJET : Détection des tendances d'actualité à partir d'articles de presse en ligne

- Dans quel besoin vous inscrivez-vous ?
L’analyse des tendances d’actualité est essentielle pour comprendre l’évolution de l’information médiatique sur des sujets spécifiques. Un système capable de détecter automatiquement ces tendances à partir d’un corpus d’articles de presse peut être utile pour les chercheurs, journalistes ou entreprises souhaitant suivre l’évolution des thématiques couvertes par les médias.

- Quel sujet allez-vous traiter ?
Le projet consiste à analyser un grand nombre d’articles de presse collectés en ligne pour identifier les tendances d’actualité sur une période donnée. Cela inclut la détection des sujets les plus couverts, des entités fréquemment mentionnées (personnes, organisations, lieux), ainsi que des mots-clés ou expressions récurrentes.
Quel type de tâche allez-vous réaliser ?

- La tâche principale est l’extraction de tendances à partir du contenu textuel des articles :
  - Extraction de mots-clés ou expressions fréquentes
  - Reconnaissance d’entités nommées (NER)
  - Classement des sujets les plus abordés selon leur fréquence
  - Possibilité de regroupement thématique ou de visualisation temporelle des tendances

- Quel type de données allez-vous exploiter ?
Des articles de presse, incluant leur titre, le chapeau (sous-titre), et le corps du texte. Ces données pourront être filtrées selon la date de publication, le média source, ou la section thématique (politique, économie, sport, etc.).

- Où allez-vous récupérer vos données ?
Les articles seront collectés via web scraping sur des sites de presse en ligne (par exemple : Le Monde, France Info, Le Figaro, etc.) en utilisant des outils comme BeautifulSoup, Scrapy, ou Selenium en Python.

- Sont-elles libres d’accès ?
L’accès aux articles dépend des conditions d’utilisation des sites web. Certains articles sont en accès libre, tandis que d’autres peuvent être soumis à des restrictions ou nécessiter une authentification (comme les fils RSS ou les sites d’agences de presse à contenu ouvert).
