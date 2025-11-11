# Classification des documents avec LLM
## 1. objectif
L’objectif de ce projet est de réaliser une classification multi-classes de documents textuels en utilisant des modèles Transformers pré-entraînés :
- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)
L’idée est de comparer leurs performances sur la même tâche.

## 2. Dataset
Pour ce projet, nous avons utilisé le dataset 20 Newsgroups, un corpus public de textes largement utilisé pour les tâches de classification de documents. Ce dataset contient environ 18 000 articles répartis en 20 catégories différentes, couvrant des sujets variés tels que l’informatique, le sport, la religion, la politique, et la science.
Chaque document est un article de forum extrait d’un newsgroup, et les catégories représentent les sujets principaux. Pour faciliter l’apprentissage, nous avons retiré les en-têtes, les pieds de page et les citations, de manière à ce que le modèle se concentre sur le contenu textuel principal.
Le dataset a été divisé en trois ensembles :
- Entraînement (train) : 80 % des documents
- Validation (val) : 10 % des documents
- Test : 10 % des documents
Cette organisation garantit une évaluation fiable des modèles tout en conservant suffisamment de données pour l’entraînement.

## 3. Prétraitement
Avant de passer à l’entraînement des modèles, les textes du dataset ont été soigneusement prétraités pour améliorer la qualité des entrées et optimiser les performances des modèles Transformers.

Nettoyage du texte
- Suppression des espaces multiples pour uniformiser les textes.
- Retrait des liens (URLs) pour éviter le bruit dans le contenu textuel.
- Suppression des caractères spéciaux inutiles, ne conservant que les lettres, chiffres et ponctuation utile.

Ce nettoyage permet de réduire le bruit et d’améliorer la compréhension sémantique par les modèles.

Tokenization
Utilisation des tokenizers pré-entraînés correspondant à chaque modèle :
- BertTokenizer pour BERT
- RobertaTokenizerFast pour RoBERTa
Limitation de la longueur des séquences à 256 tokens, avec troncature et padding automatique.

Encodage des labels
- Les catégories textuelles du dataset ont été transformées en valeurs numériques avec un LabelEncoder.
- Chaque document est donc associé à un label entier correspondant à sa classe.

Création des datasets et DataLoaders
- Conversion des données tokenisées en datasets PyTorch pour un traitement efficace.
- Création de DataLoaders pour l’entraînement, la validation et le test, avec un batch size de 16.

Cette préparation assure que les modèles reçoivent des entrées propres et correctement formatées, prêtes pour le fine-tuning sur la classification multi-classes.

## 4. Modèles utilisés
Pour la classification des documents, nous avons exploité deux modèles Transformers pré-entraînés très performants dans le traitement du langage naturel : BERT et RoBERTa. Ces modèles sont capables de capturer le sens contextuel des mots grâce à des représentations bidirectionnelles, ce qui est particulièrement utile pour comprendre et classifier des textes variés.

### 4.1. Modèle BERT
Le premier modèle, BERT (Bidirectional Encoder Representations from Transformers), dans sa version bert-base-uncased, a été pré-entraîné sur un large corpus de textes afin d’apprendre des représentations contextuelles profondes. Pour notre projet, nous avons effectué un fine-tuning en ajoutant une couche dense de sortie adaptée au nombre de classes du dataset, soit 20 catégories. Ce modèle permet de comprendre le contexte dans les deux directions de lecture, ce qui améliore la précision de la classification, surtout sur des textes de longueur moyenne.

### 4.2. Modèle RoBERTa
Le deuxième modèle, RoBERTa (Robustly Optimized BERT Approach), est une variante optimisée de BERT qui bénéficie d’un entraînement sur plus de données et d’améliorations méthodologiques, comme la suppression de la tâche de Next Sentence Prediction et une meilleure régularisation. Nous avons utilisé la version roberta-base et procédé à un fine-tuning similaire à celui de BERT pour la classification multi-classes. RoBERTa est reconnu pour sa robustesse et sa capacité à généraliser sur des textes variés et parfois bruités, ce qui le rend souvent plus performant que BERT sur plusieurs benchmarks NLP.

## 5. Résultat
L’entraînement des modèles BERT et RoBERTa sur le corpus 20 Newsgroups a permis d’obtenir des performances proches, avec une précision globale d’environ 77% et un F1-Macro autour de 0.76. Les deux modèles capturent bien les thématiques à vocabulaire stable comme sport, informatique ou vente, ce qui se traduit par de bonnes prédictions sur ces catégories. En revanche, certaines classes présentant soit un vocabulaire plus subjectif, soit des frontières thématiques proches (par exemple alt.atheism ou rec.autos), montrent des scores plus faibles, traduisant une confusion entre certains groupes. Globalement, le comportement observé est cohérent pour une tâche multiclasse textuelle avec longueur de documents variable et nuance sémantique élevée.
