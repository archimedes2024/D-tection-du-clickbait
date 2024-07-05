import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, bigrams, trigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
import string
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, fbeta_score, confusion_matrix

# Télécharger les stopwords et le lemmatizer de nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#%%

original_dataset = pd.read_csv(r"C:\Users\colin\Downloads\archive(3)\clickbait_data.csv")

df = original_dataset.copy()

# =============================================================================
# #OPTIONNEL : Ajouter un titre custom pour le prétraiter
# custom_title_df = pd.DataFrame([{'headline': 'What You Should Know About Zodiac Signs', 'clickbait': 0}])
# df = pd.concat([df, custom_title_df], ignore_index=True)
# =============================================================================

# Calculer la longueur du titre
df['title_length'] = df['headline'].apply(len)

# Fixer les contractions
df['headline_fixed'] = df['headline'].apply(lambda x: contractions.fix(x))

# Remplacer tous les nombres par 'NUMBER_TAG'
df['headline_fixed'] = df['headline_fixed'].apply(lambda x: re.sub(r'\b\d+(?:[\.,]\d+)*\b', 'NUMBER_TAG', x))

# Calculer la part de majuscules
df['uppercase_ratio'] = df['headline_fixed'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x.replace(" ", "")))

# Calculer la part de chaque type de ponctuation majeur
df['exclamation_mark_ratio'] = df['headline_fixed'].apply(lambda x: x.count('!') / len(x))
df['question_mark_ratio'] = df['headline_fixed'].apply(lambda x: x.count('?') / len(x))
df['comma_ratio'] = df['headline_fixed'].apply(lambda x: x.count(',') / len(x))
df['period_ratio'] = df['headline_fixed'].apply(lambda x: x.count('.') / len(x))
df['quotes_ratio'] = df['headline_fixed'].apply(lambda x: (x.count('"') + x.count("'")) / len(x))

def pos_tag_ratios(text):
    tags = pos_tag(word_tokenize(text))
    counts = nltk.Counter(tag for word, tag in tags)
    total = sum(counts.values())
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    adjective_tags = ['JJ', 'JJR', 'JJS']
    adverb_tags = ['RB', 'RBR', 'RBS']
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    pronoun_tags = ['PRP', 'PRP$', 'WP', 'WP$']
    superlative_tags = ['JJS', 'RBS']

    # Calculer les ratios
    adjective_ratio = sum(counts[tag] for tag in adjective_tags) / total
    adverb_ratio = sum(counts[tag] for tag in adverb_tags) / total
    noun_ratio = sum(counts[tag] for tag in noun_tags) / total
    verb_ratio = sum(counts[tag] for tag in verb_tags) / total
    pronoun_ratio = sum(counts[tag] for tag in pronoun_tags) / total
    superlative_ratio = sum(counts[tag] for tag in superlative_tags) / total

    return {
        'adjective_ratio': adjective_ratio,
        'adverb_ratio': adverb_ratio,
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'pronoun_ratio': pronoun_ratio,
        'superlative_ratio': superlative_ratio
    }


# Appliquer la fonction à chaque titre
pos_ratios = df['headline_fixed'].apply(pos_tag_ratios)

# Créer des colonnes séparées pour chaque ratio
df = pd.concat([df, pos_ratios.apply(pd.Series)], axis=1)

# Analyse de sentiment avec VADER
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline_fixed'].apply(lambda x: sia.polarity_scores(x)['compound'])

#--------

#Enlever ponctuation et normaliser
def clean_text(text):
    # Liste des ponctuations à enlever, y compris les guillemets spéciaux
    punctuation_extended = string.punctuation + "''" + '``' + '“' + '”' + '‘' + '’'
    # Enlever la ponctuation
    text_nopunct = "".join([char for char in text if char not in punctuation_extended])
    # Normaliser le texte en minuscules
    text_normalized = text_nopunct.lower()
    return text_normalized

# Appliquer la fonction de nettoyage à chaque titre
df['headline_clean'] = df['headline_fixed'].apply(clean_text)

# Créer un bag of words pour chaque titre
df['bag_of_words'] = df['headline_clean'].apply(lambda x: Counter(x.split()))

# Calculer la longueur moyenne des mots pour chaque titre
def average_word_length(bag_of_words):
    total_length = sum(len(word) * freq for word, freq in bag_of_words.items())
    total_words = sum(bag_of_words.values())
    return total_length / total_words if total_words > 0 else 0

df['average_word_length'] = df['bag_of_words'].apply(average_word_length)

#--------

# Définir les stopwords en anglais
stop_words = set(stopwords.words('english'))

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Fonction pour enlever les stopwords et lemmatiser le texte
def clean_and_lemmatize(text):
    # Lemmatiser les mots
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    # Enlever les stopwords après lemmatisation
    non_stop_words = [word for word in lemmatized_words if word not in stop_words]
    return ' '.join(non_stop_words)

# Appliquer la fonction aux titres nettoyés
df['headline_lemmatized'] = df['headline_clean'].apply(clean_and_lemmatize)

# Créer un nouveau bag of words avec les titres lemmatisés
df['bag_of_words_lemmatized'] = df['headline_lemmatized'].apply(lambda x: Counter(x.split()))


#%%
# Séparer le DataFrame en deux groupes
non_clickbait_df = df[df['clickbait'] == 0]
clickbait_df = df[df['clickbait'] == 1]

# Calculer la moyenne de chaque indicateur pour les deux groupes
indicators = ['title_length', 'uppercase_ratio', 'exclamation_mark_ratio', 'question_mark_ratio',
              'comma_ratio', 'period_ratio', 'quotes_ratio', 'adjective_ratio', 'adverb_ratio',
              'pronoun_ratio', 'superlative_ratio', 'sentiment','average_word_length']

# Créer un DataFrame pour stocker les moyennes
mean_indicators = pd.DataFrame(index=indicators, columns=['non_clickbait_mean', 'clickbait_mean'])

# Remplir le DataFrame avec les moyennes
for indicator in indicators:
    mean_indicators.loc[indicator, 'non_clickbait_mean'] = non_clickbait_df[indicator].mean()
    mean_indicators.loc[indicator, 'clickbait_mean'] = clickbait_df[indicator].mean()

print(mean_indicators)

#--------

# Fonction pour obtenir les bigrams les plus communs
def get_most_common_bigrams(texts, n=10):
    bigrams_list = [bigram for text in texts for bigram in bigrams(text.split())]
    bigram_counts = Counter(bigrams_list)
    return bigram_counts.most_common(n)

# Fonction pour obtenir les trigrams les plus communs
def get_most_common_trigrams(texts, n=10):
    trigrams_list = [trigram for text in texts for trigram in trigrams(text.split())]
    trigram_counts = Counter(trigrams_list)
    return trigram_counts.most_common(n)

# Fonction pour compter les mots uniques
def count_unique_words(texts):
    return len(set(word for text in texts for word in text.split()))

# Appliquer les fonctions aux jeux de données clickbait et non-clickbait
clickbait_texts = clickbait_df['headline_clean']
non_clickbait_texts = non_clickbait_df['headline_clean']

clickbait_bigrams = get_most_common_bigrams(clickbait_texts)
clickbait_trigrams = get_most_common_trigrams(clickbait_texts)
clickbait_unique_words = count_unique_words(clickbait_texts)

non_clickbait_bigrams = get_most_common_bigrams(non_clickbait_texts)
non_clickbait_trigrams = get_most_common_trigrams(non_clickbait_texts)
non_clickbait_unique_words = count_unique_words(non_clickbait_texts)

# Afficher les résultats
print("Clickbait Bigrams:", clickbait_bigrams)
print("Clickbait Trigrams:", clickbait_trigrams)

print("Non-Clickbait Bigrams:", non_clickbait_bigrams)
print("Non-Clickbait Trigrams:", non_clickbait_trigrams)

print("Clickbait Unique Words:", clickbait_unique_words)
print("Non-Clickbait Unique Words:", non_clickbait_unique_words)

#--------


# Recalculer les bigrams et trigrams pour les jeux de données clickbait et non-clickbait
clickbait_texts_lemmatized = clickbait_df['headline_lemmatized']
non_clickbait_texts_lemmatized = non_clickbait_df['headline_lemmatized']

clickbait_bigrams_lemmatized = get_most_common_bigrams(clickbait_texts_lemmatized)
clickbait_trigrams_lemmatized = get_most_common_trigrams(clickbait_texts_lemmatized)

non_clickbait_bigrams_lemmatized = get_most_common_bigrams(non_clickbait_texts_lemmatized)
non_clickbait_trigrams_lemmatized = get_most_common_trigrams(non_clickbait_texts_lemmatized)

print("Clickbait Bigrams (Lemmatized):", clickbait_bigrams_lemmatized)
print("Clickbait Trigrams (Lemmatized):", clickbait_trigrams_lemmatized)

print("Non-Clickbait Bigrams (Lemmatized):", non_clickbait_bigrams_lemmatized)
print("Non-Clickbait Trigrams (Lemmatized):", non_clickbait_trigrams_lemmatized)

# Fonction pour générer un nuage de mots
def generate_wordcloud(texts, title):
    wordcloud = WordCloud(collocations=False, width=800, height=400, background_color='white').generate(' '.join(texts))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
# Générer et afficher le nuage de mots pour les jeux de données clickbait et non-clickbait
generate_wordcloud(clickbait_texts_lemmatized, 'Clickbait')
generate_wordcloud(non_clickbait_texts_lemmatized, 'Traditionnel')

# Créer une matrice de termes et de documents pour l'ensemble du corpus
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['headline_lemmatized'])
y = df['clickbait']

# Appliquer le test du chi-carré
chi2_scores, p_values = chi2(X, y)

# Associer les scores chi2 avec les noms de fonctionnalités (mots)
words = np.array(vectorizer.get_feature_names_out())
# Trier les indices par scores chi2 décroissants
indices = np.argsort(chi2_scores)[::-1]
# Prendre les 100 mots les plus discriminants
discriminating_words = words[indices][:100]

# Sélectionner les 20 mots ayant les scores les plus élevés
top_words = discriminating_words[:20]
top_scores = chi2_scores[indices][:20]

# Graphique pour le chi-carré
# Créer un graphique en barres pour visualiser les scores
plt.figure(figsize=(8, 6))  # Adjust the figure size if necessary
plt.barh(range(20), top_scores, align='center', color='blue')
plt.yticks(range(20), top_words)
plt.xlabel('Score Chi-carré')
plt.title('Top 20 des mots les plus discriminants')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Sélectionner les colonnes correspondant aux 100 mots les plus discriminants
X_discriminating = X[:, indices[:100]]

# Convertir la matrice creuse en DataFrame avec les mots comme noms de colonnes
X_discriminating_df = pd.DataFrame.sparse.from_spmatrix(X_discriminating, columns=discriminating_words)

# Afficher les mots les plus discriminants
print("200 mots les plus discriminants pour déterminer le clickbait (triés):")
print(discriminating_words)

# =============================================================================
# #OPTIONNEL : Sauvegarder le titre custom traité, puis l'enlever
# custom_row_index = df.index[-1]
# custom_title_df = df.loc[[custom_row_index]].copy()
# custom_title_df_discrimination = X_discriminating_df.loc[[custom_row_index]].copy()
# custom_title_combined = pd.concat([custom_title_df_discrimination, custom_title_df], axis=1)
# custom_title_combined = custom_title_combined.select_dtypes(include=[np.number])
# custom_title_combined = custom_title_combined.drop(['clickbait', 'noun_ratio','verb_ratio'], axis=1)
# 
# df = df.drop(index=custom_row_index)
# X_discriminating_df = X_discriminating_df.drop(index=custom_row_index)
# =============================================================================


#%%
# Fonction pour trouver le meilleur seuil de classification
def find_best_threshold(model, X_test, y_test, beta=1):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    fscore = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    fscore = np.nan_to_num(fscore)
    ix = np.argmax(fscore)
    return thresholds[ix]

# Fonction pour calculer les métriques avec un seuil personnalisé
def calculate_metrics(model, X, y, threshold, beta=1):
    probs = model.predict_proba(X)[:, 1]
    predictions = (probs >= threshold).astype(int)
    accuracy = accuracy_score(y, predictions)
    recall = recall_score(y, predictions)
    fbeta = fbeta_score(y, predictions, beta=beta)
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    fpr = fp / (fp + tn)
    return accuracy, recall, fbeta, fpr

#--------

X = df[indicators]
y = df['clickbait']

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

# Entraînement des modèles
rf_model = RandomForestClassifier(n_estimators=500,random_state=123)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=2000,random_state=123)
lr_model.fit(X_train, y_train)

# Optimisation du seuil pour la forêt aléatoire
threshold_rf = find_best_threshold(rf_model, X_train, y_train, beta=0.5)

# Optimisation du seuil pour la régression logistique
threshold_lr = find_best_threshold(lr_model, X_train, y_train, beta=0.5)

# Calcul des métriques pour la forêt aléatoire sur l'ensemble de test
accuracy_rf, recall_rf, fbeta_rf, fpr_rf = calculate_metrics(rf_model, X_test, y_test, threshold_rf, beta=0.5)

# Calcul des métriques pour la régression logistique sur l'ensemble de test
accuracy_lr, recall_lr, fbeta_lr, fpr_lr = calculate_metrics(lr_model, X_test, y_test, threshold_lr, beta=0.5)

# Affichage des résultats
print(f"Random Forest - Accuracy: {accuracy_rf}, Recall: {recall_rf}, F-beta: {fbeta_rf}, FPR: {fpr_rf}")
print(f"Logistic Regression - Accuracy: {accuracy_lr}, Recall: {recall_lr}, F-beta: {fbeta_lr}, FPR: {fpr_lr}")

#--------

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_discriminating_df, y, test_size=0.2, stratify=y, random_state=123)

# Entraînement de la régression logistique avec les 100 mots les plus discriminants
lr_model2 = LogisticRegression(max_iter=2000,random_state=123)
lr_model2.fit(X_train, y_train)

# Entraînement de la forêt aléatoire avec les 100 mots les plus discriminants
rf_model2 = RandomForestClassifier(n_estimators=500,random_state=123)
rf_model2.fit(X_train, y_train)


# Optimisation du seuil pour la forêt aléatoire
threshold_rf = find_best_threshold(rf_model2, X_train, y_train, beta=0.5)

# Optimisation du seuil pour la régression logistique
threshold_lr = find_best_threshold(lr_model2, X_train, y_train, beta=0.5)

# Calcul des métriques pour la forêt aléatoire sur l'ensemble de test
accuracy_rf, recall_rf, fbeta_rf, fpr_rf = calculate_metrics(rf_model2, X_test, y_test, threshold_rf, beta=0.5)

# Calcul des métriques pour la régression logistique sur l'ensemble de test
accuracy_lr, recall_lr, fbeta_lr, fpr_lr = calculate_metrics(lr_model2, X_test, y_test, threshold_lr, beta=0.5)

# Affichage des résultats
print(f"Random Forest with Discriminating Words - Accuracy: {accuracy_rf}, "
      f"Recall: {recall_rf}, F-beta: {fbeta_rf}, FPR: {fpr_rf}")
print(f"Logistic Regression with Discriminating Words - Accuracy: {accuracy_lr}, "
      f"Recall: {recall_lr}, F-beta: {fbeta_lr}, FPR: {fpr_lr}")

#--------
# Réinitialiser l'index de df[indicators] pour s'assurer qu'il correspond à celui de X_discriminating_df
indicators_df = df[indicators].reset_index(drop=True)

# Convertir X_discriminating_df en un DataFrame dense
X_discriminating_df = X_discriminating_df.sparse.to_dense()

# Convertir les noms de colonnes de X_discriminating_df en chaînes de caractères
X_discriminating_df.columns = X_discriminating_df.columns.astype(str)

# Fusion des mots discriminants et des indicateurs
X_combined = pd.concat([X_discriminating_df, indicators_df], axis=1)

# Séparation en ensembles d'entraînement et de test
X_train_combined, X_test_combined, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=123)

# Entraînement de la régression logistique avec les mots et les indicateurs combinés
lr_model_combined = LogisticRegression(max_iter=2000,random_state=123)
lr_model_combined.fit(X_train_combined, y_train)

# Entraînement de la forêt aléatoire avec les mots et les indicateurs combinés
rf_model_combined = RandomForestClassifier(n_estimators=500, random_state=123)
rf_model_combined.fit(X_train_combined, y_train)

# Optimisation du seuil pour la forêt aléatoire combinée
threshold_rf_combined = find_best_threshold(rf_model_combined, X_train_combined, y_train, beta=0.5)

# Optimisation du seuil pour la régression logistique combinée
threshold_lr_combined = find_best_threshold(lr_model_combined, X_train_combined, y_train, beta=0.5)

# Calcul des métriques pour la forêt aléatoire combinée sur l'ensemble de test
accuracy_rf_combined, recall_rf_combined, fbeta_rf_combined, fpr_rf_combined = calculate_metrics(
    rf_model_combined, X_test_combined, y_test, threshold_rf_combined, beta=0.5)

# Calcul des métriques pour la régression logistique combinée sur l'ensemble de test
accuracy_lr_combined, recall_lr_combined, fbeta_lr_combined, fpr_lr_combined = calculate_metrics(
    lr_model_combined, X_test_combined, y_test, threshold_lr_combined, beta=0.5)

# Affichage des résultats
print(f"Combined Random Forest - Accuracy: {accuracy_rf_combined}, "
      f"Recall: {recall_rf_combined}, F-beta: {fbeta_rf_combined}, FPR: {fpr_rf_combined}")
print(f"Combined Logistic Regression - Accuracy: {accuracy_lr_combined}, "
      f"Recall: {recall_lr_combined}, F-beta: {fbeta_lr_combined}, FPR: {fpr_lr_combined}")

# Afficher les coefficients de la régression logistique
lr_coefficients_combined = lr_model_combined.coef_[0]
lr_coefficients_combined_df = pd.DataFrame({'Feature': X_combined.columns, 'Coefficient': lr_coefficients_combined})
lr_coefficients_combined_df = lr_coefficients_combined_df.reindex(lr_coefficients_combined_df['Coefficient'].abs().sort_values(ascending=False).index)

# =============================================================================
# #OPTIONNEL : Prédire titre custom
# predicted_class_custom = rf_model_combined.predict(custom_title_combined)
# print(predicted_class_custom)
# =============================================================================

