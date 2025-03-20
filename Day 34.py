from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample input corpus
corpus = [
    "I love machine learning",
    "Machine learning is fun",
    "Deep learning is amazing"
]

# Bag of Words (BoW) Representation
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(corpus)

print("Bag of Words Representation:")
print(vectorizer.get_feature_names_out())
print(X_bow.toarray())

# TF-IDF Representation
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

print("\nTF-IDF Representation:")
print(tfidf_vectorizer.get_feature_names_out())
print(X_tfidf.toarray())