from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample text data and their corresponding categories
corpus = [
    "I love programming",
    "Python is great",
    "Machine learning is amazing",
    "I enjoy coding in Python",
    "Deep learning is powerful"
]
labels = ["programming", "python", "ml", "python", "ml"]  # Corresponding categories

# Create a pipeline with CountVectorizer and MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(corpus, labels)

# New text to classify
new_text = ["Python makes coding easy"]
predicted_label = model.predict(new_text)

print("Predicted Category:", predicted_label[0])