from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

# Sample text data and their corresponding categories
corpus = [
    "Spam messages are annoying",
    "I won a lottery",
    "This is a normal message",
    "Claim your free prize now",
    "Hello, how are you?"
]
labels = ["spam", "spam", "ham", "spam", "ham"]  # Categories (spam or ham)

# Create a pipeline with TfidfVectorizer and DecisionTreeClassifier
model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())

# Train the model
model.fit(corpus, labels)

# New text to classify
new_text = ["Congratulations, you have won a gift!"]
predicted_label = model.predict(new_text)

print("Predicted Category:", predicted_label[0])
