from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Sample text data and their corresponding categories
corpus = [
    "Buy now and win a prize",
    "Normal email content",
    "Congratulations, you won!",
    "Exclusive deal just for you",
    "Meeting scheduled for tomorrow"
]
labels = ["spam", "ham", "spam", "spam", "ham"]  # Categories (spam or ham)

# Create a pipeline with TfidfVectorizer and RandomForestClassifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(corpus, labels)

# New text to classify
new_text = ["Limited time offer! Claim your prize now"]
predicted_label = model.predict(new_text)

print("Predicted Category:", predicted_label[0])
