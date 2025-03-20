from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# Sample text to summarize
text = """
Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines 
that can mimic human cognition. It involves tasks such as learning, reasoning, problem-solving, 
and language understanding. AI applications range from voice assistants to self-driving cars, 
revolutionizing various industries.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)

# Print summarized text
print("Summarized Text:", summary[0]['summary_text'])
