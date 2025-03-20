from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "Transformers are revolutionizing NLP."

# Tokenize sentence
tokens = tokenizer.tokenize(sentence)
input_ids = tokenizer.encode(sentence, add_special_tokens=True)

# Print tokenized output
print("Tokenized Sentence:", tokens)
print("Input IDs:", input_ids)