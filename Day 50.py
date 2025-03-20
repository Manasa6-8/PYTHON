import numpy as np

def self_attention():
    # Define dimensions
    d_k = 4  # Dimension of key/query/value vectors
    
    # Generate random query, key, and value vectors
    query = np.random.rand(d_k)
    key = np.random.rand(d_k)
    value = np.random.rand(d_k)
    
    # Compute attention scores (dot product of query and key)
    attention_score = np.dot(query, key)
    
    # Apply softmax for normalized attention weights
    attention_weight = np.exp(attention_score) / np.exp(attention_score).sum()
    
    # Compute final output as weighted sum of values
    output = attention_weight * value
    
    print("Query Vector:", query)
    print("Key Vector:", key)
    print("Value Vector:", value)
    print("Attention Score:", attention_score)
    print("Attention Weight:", attention_weight)
    print("Final Output:", output)

# Run self-attention function
self_attention()