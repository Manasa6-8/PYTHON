import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Define the prompt
prompt = "Artificial Intelligence is..."

# Generate text completion
response = openai.ChatCompletion.create(
    model="text-davinci-003",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=50
)

# Print the generated completion
print("Generated Text:", response["choices"][0]["message"]["content"].strip())