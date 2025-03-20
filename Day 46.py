def rule_based_chatbot():
    responses = {
        "hello": "Hi! How can I assist you?",
        "how are you?": "I'm just a bot, but I'm doing great! How about you?",
        "bye": "Goodbye! Have a wonderful day!"
    }
    
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input in responses:
            print(f"Bot: {responses[user_input]}")
            if user_input == "bye":
                break
        else:
            print("Bot: I'm not sure how to respond to that.")

# Run the chatbot
rule_based_chatbot()
