def simple_chatbot():
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input == "hello":
            print("Bot: Hi! How can I help you?")
        elif user_input == "bye":
            print("Bot: Goodbye! Have a nice day!")
            break
        else:
            print("Bot: I'm not sure how to respond to that.")

# Run the chatbot
simple_chatbot()