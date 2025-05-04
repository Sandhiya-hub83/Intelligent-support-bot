# Install required packages (uncomment if running in Colab)
# !pip install transformers scikit-learn

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# Sample dataset: Expand as needed
training_data = {
    "greeting": ["hello", "hi", "good morning", "good evening", "hey there"],
    "goodbye": ["bye", "see you later", "goodbye", "take care"],
    "order_status": ["where is my order", "track my order", "order status", "has my order shipped"],
    "refund_policy": ["how to get a refund", "refund policy", "I want a refund", "return my product"],
    "complaint": ["bad service", "not happy", "want to complain", "issue with my order"]
}

responses = {
    "greeting": ["Hello! How can I help you today?", "Hi there! What can I do for you?"],
    "goodbye": ["Goodbye! Have a great day!", "Take care!"],
    "order_status": ["Please share your order ID, and I’ll check the status for you."],
    "refund_policy": ["Our refund policy is 30 days. Would you like me to initiate the process?"],
    "complaint": ["I'm really sorry to hear that. Let me escalate this to our support team."]
}

# Prepare training data
X, y = [], []
for intent, phrases in training_data.items():
    for phrase in phrases:
        X.append(phrase)
        y.append(intent)

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

classifier = LogisticRegression()
classifier.fit(X_vectorized, y)

# Sentiment analysis (optional enhancement)
sentiment_analyzer = pipeline("sentiment-analysis")

def get_intent(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    intent = classifier.predict(user_input_vectorized)[0]
    return intent

def chatbot_response(user_input):
    intent = get_intent(user_input)
    sentiment = sentiment_analyzer(user_input)[0]
    if sentiment["label"] == "NEGATIVE" and intent != "complaint":
        return "I'm sorry to hear that. Can you tell me more about the issue?"
    return random.choice(responses[intent])

# Chat loop (can be integrated into a UI)
print("Customer Support Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Thank you for chatting with us!")
        break
    print("Bot:", chatbot_response(user_input))
