from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Training examples
training_sentences = [
    "I forgot my password", "reset my password", "how to change password",
    "I have a billing issue", "payment problem", "invoice not received",
    "my app is not working", "getting an error", "technical issue in my account",
    "I want to close my account", "delete my account"
]

training_labels = [
    "password_reset", "password_reset", "password_reset",
    "billing", "billing", "billing",
    "technical_support", "technical_support", "technical_support",
    "account_closure", "account_closure"
]

# Responses for each intent
responses = {
    'password_reset': "Please click on 'Forgot Password' on the login page to reset your password.",
    'billing': "For billing issues, please check your payment history or contact billing@support.com.",
    'technical_support': "Our technical team is looking into this. Meanwhile, please try restarting the app.",
    'account_closure': "We’re sorry to see you go! Please fill the account closure form here: [Close Account](#).",
    'default': "I'm sorry, I couldn't understand your request. Please rephrase or contact support."
}

# Train the model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
model = LogisticRegression()
model.fit(X_train, training_labels)

# Show instructions
print("Welcome to Customer Support Chatbot!")
print("You can ask things like: 'I forgot my password', 'I have a billing issue', 'There's an error in my app', 'I want to close my account'.")

# Take input
user_input = input("\nHow can I help you today? ")

# Predict
X_test = vectorizer.transform([user_input])
predicted_intent = model.predict(X_test)[0]
response = responses.get(predicted_intent, responses['default'])

print(f"\nBot Response: {response}")
