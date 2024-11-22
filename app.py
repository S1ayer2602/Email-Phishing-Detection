import streamlit as st
import pickle
import re

# Load all models and vectorizer
with open("Models\\naive_bayes_model.pkl", "rb") as nb_file:
    nb_model = pickle.load(nb_file)

with open("Models\\logistic-regression.pkl", "rb") as lr_file:
    lr_model = pickle.load(lr_file)

with open("Models\\SGD.pkl", "rb") as svm_file:
    svm_model = pickle.load(svm_file)

with open("Models\\tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)          # Remove hyperlinks
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuations
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces
    return text

# Prediction function
def predict_email(email_text, model):
    # Preprocess the email text
    email_text = preprocess_text(email_text)

    # Vectorize the text
    email_vector = vectorizer.transform([email_text])

    # Make prediction
    prediction = model.predict(email_vector)

    # Interpret the result
    return "Phishing Mail" if prediction[0] == 0 else "Safe Mail"

# Streamlit App
def main():
    st.title("Email Classification App")
    st.write("Classify emails as 'Phishing' or 'Safe' using different ML models.")

    # Model selection
    model_choice = st.selectbox("Choose a model:", ["Naive Bayes", "Logistic Regression", "SVM"])

    # Input email text
    email_text = st.text_area("Enter the email content:")

    if st.button("Classify"):
        if email_text.strip() == "":
            st.warning("Please enter the email content.")
        else:
            # Select the model based on user choice
            if model_choice == "Naive Bayes":
                model = nb_model
            elif model_choice == "Logistic Regression":
                model = lr_model
            elif model_choice == "SVM":
                model = svm_model
            else:
                st.error("Invalid model selection.")
                return

            # Make prediction
            result = predict_email(email_text, model)
            st.success(f"The email is classified as: {result}")

if __name__ == "__main__":
    main()


