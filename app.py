import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset (assuming df is your DataFrame)
df = pd.read_csv('dsa_scores_dataset.csv')  # Replace with your actual dataset filename

# Function to preprocess data
def preprocess_data(df):
    X = df[['basics', 'stl', 'sorting', 'searching', 'graphs', 'trees', 'dynamic programming', 'number theory']] / 100
    y = df['dsa_score'] / 100
    return X, y

# Function to train and save model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.joblib')
    return model, X_test, y_test

# Function to load model
def load_model():
    return joblib.load('model.joblib')

# Function to make predictions
def make_predictions(model, new_data):
    feature_names = ['basics', 'stl', 'sorting', 'searching', 'graphs', 'trees', 'dynamic programming', 'number theory']
    new_data_processed = np.array(new_data).reshape(1, -1) / 100
    X_pred = pd.DataFrame(new_data_processed, columns=feature_names)
    predictions = model.predict(X_pred)
    return predictions * 100  # Convert back to original scale

# Define the Streamlit app
def main():
    st.title('DSA Score Prediction App')

    # Top navigation bar
    nav_choice = st.radio("Navigation", ["Home", "Return"])

    if nav_choice == "Home":
        st.write('## Home')
        st.write('### Predict DSA Score')

        st.write('Enter the values for different algorithms (out of 100):')
        basics = st.slider('Basics', 0, 100, 50)
        stl = st.slider('STL', 0, 100, 50)
        sorting = st.slider('Sorting', 0, 100, 50)
        searching = st.slider('Searching', 0, 100, 50)
        graphs = st.slider('Graphs', 0, 100, 50)
        trees = st.slider('Trees', 0, 100, 50)
        dp = st.slider('Dynamic Programming', 0, 100, 50)
        number_theory = st.slider('Number Theory', 0, 100, 50)

        new_data = [basics, stl, sorting, searching, graphs, trees, dp, number_theory]

        if st.button('Predict'):
            model = load_model()
            predictions = make_predictions(model, new_data)

            st.write('### Predictions:')
            st.write(f"Predicted DSA score: {predictions[0]}")

    elif nav_choice == "Return":
        st.write('## Return')
        st.write('### Return to main page - [CTC Sure](https://ctc-sure.vercel.app/)')
        st.write('This app predicts DSA scores based on manually input algorithm metrics.')

# Run the app
if __name__ == '__main__':
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train and save model
    trained_model, _, _ = train_model(X, y)
    
    # Run the app
    main()
