import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from transformers import pipeline

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['return', 'risk', 'sharpe_ratio'], axis=1)
    y = data[['return', 'risk', 'sharpe_ratio']]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Scikit-learn model
def train_sklearn_model(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

# PyTorch model
class PortfolioNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PortfolioNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_pytorch_model(X_train, y_train):
    model = PortfolioNet(X_train.shape[1], 64, y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train.values))
        loss = criterion(outputs, torch.FloatTensor(y_train.values))
        loss.backward()
        optimizer.step()
    
    return model

# TensorFlow model
def train_tensorflow_model(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model

# NLP for opinion mining
def perform_sentiment_analysis(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Main function
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('portfolio_data.csv')
    
    # Train models
    sklearn_model = train_sklearn_model(X_train, y_train)
    pytorch_model = train_pytorch_model(X_train, y_train)
    tensorflow_model = train_tensorflow_model(X_train, y_train)
    
    # Make predictions
    sklearn_pred = sklearn_model.predict(X_test)
    pytorch_pred = pytorch_model(torch.FloatTensor(X_test.values)).detach().numpy()
    tensorflow_pred = tensorflow_model.predict(X_test)
    
    # Evaluate models
    for i, model_name in enumerate(['Scikit-learn', 'PyTorch', 'TensorFlow']):
        pred = [sklearn_pred, pytorch_pred, tensorflow_pred][i]
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # Perform sentiment analysis on financial news
    news_text = "NVIDIA reports record-breaking quarterly earnings, surpassing analyst expectations."
    sentiment, score = perform_sentiment_analysis(news_text)
    print(f"Sentiment: {sentiment}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
