import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Create dummy data for the model
# Features: MFCC features (40 dimensions)
X_train = np.random.rand(100, 40)

# Labels: Emotion codes (01-08)
emotions = ['01', '02', '03', '04', '05', '06', '07', '08']
y_train = np.random.choice(emotions, size=100)

print("Creating a dummy model for demonstration purposes...")

# Create and train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Create directory if it doesn't exist
os.makedirs('Models', exist_ok=True)

# Save the model
model_path = 'Models/emotion_model.pkl'
joblib.dump(model, model_path)
print(f"Dummy model saved to {model_path}")
print("This is just for demonstration. In a real scenario, you'd need to use the RAVDESS dataset.") 