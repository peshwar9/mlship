from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Train a simple model
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save it
joblib.dump(model, 'model.pkl')
print('✅ Model saved to model.pkl')

