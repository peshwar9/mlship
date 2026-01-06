# mlship Quick Start Guide

Get your ML models serving in minutes.

## Installation

```bash
pip install mlship
```

That's it! No Docker, no configuration files, no setup.

---

## HuggingFace Hub Models

The fastest way to try mlship is with HuggingFace Hub models (no model files needed).

### Example 1: Sentiment Analysis

**Serve:**
```bash
mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This product is amazing!"}'
```

**Expected Response:**
```json
{
  "prediction": "POSITIVE",
  "probability": 0.9998,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

**Try negative sentiment:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This is terrible and disappointing"}'
```

**Expected Response:**
```json
{
  "prediction": "NEGATIVE",
  "probability": 0.9997,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

---

### Example 2: Text Generation (GPT-2)

**Serve:**
```bash
mlship serve gpt2 --source huggingface
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "Once upon a time"}'
```

**Expected Response:**
```json
{
  "prediction": "Once upon a time, the world was a place of great beauty...",
  "model_name": "gpt2"
}
```

**Try a different prompt:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "The future of AI is"}'
```

---

### Example 3: Question Answering

**Serve:**
```bash
mlship serve distilbert-base-cased-distilled-squad --source huggingface
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"question": "What is mlship?", "context": "mlship is a tool that turns ML models into REST APIs with one command. It supports sklearn, PyTorch, TensorFlow, and HuggingFace models."}}'
```

**Expected Response:**
```json
{
  "prediction": "a tool that turns ML models into REST APIs",
  "probability": 0.89,
  "model_name": "distilbert-base-cased-distilled-squad"
}
```

---

## Local Model Examples

### Example 4: Scikit-learn Model

**Create and train a model:**

```python
# train_sklearn_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Create sample data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'sklearn_model.pkl')
print('✅ Model saved to sklearn_model.pkl')
```

```bash
python train_sklearn_model.py
```

**Serve:**
```bash
mlship serve sklearn_model.pkl
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 2.3, -0.5, 1.2]}'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "probability": 0.87,
  "model_name": "sklearn_model"
}
```

---

### Example 5: PyTorch Model

**Create and train a model:**

```python
# train_pytorch_model.py
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

# Create and save model
model = SimpleModel()
model.eval()

# Save FULL model (not just state_dict)
torch.save(model, 'pytorch_model.pt')
print('✅ Model saved to pytorch_model.pt')
```

```bash
python train_pytorch_model.py
```

**Serve:**
```bash
mlship serve pytorch_model.pt
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

**Expected Response:**
```json
{
  "prediction": [0.234, -0.156],
  "model_name": "pytorch_model"
}
```

---

### Example 6: TensorFlow/Keras Model

**Create and train a model:**

```python
# train_tensorflow_model.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create sample data
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

# Define model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=10, verbose=0)

# Save model
model.save('tensorflow_model.h5')
print('✅ Model saved to tensorflow_model.h5')
```

```bash
python train_tensorflow_model.py
```

**Serve:**
```bash
mlship serve tensorflow_model.h5
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, -0.3, 0.8]}'
```

**Expected Response:**
```json
{
  "prediction": 0.6234,
  "model_name": "tensorflow_model"
}
```

---

## Additional Features

### Interactive API Documentation

Open your browser to `http://localhost:8000/docs` for automatic Swagger UI with:
- Try out API calls directly
- See request/response schemas
- Test different inputs

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded"
}
```

### Model Info

```bash
curl http://localhost:8000/info
```

**Response:**
```json
{
  "name": "sklearn_model",
  "framework": "sklearn",
  "type": "RandomForestClassifier",
  "input_features": 4
}
```

### Custom Port

```bash
mlship serve model.pkl --port 5000
```

### Model Name

```bash
mlship serve model.pkl --name "fraud-detector"
```

---

## Framework Support

| Framework | Install | Serve Command |
|-----------|---------|---------------|
| **scikit-learn** | `pip install mlship scikit-learn` | `mlship serve model.pkl` |
| **PyTorch** | `pip install mlship torch` | `mlship serve model.pt` |
| **TensorFlow** | `pip install mlship tensorflow` | `mlship serve model.h5` |
| **HuggingFace Hub** | `pip install mlship transformers` | `mlship serve model-id --source huggingface` |

Or install everything:
```bash
pip install mlship[all]
```

---

## Tips

1. **Start with HuggingFace Hub models** - No files needed, easiest way to try mlship
2. **Use small models first** - `distilbert-base-uncased-finetuned-sst-2-english` is great for testing (268MB)
3. **Check the docs** - `http://localhost:8000/docs` shows all endpoints and schemas
4. **Test with curl first** - Verify API works before integrating with your app
5. **Use --reload in dev** - `mlship serve model.pkl --reload` auto-restarts on code changes

---

## What's Next?

- **Production deployment?** See [CONTRIBUTING.md](CONTRIBUTING.md) for best practices
- **Why mlship?** Read [WHY_MLSHIP.md](WHY_MLSHIP.md) to understand how mlship compares to other tools
- **Custom pipelines?** Check the full [README.md](README.md) for advanced features
- **Found a bug?** Report it at [GitHub Issues](https://github.com/sudhanvalabs/mlship/issues)

---

## Troubleshooting

**"Module not found" error?**
```bash
# Make sure framework is installed
pip install transformers  # for HuggingFace
pip install torch         # for PyTorch
pip install tensorflow    # for TensorFlow
```

**Port already in use?**
```bash
mlship serve model.pkl --port 5000  # Try different port
```

**Model not loading?**
- For PyTorch: Save full model with `torch.save(model, 'model.pt')`, not state_dict
- For HuggingFace: Use `--source huggingface` flag for Hub models
- Check file exists: `ls -lh model.pkl`

---

**Happy serving!** 🚀
