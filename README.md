# mlship

**Turn any ML model into a REST API with one command.**

```bash
mlship serve model.pkl
```

Deploy your machine learning models locally in seconds—no Docker, no YAML, no configuration files.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Why mlship?** Read [WHY_MLSHIP.md](WHY_MLSHIP.md) to see how mlship compares to transformers-serve, vLLM, Ollama, and BentoML.

---

## Features

- ✅ **One-command deployment** - No configuration needed
- ✅ **Multi-framework** - sklearn, PyTorch, TensorFlow, HuggingFace (local + Hub)
- ✅ **HuggingFace Hub** - Serve models directly from Hub without downloading
- ✅ **Auto-generated API** - REST API with interactive docs
- ✅ **Works offline** - Zero internet dependency after installation
- ✅ **Fast** - Deploy in seconds, predictions in milliseconds

---

## Quick Start

```bash
# Install
pip install mlship

# Serve any model
mlship serve model.pkl
```

### Try HuggingFace Hub Models (No Files Needed!)

```bash
# Sentiment analysis
mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface

# Test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This product is amazing!"}'
```

**📖 See [QUICKSTART.md](QUICKSTART.md)** for complete hands-on examples with:
- HuggingFace models (sentiment, GPT-2, Q&A)
- Local models (sklearn, PyTorch, TensorFlow)
- Training code, curl commands, and expected responses

---

## Supported Frameworks

| Framework | File Format | Example |
|-----------|------------|---------|
| **scikit-learn** | `.pkl`, `.joblib` | `mlship serve model.pkl` |
| **PyTorch** | `.pt`, `.pth` | `mlship serve model.pt` |
| **TensorFlow** | `.h5`, `.keras`, SavedModel | `mlship serve model.h5` |
| **HuggingFace (local)** | Model directory | `mlship serve ./sentiment-model/` |
| **HuggingFace (Hub)** | Model ID | `mlship serve bert-base-uncased --source huggingface` |

---

## HuggingFace Hub Support

Serve models directly from HuggingFace Hub:

```bash
mlship serve gpt2 --source huggingface
mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface
```

Models are downloaded on first use and cached locally. See [QUICKSTART.md](QUICKSTART.md) for more examples.

---

## API Endpoints

Every model automatically gets:

- **POST `/predict`** - Make predictions
- **GET `/health`** - Health check
- **GET `/info`** - Model metadata
- **GET `/docs`** - Interactive Swagger UI documentation

Examples in [QUICKSTART.md](QUICKSTART.md).

---

## Advanced Usage

```bash
# Custom port
mlship serve model.pkl --port 5000

# Development mode (auto-reload on code changes)
mlship serve model.pkl --reload

# Custom model name
mlship serve model.pkl --name "fraud-detector"

# Custom preprocessing/postprocessing
mlship serve model.pkl --pipeline my_module.MyPipeline
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for custom pipeline documentation.

---

## Use Cases

**For Students & Learners**
- Learn model serving without framework-specific tools
- One tool works for entire ML curriculum (sklearn → PyTorch → transformers)

**For Data Scientists**
- Prototype models locally before production
- Test models with realistic API interactions
- Share models with teammates without cloud setup

**For Educators**
- Teach framework-agnostic model serving concepts
- Create reproducible examples that work across frameworks

Read [WHY_MLSHIP.md](WHY_MLSHIP.md) for detailed positioning.

---

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Hands-on getting started guide
- **[WHY_MLSHIP.md](WHY_MLSHIP.md)** - Positioning and comparisons
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical design

---

## Installation

```bash
pip install mlship
```

**With specific frameworks:**
```bash
pip install mlship[sklearn]       # scikit-learn
pip install mlship[pytorch]       # PyTorch
pip install mlship[tensorflow]    # TensorFlow
pip install mlship[huggingface]   # HuggingFace
pip install mlship[all]           # All frameworks
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Running tests
- Code style guidelines
- Custom pipeline development

---

## Support

- **Issues**: [GitHub Issues](https://github.com/prabhueshwarla/mlship/issues)
- **Documentation**: See docs linked above
- **Examples**: Check the `examples/` directory

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## What Makes mlship Different?

mlship is the **only zero-code tool** that supports sklearn, PyTorch, TensorFlow, AND HuggingFace models with a single command. Read [WHY_MLSHIP.md](WHY_MLSHIP.md) for detailed comparison with transformers-serve, vLLM, Ollama, and BentoML.

**Quick comparison:**
- ✅ Multi-framework (not just one)
- ✅ Zero code required (no Python files)
- ✅ Local-first (no cloud dependency)
- ✅ HuggingFace Hub integration
- ✅ Perfect for learning and prototyping

---

## Roadmap

**✅ Implemented:**

- ✅ **Multi-framework support** - sklearn, PyTorch, TensorFlow, HuggingFace
- ✅ **HuggingFace Hub integration** - Serve models directly from Hub with `--source huggingface`
- ✅ **Zero-code deployment** - One command to serve any model
- ✅ **Auto-generated REST API** - With interactive Swagger docs
- ✅ **Custom pipelines** - Preprocessing/postprocessing support
- ✅ **Local-first** - Works completely offline (after installation)

**🔄 Planned:**

- 🔄 **PyTorch Hub integration** - Serve models directly from PyTorch Hub with `--source pytorch-hub`
- 🔄 **TensorFlow Hub integration** - Serve models from TensorFlow Hub with `--source tensorflow-hub`
- 🔄 **XGBoost & LightGBM support** - First-class support for gradient boosting frameworks
- 🔄 **Model versioning** - Support specific model versions (e.g., `--revision main`)
- 🔄 **GPU support** - Automatic GPU detection and utilization
- 🔄 **Batch inference** - Efficient batch prediction endpoints
- 🔄 **Authentication** - Optional API key authentication for deployments

Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md) or [open an issue](https://github.com/prabhueshwarla/mlship/issues) with your ideas!

---

**Happy serving!** 🚀
