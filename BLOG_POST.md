# Why I Built mlship: One Tool for All ML Frameworks

If you're learning machine learning, you've probably noticed something frustrating: every framework has its own way to serve models.

Trained a scikit-learn model? Use Flask or FastAPI and write your own server code. Built a PyTorch model? Maybe try TorchServe (if you can figure out the configuration). TensorFlow? TF Serving with Docker. HuggingFace? There's transformers-serve, but it only works for transformer models.

For students and data scientists who work across frameworks, this fragmentation is exhausting. You spend more time learning deployment tools than actually deploying models.

## What if one command worked for everything?

That's why I built **mlship**. It's a zero-configuration CLI that turns any ML model into a REST API with a single command:

```bash
mlship serve model.pkl
```

That's it. No Docker. No YAML. No framework-specific configuration. It works for:
- scikit-learn (`.pkl`, `.joblib`)
- PyTorch (`.pt`, `.pth` with TorchScript)
- TensorFlow (`.h5`, `.keras`, SavedModel)
- HuggingFace models (local or directly from the Hub)

## Example: Serving a HuggingFace model from the Hub

You don't even need to download the model file:

```bash
# Install
pip install mlship[huggingface]

# Serve a sentiment analysis model
mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface

# Test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This product is amazing!"}'
```

You get an auto-generated REST API with:
- `/predict` - Make predictions
- `/health` - Health check
- `/info` - Model metadata
- `/docs` - Interactive Swagger UI

## Why mlship matters

**For students:** Learn model serving concepts once, use them across your entire ML curriculum. Stop wrestling with framework-specific tools when you should be learning ML.

**For data scientists:** Prototype locally without Docker or cloud setup. Test your models with realistic API interactions before investing in production infrastructure.

**For educators:** Teach framework-agnostic concepts. Your students can focus on ML fundamentals instead of deployment tooling.

## What's different?

Unlike BentoML (requires Python code), TorchServe (PyTorch only), TF Serving (TensorFlow only), or transformers-serve (HuggingFace only), **mlship is the only zero-code tool that supports all major frameworks**.

It's deliberately simple:
- No configuration files
- No custom Python code required
- Works offline after installation
- Local-first (no cloud dependency)

Think of it as "one tool for your entire ML journey" - from your first scikit-learn classifier to production-grade transformers.

## Try it yourself

```bash
pip install mlship
mlship serve your_model.pkl
```

Full examples in the [Quick Start Guide](https://github.com/sudhanvalabs/mlship/blob/main/QUICKSTART.md).

## We need your help

mlship is far from perfect - it's a young project with rough edges. But that's exactly why we need your help to make it better.

We're looking for contributors:
- Support for more frameworks (XGBoost, LightGBM)
- More HuggingFace task types (Q&A, translation)
- GPU support
- Bug fixes and improvements
- Documentation improvements
- Your ideas!

**Found a bug?** Please [open an issue](https://github.com/sudhanvalabs/mlship/issues) - we want to fix it.

**Have a feature idea?** Open an issue and let's discuss it.

**Want to contribute code?** Check out [CONTRIBUTING.md](https://github.com/sudhanvalabs/mlship/blob/main/CONTRIBUTING.md) to get started.

Read the comparison with other tools: [WHY_MLSHIP.md](https://github.com/sudhanvalabs/mlship/blob/main/WHY_MLSHIP.md)

---

**GitHub:** https://github.com/sudhanvalabs/mlship
**PyPI:** https://pypi.org/project/mlship/
**License:** MIT

Let's make ML model serving simple for everyone.
