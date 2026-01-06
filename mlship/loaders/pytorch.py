"""PyTorch model loader."""

from pathlib import Path
from typing import Any, Dict, List, Union

from mlship.errors import ModelLoadError, ValidationError
from mlship.loaders.base import ModelLoader


class PyTorchLoader(ModelLoader):
    """Loader for PyTorch models."""

    def load(self, model_path: Union[Path, str]) -> Any:
        """Load PyTorch model from .pt/.pth file."""
        # Convert to Path if string
        if isinstance(model_path, str):
            model_path = Path(model_path)

        try:
            import torch

            # Try loading as TorchScript first (recommended format)
            try:
                model = torch.jit.load(model_path, map_location="cpu")
                model.eval()
                return model
            except Exception:
                # Not a TorchScript model, try regular pickle format
                pass

            # Load model (CPU only for now)
            # weights_only=False is needed for custom model classes
            model = torch.load(model_path, map_location="cpu", weights_only=False)

            # Handle different save formats
            if isinstance(model, dict) and "model_state_dict" in model:
                # If saved as state dict, we need the model architecture
                # This is a limitation - user needs to provide full model
                raise ModelLoadError(
                    "Model saved as state_dict only. mlship needs the full model.\n\n"
                    "Recommended: Use TorchScript format (works with custom models):\n"
                    "  scripted = torch.jit.script(model)\n"
                    "  torch.jit.save(scripted, 'model.pt')\n\n"
                    "Or save the full model (may have import issues):\n"
                    "  torch.save(model, 'model.pt')"
                )

            # Set to eval mode
            if hasattr(model, "eval"):
                model.eval()

            return model

        except ImportError:
            raise ModelLoadError(
                "PyTorch is not installed.\n\n"
                "Install with: pip install torch\n"
                "Or: pip install mlship[pytorch]"
            )
        except Exception as e:
            error_msg = str(e)

            # Check for common pickling errors
            if "Can't get attribute" in error_msg or "No module named" in error_msg:
                raise ModelLoadError(
                    f"Failed to load PyTorch model: {e}\n\n"
                    "This error usually means the model was saved with torch.save() from a script.\n"
                    "Custom model classes can't be loaded this way.\n\n"
                    "✅ RECOMMENDED FIX - Use TorchScript:\n"
                    "  scripted = torch.jit.script(model)\n"
                    "  torch.jit.save(scripted, 'model.pt')\n\n"
                    "Or use torch.jit.trace() if your model isn't compatible with script:\n"
                    "  example_input = torch.randn(1, input_size)\n"
                    "  traced = torch.jit.trace(model, example_input)\n"
                    "  torch.jit.save(traced, 'model.pt')"
                )

            raise ModelLoadError(f"Failed to load PyTorch model: {e}")

    def predict(
        self, model: Any, features: Union[List[float], List[List[float]]]
    ) -> Dict[str, Any]:
        """Run prediction on input features."""
        import torch

        # Convert to tensor
        X = torch.tensor(features, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(X)

        # Handle different output types
        if output.dim() == 2 and output.shape[1] > 1:
            # Classification (multi-class)
            probabilities = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(output, dim=1)[0]

            return {
                "prediction": int(prediction.item()),
                "probability": float(probabilities.max().item()),
            }
        elif output.dim() == 2 and output.shape[1] == 1:
            # Regression or binary classification
            prediction = output[0, 0]
            return {"prediction": float(prediction.item())}
        else:
            # Single value regression
            return {"prediction": float(output.item())}

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract model metadata."""
        metadata = {
            "model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "framework": "pytorch",
        }

        # Try to infer output type from model class name
        model_name = model.__class__.__name__.lower()
        if "classifier" in model_name or "classification" in model_name:
            metadata["output_type"] = "classification"
        elif "regressor" in model_name or "regression" in model_name:
            metadata["output_type"] = "regression"

        return metadata

    def validate_input(self, model: Any, features: Union[List[float], List[List[float]]]) -> None:
        """Validate input shape."""
        import torch

        # Basic validation - PyTorch will raise its own errors if shape is wrong
        # We could enhance this by inspecting model architecture
        try:
            X = torch.tensor(features, dtype=torch.float32)
            if X.dim() == 1:
                X = X.unsqueeze(0)

            # Try a forward pass to catch shape errors early
            with torch.no_grad():
                _ = model(X)

        except RuntimeError as e:
            if "size mismatch" in str(e) or "shape" in str(e):
                raise ValidationError(
                    f"Invalid input shape\n\n"
                    f"Error: {e}\n\n"
                    f"PyTorch models require specific input dimensions. "
                    f"Check your model's expected input shape."
                )
            else:
                raise ValidationError(f"Input validation failed: {e}")
