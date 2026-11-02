"""Test the notebook loading method and a TensorFlow checkpoint fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tensorflow as tf

from validation_common import (
    CHECKPOINT,
    WORKSPACE,
    build_documented_model,
    restore_model_weights,
)


def main() -> int:
    result: dict[str, object] = {
        "tensorflow": tf.__version__,
        "checkpoint": str(CHECKPOINT),
        "documented_load_weights": {},
        "tf_checkpoint_restore": {},
    }

    model = build_documented_model()
    try:
        model.load_weights(str(CHECKPOINT))
        result["documented_load_weights"] = {
            "success": True,
            "message": "model.load_weights(checkpoint_prefix) succeeded",
        }
    except Exception as exc:  # noqa: BLE001 - record exact compatibility error
        result["documented_load_weights"] = {
            "success": False,
            "exception": type(exc).__name__,
            "message": str(exc),
        }

    model = build_documented_model()
    before = [weight.numpy().copy() for weight in model.weights]
    try:
        restored = restore_model_weights(model)
        after = [weight.numpy() for weight in model.weights]
        changed = sum(
            not (before_weight == after_weight).all()
            for before_weight, after_weight in zip(before, after)
        )
        result["tf_checkpoint_restore"] = {
            "success": True,
            "status": "all model kernel/bias tensors assigned via tf.train.load_checkpoint",
            "restored_model_weight_tensors": restored,
            "changed_model_weight_tensors": changed,
        }
    except Exception as exc:  # noqa: BLE001 - record exact restore result
        result["tf_checkpoint_restore"] = {
            "success": False,
            "exception": type(exc).__name__,
            "message": str(exc),
        }

    output = WORKSPACE / "ml_validation" / "checkpoint_restore.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["tf_checkpoint_restore"].get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
