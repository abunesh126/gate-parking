"""Inspect TensorFlow checkpoint variables and shapes."""

from __future__ import annotations

import json
import sys

import tensorflow as tf

from validation_common import CHECKPOINT, build_documented_model


def main() -> int:
    variables = [
        {"name": name, "shape": list(shape)}
        for name, shape in tf.train.list_variables(str(CHECKPOINT))
    ]
    model = build_documented_model()
    result = {
        "checkpoint_prefix": str(CHECKPOINT),
        "index_exists": CHECKPOINT.with_suffix(".index").is_file(),
        "data_exists": CHECKPOINT.with_name(
            CHECKPOINT.name + ".data-00000-of-00001"
        ).is_file(),
        "variables": variables,
        "model_weights": [
            {"name": weight.name, "shape": weight.shape.as_list()}
            for weight in model.weights
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
