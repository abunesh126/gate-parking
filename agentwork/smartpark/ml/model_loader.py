"""Load the reference TensorFlow checkpoint into the documented CNN."""

from pathlib import Path

from .model import build_model


def load_checkpoint_model(checkpoint_prefix: Path):
    if not checkpoint_prefix.with_suffix(".index").is_file():
        raise FileNotFoundError(f"Checkpoint index not found: {checkpoint_prefix}.index")
    data_file = checkpoint_prefix.with_name(
        checkpoint_prefix.name + ".data-00000-of-00001"
    )
    if not data_file.is_file():
        raise FileNotFoundError(f"Checkpoint data not found: {data_file}")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for CNN recognition") from exc

    model = build_model()
    reader = tf.train.load_checkpoint(str(checkpoint_prefix))
    restored = 0
    weighted_layers = [layer for layer in model.layers if layer.weights]
    for layer_index, layer in enumerate(weighted_layers):
        for weight in layer.weights:
            kind = "kernel" if "kernel" in weight.name.lower() else "bias"
            name = (
                f"layer_with_weights-{layer_index}/{kind}/"
                ".ATTRIBUTES/VARIABLE_VALUE"
            )
            value = reader.get_tensor(name)
            if tuple(value.shape) != tuple(weight.shape):
                raise ValueError(
                    f"Checkpoint shape mismatch for {weight.name}: "
                    f"{value.shape} != {tuple(weight.shape)}"
                )
            weight.assign(value)
            restored += 1
    if restored != len(model.weights):
        raise ValueError(
            f"Checkpoint restored {restored} of {len(model.weights)} model tensors"
        )
    return model
