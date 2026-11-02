"""The exact CNN architecture used by the reference notebooks."""

IMAGE_SIZE = (28, 28)
CLASS_NAMES = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def build_model():
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the CNN") from exc
    return Sequential(
        [
            Conv2D(16, (22, 22), input_shape=(28, 28, 3), activation="relu", padding="same"),
            Conv2D(32, (16, 16), activation="relu", padding="same"),
            Conv2D(64, (8, 8), activation="relu", padding="same"),
            Conv2D(64, (4, 4), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(4, 4)),
            Dropout(0.4),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
