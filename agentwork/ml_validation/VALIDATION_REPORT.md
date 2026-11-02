# SmartPark CNN Runtime Validation Report

Generated from the isolated validation environment; the original repository was not modified.

## 1. Dataset paths and structure

- Archive: `C:\Users\abune\OneDrive\Desktop\gate parking\License-Plate-Number-Detection\data.zip`
- Archive training split: `C:\Users\abune\OneDrive\Desktop\gate parking\License-Plate-Number-Detection\data.zip!data/train`
- Archive validation split: `C:\Users\abune\OneDrive\Desktop\gate parking\License-Plate-Number-Detection\data.zip!data/val`
- Extracted directories are present at `License-Plate-Number-Detection\data\train` and `License-Plate-Number-Detection\data\val`; they were inspected independently and match the archive counts.
- The notebook expects `data/train` and `data/val` after extraction (not `data/data/train`).
- Separate `checkpoints\val`: `C:\Users\abune\OneDrive\Desktop\gate parking\License-Plate-Number-Detection\checkpoints\val`; it contains 216 images and is byte-for-byte identical to the archive validation split by filename and SHA-256.

## 2. Dataset counts and image health

| Class | Train | Validation |
|---|---:|---:|
| 0 | 24 | 6 |
| 1 | 24 | 6 |
| 2 | 24 | 6 |
| 3 | 24 | 6 |
| 4 | 24 | 6 |
| 5 | 24 | 6 |
| 6 | 24 | 6 |
| 7 | 24 | 6 |
| 8 | 24 | 6 |
| 9 | 24 | 6 |
| A | 24 | 6 |
| B | 24 | 6 |
| C | 24 | 6 |
| D | 24 | 6 |
| E | 24 | 6 |
| F | 24 | 6 |
| G | 24 | 6 |
| H | 24 | 6 |
| I | 24 | 6 |
| J | 24 | 6 |
| K | 24 | 6 |
| L | 24 | 6 |
| M | 24 | 6 |
| N | 24 | 6 |
| O | 24 | 6 |
| P | 24 | 6 |
| Q | 24 | 6 |
| R | 24 | 6 |
| S | 24 | 6 |
| T | 24 | 6 |
| U | 24 | 6 |
| V | 24 | 6 |
| W | 24 | 6 |
| X | 24 | 6 |
| Y | 24 | 6 |
| Z | 24 | 6 |

- Totals: 864 training, 216 validation.
- Missing/empty classes: none in either split.
- Extracted images: all 28x28, 3-channel JPEGs; unreadable/corrupt images: 0 in both splits.
- `checkpoints\val`: 216 images, same 28x28x3 JPEG profile, no unreadable images.

## 3. Isolated environment

- Environment: `agentwork\.venv-ml-validation`
- Python: 3.12.10
- TensorFlow: 2.20.0
- Keras: 3.15.1
- NumPy: 2.2.6
- OpenCV: 4.14.0
- Installation errors: none. CPU validation was used; no CUDA setup was required.

## 4. Reconstructed CNN

The exact documented architecture was reconstructed:

```text
Input 28x28x3
Conv2D 16, 22x22, ReLU, same
Conv2D 32, 16x16, ReLU, same
Conv2D 64, 8x8, ReLU, same
Conv2D 64, 4x4, ReLU, same
MaxPooling2D 4x4
Dropout 0.4
Flatten
Dense 128, ReLU
Dense 36, softmax
```

Class order: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`.

## 5. Checkpoint inspection and restore

- Prefix: `C:\Users\abune\OneDrive\Desktop\gate parking\License-Plate-Number-Detection\checkpoints\my_checkpoint`
- TensorFlow checkpoint variables contain six weighted layers with shapes matching the four convolutional layers and two dense layers, plus Adam optimizer slot variables.
- Notebook call `model.load_weights(checkpoint_prefix)`: **failed under Keras 3.15.1** with `ValueError` because Keras 3 does not accept this extensionless TensorFlow prefix.
- Fallback: `tf.train.load_checkpoint()` was used to assign every kernel/bias tensor by checkpoint name and exact shape; 12 of 12 model tensors were restored and all changed from initialization.
- Conclusion: full CNN inference-weight restoration succeeded. Optimizer slot state was inspected but not needed for inference and was not restored into a Keras optimizer; this is not a training-resume claim.

## 6. Inference preprocessing

- Training notebook preprocessing: `ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)` and `target_size=(28,28)`.
- Documented inference: segmented binary character, resize to 28x28, replicate grayscale into three channels, reshape to `(1,28,28,3)`, and no explicit `/255` normalization.
- Validation images are already 28x28, 3-channel JPEGs. The documented character path was tested as grayscale followed by three-channel replication; raw RGB and raw BGR variants were also tested.
- Documented/raw scale result: 212/216 = 98.148148%.
- Training-scale `/255` result: 212/216 = 98.148148%.
- Grayscale raw, grayscale `/255`, RGB raw, and BGR raw all produced the same 212/216 predictions on this set. This does not remove the documented training/inference scale inconsistency.

## 7. Actual validation results

- Dataset: individual labelled characters, not full vehicle plates.
- Total: 216
- Correct: 212
- Incorrect: 4
- Character accuracy (documented/raw preprocessing): **98.148148%**
- Mean batch prediction latency: 1.3276 ms/image (CPU; measured over the prediction batch).

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| 0 | 5 | 6 | 83.33% |
| 1 | 6 | 6 | 100.00% |
| 2 | 6 | 6 | 100.00% |
| 3 | 6 | 6 | 100.00% |
| 4 | 6 | 6 | 100.00% |
| 5 | 4 | 6 | 66.67% |
| 6 | 6 | 6 | 100.00% |
| 7 | 6 | 6 | 100.00% |
| 8 | 6 | 6 | 100.00% |
| 9 | 6 | 6 | 100.00% |
| A | 6 | 6 | 100.00% |
| B | 6 | 6 | 100.00% |
| C | 6 | 6 | 100.00% |
| D | 6 | 6 | 100.00% |
| E | 6 | 6 | 100.00% |
| F | 6 | 6 | 100.00% |
| G | 6 | 6 | 100.00% |
| H | 6 | 6 | 100.00% |
| I | 6 | 6 | 100.00% |
| J | 6 | 6 | 100.00% |
| K | 6 | 6 | 100.00% |
| L | 6 | 6 | 100.00% |
| M | 6 | 6 | 100.00% |
| N | 6 | 6 | 100.00% |
| O | 5 | 6 | 83.33% |
| P | 6 | 6 | 100.00% |
| Q | 6 | 6 | 100.00% |
| R | 6 | 6 | 100.00% |
| S | 6 | 6 | 100.00% |
| T | 6 | 6 | 100.00% |
| U | 6 | 6 | 100.00% |
| V | 6 | 6 | 100.00% |
| W | 6 | 6 | 100.00% |
| X | 6 | 6 | 100.00% |
| Y | 6 | 6 | 100.00% |
| Z | 6 | 6 | 100.00% |

### Confusion matrix

Rows are actual classes and columns are predicted classes, in `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` order.

| Actual\Pred | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` | `8` | `9` | `A` | `B` | `C` | `D` | `E` | `F` | `G` | `H` | `I` | `J` | `K` | `L` | `M` | `N` | `O` | `P` | `Q` | `R` | `S` | `T` | `U` | `V` | `W` | `X` | `Y` | `Z` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0` | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `1` | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `2` | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `3` | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `4` | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `5` | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `6` | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `7` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `8` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `9` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `A` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `B` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `C` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `D` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `E` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `F` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `G` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `H` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `I` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `J` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `K` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `L` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `M` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `N` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `O` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `P` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Q` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `R` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `S` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `T` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| `U` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| `V` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| `W` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| `X` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| `Y` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| `Z` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |

Non-diagonal errors: `0 -> O` (1), `O -> 0` (1), `5 -> S` (2).

## 8. Limitations

- This measures character classification only. It does not measure contour/YOLO plate localization or complete plate-string recognition.
- The archive and `checkpoints\val` are the same validation images, so they are not independent datasets.
- The notebook's legacy `predict_classes()` API and prefix-based Keras loading are not directly supported by current Keras 3; validation used `argmax(model.predict(...))` and the verified TensorFlow checkpoint reader fallback.
- No retraining or SmartPark application/frontend work was started.

## 9. Reproduction commands

From `C:\Users\abune\OneDrive\Desktop\gate parking\agentwork`:

```powershell
.venv-ml-validation\Scripts\python.exe -m pip install -r ml_validation\requirements-ml-validation.txt
Set-Location ml_validation
..\.venv-ml-validation\Scripts\python.exe inspect_dataset.py
..\.venv-ml-validation\Scripts\python.exe inspect_checkpoint.py
..\.venv-ml-validation\Scripts\python.exe test_checkpoint_restore.py
..\.venv-ml-validation\Scripts\python.exe evaluate_cnn.py
```
