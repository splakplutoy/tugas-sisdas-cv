# Casting Product Defect Detection

This project compares a custom CNN and a Teachable Machine model for binary visual inspection of casting products:

- `def_front`: defective product
- `ok_front`: non-defective product

Dataset: Kaggle `ravirajsinh45/real-life-industrial-dataset-of-casting-product`

## Notebooks

- `tugas_sisdas_cv_training.ipynb`: downloads the dataset, explores class counts, trains the custom CNN, saves the best weights, and plots training curves.
- `tugas_sisdas_cv_evaluation.ipynb`: loads the custom CNN and Teachable Machine model, evaluates both on the held-out test set, prints classification reports, plots confusion matrices, computes ROC AUC, checks thresholds, and shows error samples.

## Main Improvements

- The test set is no longer used as validation during training.
- The training set is split into training and validation subsets with `validation_split=0.2`.
- Random seeds are set for more reproducible runs.
- The custom CNN now uses `GlobalAveragePooling2D` instead of a large `Flatten` + `Dense(512)` block.
- Batch normalization, stronger augmentation, early stopping, best-weight checkpointing, and learning-rate reduction are used.
- Teachable Machine labels are mapped explicitly from `labels.txt` instead of relying on a hard-coded class inversion.
- Evaluation includes confusion matrix, classification report, ROC AUC, threshold analysis, and safer error visualization.

## Expected Files

For evaluation in Google Colab, place these files in Google Drive:

```text
/content/drive/MyDrive/Model Tugas CNN/Model Custom/best_model_kustom_qc.weights.h5
/content/drive/MyDrive/Model Tugas CNN/Model Custom/model_kustom_qc.keras
/content/drive/MyDrive/Model Tugas CNN/Model Teachable Machine/keras_model.h5
/content/drive/MyDrive/Model Tugas CNN/Model Teachable Machine/labels.txt
```

The provided Teachable Machine label file uses:

```text
0 Barang Bagus
1 Barang Cacat
```

So output index `0` maps to `ok_front`, and output index `1` maps to `def_front`.
This repository also includes `labels.txt` with that mapping.

## Workflow

1. Run `tugas_sisdas_cv_training.ipynb`.
2. Copy or save the generated custom model files to the Google Drive paths above.
3. Make sure `keras_model.h5` and `labels.txt` from Teachable Machine are in the expected Teachable Machine folder.
4. Run `tugas_sisdas_cv_evaluation.ipynb`.

## Evaluation Note

The final test set should only be used in the evaluation notebook. Do not use the test set for model selection, tuning, or early stopping.
