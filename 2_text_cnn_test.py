from pathlib import Path
import importlib.util
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = BASE_DIR / "1_text_cnn_trening.py"
TEST_CSV = BASE_DIR / "firewall_logs_labeled" / "2_Test_logs_combiend_labeled.csv"
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
RESULTS_DIR = BASE_DIR / "test_results"

MODEL_PATH = ARTIFACT_DIR / "textcnn_model_trained.pth"
VOCAB_PATH = ARTIFACT_DIR / "vocab_trained.pth"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder_trained.pth"
SCALER_PATH = ARTIFACT_DIR / "numeric_scaler_trained.pth"

PREDICTIONS_CSV = RESULTS_DIR / "2_Test_logs_combiend_predictions.csv"
METRICS_TXT = RESULTS_DIR / "2_Test_metrics.txt"
CONFUSION_MATRIX_CSV = RESULTS_DIR / "2_Test_confusion_matrix.csv"

BATCH_SIZE = 64


def load_training_module():
    spec = importlib.util.spec_from_file_location("text_cnn_training", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)} min {remaining_seconds:.2f} s"


class FirewallLogTestDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, vocab, training_module):
        self.texts = texts
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.training_module = training_module

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_indices = self.training_module.encode_text(
            self.texts[index],
            self.vocab,
            self.training_module.MAX_SEQUENCE_LENGTH,
        )
        return text_indices, self.numeric_features[index], self.labels[index]


def transform_numeric_features(df: pd.DataFrame, scaler, training_module) -> np.ndarray:
    numeric_df = df[training_module.NUMERIC_COLUMNS].replace("-", 0).copy()

    for column in training_module.NUMERIC_COLUMNS:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce").fillna(0)

    numeric_df["bytes"] = np.log1p(numeric_df["bytes"])
    numeric_df["packets"] = np.log1p(numeric_df["packets"])

    return scaler.transform(numeric_df).astype(np.float32)


def prepare_test_data(csv_path: Path, vocab, label_encoder, scaler, training_module):
    df = pd.read_csv(csv_path)

    required_columns = (
        training_module.TEXT_COLUMNS
        + training_module.NUMERIC_COLUMNS
        + [training_module.LABEL_COLUMN]
    )
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Nedostaju stupci u CSV datoteci: {missing_columns}")

    for column in training_module.NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column].replace("-", 0), errors="coerce").fillna(0).astype(int)

    texts = df.apply(training_module.row_to_text, axis=1).tolist()
    labels = label_encoder.transform(df[training_module.LABEL_COLUMN])
    numeric_features = transform_numeric_features(df, scaler, training_module)

    dataset = FirewallLogTestDataset(texts, numeric_features, labels, vocab, training_module)
    return df, dataset


def evaluate():
    training_module = load_training_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = torch_load(VOCAB_PATH)
    label_encoder = torch_load(LABEL_ENCODER_PATH)
    scaler = torch_load(SCALER_PATH)

    df, dataset = prepare_test_data(TEST_CSV, vocab, label_encoder, scaler, training_module)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = training_module.TextCNN(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=len(label_encoder.classes_),
        num_numeric_features=len(training_module.NUMERIC_COLUMNS),
        pad_index=vocab[training_module.PAD_TOKEN],
    ).to(device)

    model.load_state_dict(torch_load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    all_true = []
    all_predictions = []

    test_start = time.perf_counter()
    with torch.no_grad():
        for text_inputs, numeric_inputs, labels in dataloader:
            text_inputs = text_inputs.to(device)
            numeric_inputs = numeric_inputs.to(device)
            labels = labels.to(device)

            outputs = model(text_inputs, numeric_inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            all_true.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    test_duration = time.perf_counter() - test_start
    average_loss = total_loss / total
    accuracy = accuracy_score(all_true, all_predictions) * 100

    true_labels = label_encoder.inverse_transform(all_true)
    predicted_labels = label_encoder.inverse_transform(all_predictions)
    class_names = list(label_encoder.classes_)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

    df["predicted_label"] = predicted_labels
    df["prediction_correct"] = df[training_module.LABEL_COLUMN].eq(df["predicted_label"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_CSV, index=False)
    matrix_df.to_csv(CONFUSION_MATRIX_CSV)
    METRICS_TXT.write_text(
        (
            f"Test skup: {TEST_CSV.name}\n"
            f"Broj zapisa: {len(dataset)}\n"
            f"Loss: {average_loss:.4f}\n"
            f"Accuracy: {accuracy:.2f}%\n"
            f"Vrijeme testiranja: {format_duration(test_duration)}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    print(f"Ucitan test skup: {TEST_CSV.name}")
    print(f"Broj zapisa: {len(dataset)}")
    print(f"Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Vrijeme testiranja: {format_duration(test_duration)}")
    print()
    print(report)
    print(f"Predikcije spremljene u: {PREDICTIONS_CSV}")
    print(f"Metrike spremljene u: {METRICS_TXT}")
    print(f"Confusion matrix spremljen u: {CONFUSION_MATRIX_CSV}")


if __name__ == "__main__":
    evaluate()
