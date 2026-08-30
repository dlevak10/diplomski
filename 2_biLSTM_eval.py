from pathlib import Path
import importlib.util
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = BASE_DIR / "1_biLSTM_train.py"
EVAL_CSV = BASE_DIR / "firewall_logs_labeled" / "2_eval_logs_combined_labeled.csv"
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
RESULTS_DIR = BASE_DIR / "eval_results"

MODEL_PATH = ARTIFACT_DIR / "bilstm_model_trained.pth"
VOCAB_PATH = ARTIFACT_DIR / "bilstm_vocab_trained.pth"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "bilstm_label_encoder_trained.pth"

BATCH_SIZE = 64


def load_training_module():
    spec = importlib.util.spec_from_file_location("bilstm_training", TRAINING_SCRIPT)
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


def prepare_stage_data(csv_path: Path, vocab, label_encoder, training_module):
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
    labels = label_encoder.transform(df[training_module.LABEL_COLUMN]).tolist()
    dataset = training_module.FirewallLogDataset(texts, labels, vocab)

    return df, dataset


def evaluate(
    csv_path: Path = EVAL_CSV,
    stage: str = "eval",
    results_dir: Path = RESULTS_DIR,
):
    training_module = load_training_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = torch_load(VOCAB_PATH)
    label_encoder = torch_load(LABEL_ENCODER_PATH)

    df, dataset = prepare_stage_data(csv_path, vocab, label_encoder, training_module)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = training_module.BiLSTM(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        num_classes=len(label_encoder.classes_),
        pad_index=vocab[training_module.PAD_TOKEN],
    ).to(device)

    model.load_state_dict(torch_load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    all_true = []
    all_predictions = []

    stage_start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            all_true.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    stage_duration = time.perf_counter() - stage_start
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

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = results_dir / f"bilstm_{stage}_predictions.csv"
    metrics_txt = results_dir / f"bilstm_{stage}_metrics.txt"
    confusion_matrix_csv = results_dir / f"bilstm_{stage}_confusion_matrix.csv"

    df.to_csv(predictions_csv, index=False)
    matrix_df.to_csv(confusion_matrix_csv)
    metrics_txt.write_text(
        (
            "Model: BiLSTM\n"
            f"Stadij: {stage}\n"
            f"Skup: {csv_path.name}\n"
            f"Broj zapisa: {len(dataset)}\n"
            f"Loss: {average_loss:.4f}\n"
            f"Accuracy: {accuracy:.2f}%\n"
            f"Vrijeme {stage}: {format_duration(stage_duration)}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    print("Model: BiLSTM")
    print(f"Stadij: {stage}")
    print(f"Ucitan skup: {csv_path.name}")
    print(f"Broj zapisa: {len(dataset)}")
    print(f"Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Vrijeme {stage}: {format_duration(stage_duration)}")
    print()
    print(report)
    print(f"Predikcije spremljene u: {predictions_csv}")
    print(f"Metrike spremljene u: {metrics_txt}")
    print(f"Confusion matrix spremljen u: {confusion_matrix_csv}")


if __name__ == "__main__":
    evaluate()
