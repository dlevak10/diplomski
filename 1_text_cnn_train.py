from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "firewall_logs_labeled" / "1_train_logs_combined_labeled.csv"
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

LABEL_COLUMN = "label"
MAX_SEQUENCE_LENGTH = 64
MIN_TOKEN_FREQUENCY = 1

TEXT_COLUMNS = [
    "device_hostname",
    "action",
    "policy_id",
    "rule_name",
    "src_zone",
    "dst_zone",
    "src_interface",
    "dst_interface",
    "src_ip",
    "dst_ip",
    "nat_src_ip",
    "nat_dst_ip",
    "protocol",
    "tcp_flags",
    "session_status",
    "session_reason",
]

NUMERIC_COLUMNS = [
    "src_port",
    "dst_port",
    "nat_src_port",
    "nat_dst_port",
    "bytes",
    "packets",
]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_value(value) -> str:
    if pd.isna(value):
        return "missing"
    value = str(value).strip()
    if value == "" or value == "-":
        return "none"
    return value.replace(" ", "_")


def port_bucket(port: int) -> str:
    if port == 0:
        return "none"
    if port < 1024:
        return "well_known"
    if port < 49152:
        return "registered"
    return "ephemeral"


def traffic_size_bucket(value: int) -> str:
    if value <= 0:
        return "none"
    if value < 500:
        return "small"
    if value < 10000:
        return "medium"
    return "large"


def row_to_text(row: pd.Series) -> str:
    tokens = [f"{column}={clean_value(row[column])}" for column in TEXT_COLUMNS]

    src_port = int(row["src_port"])
    dst_port = int(row["dst_port"])
    bytes_count = int(row["bytes"])
    packets_count = int(row["packets"])

    tokens.extend(
        [
            f"src_port_bucket={port_bucket(src_port)}",
            f"dst_port={dst_port}",
            f"dst_port_bucket={port_bucket(dst_port)}",
            f"bytes_bucket={traffic_size_bucket(bytes_count)}",
            f"packets_bucket={traffic_size_bucket(packets_count)}",
        ]
    )

    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return text.split()


def build_vocab(texts: list[str], min_frequency: int = MIN_TOKEN_FREQUENCY) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common():
        if count >= min_frequency and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> torch.Tensor:
    unk_index = vocab[UNK_TOKEN]
    pad_index = vocab[PAD_TOKEN]
    indices = [vocab.get(token, unk_index) for token in tokenize(text)]

    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices.extend([pad_index] * (max_len - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


def normalize_numeric_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    numeric_df = df[NUMERIC_COLUMNS].replace("-", 0).copy()

    for column in NUMERIC_COLUMNS:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce").fillna(0)

    numeric_df["bytes"] = np.log1p(numeric_df["bytes"])
    numeric_df["packets"] = np.log1p(numeric_df["packets"])

    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric_df).astype(np.float32)
    return numeric_features, scaler


class FirewallLogDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        numeric_features: np.ndarray,
        labels: np.ndarray,
        vocab: dict[str, int],
        max_len: int = MAX_SEQUENCE_LENGTH,
    ):
        self.texts = texts
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_indices = encode_text(self.texts[index], self.vocab, self.max_len)
        return text_indices, self.numeric_features[index], self.labels[index]


def prepare_data(csv_path: Path) -> tuple[FirewallLogDataset, LabelEncoder, StandardScaler, dict[str, int]]:
    df = pd.read_csv(csv_path)

    missing_columns = sorted(
        set(TEXT_COLUMNS + NUMERIC_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    )
    if missing_columns:
        raise ValueError(f"Nedostaju stupci u CSV datoteci: {missing_columns}")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column].replace("-", 0), errors="coerce").fillna(0).astype(int)

    texts = df.apply(row_to_text, axis=1).tolist()
    vocab = build_vocab(texts)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[LABEL_COLUMN])

    numeric_features, scaler = normalize_numeric_features(df)
    dataset = FirewallLogDataset(texts, numeric_features, labels, vocab)

    print(f"Ucitan skup: {csv_path.name}")
    print(f"Broj zapisa: {len(df)}")
    print(f"Broj tokena u vokabularu: {len(vocab)}")
    print(f"Klase: {list(label_encoder.classes_)}")

    return dataset, label_encoder, scaler, vocab


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_numeric_features: int,
        pad_index: int,
        kernel_sizes: list[int] | None = None,
        num_filters: int = 100,
        dropout: float = 0.5,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (kernel_size, embed_dim)) for kernel_size in kernel_sizes]
        )
        self.numeric_norm = nn.LayerNorm(num_numeric_features)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes) + num_numeric_features, num_classes)

    def forward(self, text_inputs: torch.Tensor, numeric_inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_inputs)
        embedded = embedded.unsqueeze(1)

        conv_outputs = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled_outputs = [
            F.max_pool1d(output, output.size(2)).squeeze(2) for output in conv_outputs
        ]
        text_features = torch.cat(pooled_outputs, dim=1)
        numeric_features = self.numeric_norm(numeric_inputs)

        features = torch.cat([text_features, numeric_features], dim=1)
        features = self.dropout(features)
        return self.fc(features)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)} min {remaining_seconds:.2f} s"


if __name__ == "__main__":
    dataset, label_encoder, scaler, vocab = prepare_data(TRAIN_CSV)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=len(label_encoder.classes_),
        num_numeric_features=len(NUMERIC_COLUMNS),
        pad_index=vocab[PAD_TOKEN],
    ).to(device)

    labels_for_weights = dataset.labels
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(labels_for_weights, len(label_encoder.classes_)).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    final_loss = 0.0
    final_accuracy = 0.0
    training_start = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for text_inputs, numeric_inputs, labels in dataloader:
            text_inputs = text_inputs.to(device)
            numeric_inputs = numeric_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(text_inputs, numeric_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        final_loss = average_loss
        final_accuracy = accuracy

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss: {average_loss:.4f} | Accuracy: {accuracy:.2f}%"
        )

    training_duration = time.perf_counter() - training_start
    print(
        f"Zavrsni rezultat treniranja | "
        f"Loss: {final_loss:.4f} | Accuracy: {final_accuracy:.2f}% | "
        f"Vrijeme treniranja: {format_duration(training_duration)}"
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / "textcnn_model_trained.pth") #naucene tezine textcnn modela
    torch.save(vocab, ARTIFACT_DIR / "textcnn_vocab_trained.pth") #vokabular tj. mapa tokena
    torch.save(label_encoder, ARTIFACT_DIR / "textcnn_label_encoder_trained.pth") #mapiranje labela u brojeve
    torch.save(scaler, ARTIFACT_DIR / "textcnn_numeric_scaler_trained.pth") #scaler za normalizaciju numeričkih značajki

    print(f"Model i pomocni objekti spremljeni su u: {ARTIFACT_DIR}")
