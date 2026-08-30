from collections import Counter
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "firewall_logs_labeled" / "1_train_logs_combined_labeled.csv"
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

MODEL_PATH = ARTIFACT_DIR / "bilstm_model_trained.pth"
VOCAB_PATH = ARTIFACT_DIR / "bilstm_vocab_trained.pth"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "bilstm_label_encoder_trained.pth"

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


class FirewallLogDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int = MAX_SEQUENCE_LENGTH,
    ):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        text_indices = encode_text(self.texts[index], self.vocab, self.max_len)
        return text_indices, self.labels[index]


def prepare_data(csv_path: Path) -> tuple[FirewallLogDataset, LabelEncoder, dict[str, int]]:
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
    labels = label_encoder.fit_transform(df[LABEL_COLUMN]).tolist()

    dataset = FirewallLogDataset(texts, labels, vocab)

    print(f"Ucitan skup: {csv_path.name}")
    print(f"Broj zapisa: {len(df)}")
    print(f"Broj tokena u vokabularu: {len(vocab)}")
    print(f"Klase: {list(label_encoder.classes_)}")

    return dataset, label_encoder, vocab


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_index: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        _, (hidden, _) = self.lstm(embedded)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
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
    dataset, label_encoder, vocab = prepare_data(TRAIN_CSV)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTM(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        num_classes=len(label_encoder.classes_),
        pad_index=vocab[PAD_TOKEN],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(dataset.labels, len(label_encoder.classes_)).to(device)
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

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
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
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(vocab, VOCAB_PATH)
    torch.save(label_encoder, LABEL_ENCODER_PATH)

    print(f"BiLSTM model spremljen u: {MODEL_PATH}")
    print(f"BiLSTM vokabular spremljen u: {VOCAB_PATH}")
    print(f"BiLSTM label encoder spremljen u: {LABEL_ENCODER_PATH}")
