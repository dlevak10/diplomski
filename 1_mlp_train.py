from pathlib import Path
import ipaddress
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "firewall_logs_labeled" / "1_train_logs_combined_labeled.csv"
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

MODEL_PATH = ARTIFACT_DIR / "mlp_model_trained.pth"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "mlp_label_encoder_trained.pth"
SCALER_PATH = ARTIFACT_DIR / "mlp_feature_scaler_trained.pth"
CATEGORY_ENCODERS_PATH = ARTIFACT_DIR / "mlp_category_encoders_trained.pth"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "mlp_feature_columns_trained.pth"

LABEL_COLUMN = "label"

IP_COLUMNS = [
    "src_ip",
    "dst_ip",
    "nat_src_ip",
    "nat_dst_ip",
]

NUMERIC_COLUMNS = [
    "src_port",
    "dst_port",
    "nat_src_port",
    "nat_dst_port",
    "bytes",
    "packets",
]

CATEGORICAL_COLUMNS = [
    "device_hostname",
    "action",
    "policy_id",
    "rule_name",
    "src_zone",
    "dst_zone",
    "src_interface",
    "dst_interface",
    "protocol",
    "tcp_flags",
    "session_status",
    "session_reason",
]

TIME_COLUMNS = [
    "hour",
    "minute",
    "second",
]

AGGREGATE_COLUMNS = [
    "unique_dst_ip_per_src",
    "unique_dst_port_per_src",
    "conn_attempts",
]


def clean_value(value) -> str:
    if pd.isna(value):
        return "missing"
    value = str(value).strip()
    if value == "" or value == "-":
        return "none"
    return value


def safe_ipv4_to_int(value) -> int:
    value = clean_value(value)
    if value == "none":
        return 0
    try:
        return int(ipaddress.IPv4Address(value))
    except ValueError:
        return 0


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unique_dst_ip_per_src"] = df.groupby(["time_window", "src_ip"])["dst_ip"].transform("nunique")
    df["unique_dst_port_per_src"] = df.groupby(["time_window", "src_ip"])["dst_port"].transform("nunique")
    df["conn_attempts"] = df.groupby(
        ["time_window", "src_ip", "dst_ip", "dst_port"]
    )["timestamp"].transform("count")
    return df


def transform_with_known_classes(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    class_to_index = {class_name: index for index, class_name in enumerate(encoder.classes_)}
    return series.map(lambda value: class_to_index.get(value, -1))


def build_feature_frame(
    df: pd.DataFrame,
    category_encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    required_columns = (
        ["timestamp"]
        + IP_COLUMNS
        + NUMERIC_COLUMNS
        + CATEGORICAL_COLUMNS
        + [LABEL_COLUMN]
    )
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Nedostaju stupci u CSV datoteci: {missing_columns}")

    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = timestamps.dt.hour.fillna(0).astype(int)
    df["minute"] = timestamps.dt.minute.fillna(0).astype(int)
    df["second"] = timestamps.dt.second.fillna(0).astype(int)
    df["time_window"] = timestamps.dt.floor("min").fillna(pd.Timestamp("1970-01-01"))

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column].replace("-", 0), errors="coerce").fillna(0)

    df = add_aggregate_features(df)

    feature_frame = pd.DataFrame(index=df.index)

    for column in IP_COLUMNS:
        feature_frame[f"{column}_int"] = df[column].apply(safe_ipv4_to_int)

    for column in NUMERIC_COLUMNS + TIME_COLUMNS + AGGREGATE_COLUMNS:
        feature_frame[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if category_encoders is None:
        category_encoders = {}

    for column in CATEGORICAL_COLUMNS:
        values = df[column].map(clean_value)
        encoded_column = f"{column}_encoded"

        if fit:
            encoder = LabelEncoder()
            feature_frame[encoded_column] = encoder.fit_transform(values)
            category_encoders[column] = encoder
        else:
            encoder = category_encoders[column]
            feature_frame[encoded_column] = transform_with_known_classes(values, encoder)

    return feature_frame, category_encoders


def prepare_data(
    csv_path: Path,
) -> tuple["TrafficDataset", LabelEncoder, StandardScaler, dict[str, LabelEncoder], list[str]]:
    df = pd.read_csv(csv_path)
    feature_frame, category_encoders = build_feature_frame(df, fit=True)
    feature_columns = list(feature_frame.columns)

    scaler = StandardScaler()
    features = scaler.fit_transform(feature_frame).astype(np.float32)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[LABEL_COLUMN]).tolist()

    dataset = TrafficDataset(features, labels)

    print(f"Ucitan skup: {csv_path.name}")
    print(f"Broj zapisa: {len(df)}")
    print(f"Broj znacajki: {len(feature_columns)}")
    print(f"Klase: {list(label_encoder.classes_)}")

    return dataset, label_encoder, scaler, category_encoders, feature_columns


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    category_encoders: dict[str, LabelEncoder],
    feature_columns: list[str],
) -> np.ndarray:
    feature_frame, _ = build_feature_frame(
        df,
        category_encoders=category_encoders,
        fit=False,
    )

    missing_features = sorted(set(feature_columns) - set(feature_frame.columns))
    if missing_features:
        raise ValueError(f"Nedostaju znacajke nakon obrade: {missing_features}")

    feature_frame = feature_frame[feature_columns]
    return scaler.transform(feature_frame).astype(np.float32)


class TrafficDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: list[int]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)} min {remaining_seconds:.2f} s"


if __name__ == "__main__":
    dataset, label_encoder, scaler, category_encoders, feature_columns = prepare_data(TRAIN_CSV)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_classes=len(label_encoder.classes_),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(dataset.labels, len(label_encoder.classes_)).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
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
    torch.save(label_encoder, LABEL_ENCODER_PATH)
    torch.save(scaler, SCALER_PATH)
    torch.save(category_encoders, CATEGORY_ENCODERS_PATH)
    torch.save(feature_columns, FEATURE_COLUMNS_PATH)

    print(f"MLP model spremljen u: {MODEL_PATH}")
    print(f"MLP label encoder spremljen u: {LABEL_ENCODER_PATH}")
    print(f"MLP scaler spremljen u: {SCALER_PATH}")
    print(f"MLP enkoderi kategorija spremljeni u: {CATEGORY_ENCODERS_PATH}")
    print(f"MLP popis znacajki spremljen u: {FEATURE_COLUMNS_PATH}")
