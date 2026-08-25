import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import ipaddress
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
# -----------------------------#
# 1.  Definicija modela        #
# -----------------------------#
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TrafficDataset(Dataset):
    def __init__(self, df, label_encoder, scaler, feature_cols):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(label_encoder.transform(df['label']), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def add_aggregate_features(df, time_window_col='time_window'):
    df['unique_dst_ip_per_src'] = df.groupby([time_window_col, 'src_ip'])['dst_ip'].transform('nunique')
    df['unique_dst_port_per_src'] = df.groupby([time_window_col, 'src_ip'])['dst_port'].transform('nunique')
    df['conn_attempts'] = df.groupby([time_window_col, 'src_ip', 'dst_ip', 'dst_port'])['timestamp'].transform('count')
    return df

def safe_label_encode(series, label_encoder):
    classes = list(label_encoder.classes_)
    # Ako je vrijednost poznata se transformira a inače je -1
    return series.apply(lambda x: label_encoder.transform([x])[0] if x in classes else -1)

def preprocess_test(csv_path, label_encoder, scaler, feature_cols):
    df = pd.read_csv(csv_path)

    #IP adrese u int
    for col in ['src_ip', 'dst_ip']:
        df[col] = df[col].apply(lambda x: int(ipaddress.IPv4Address(x)))

    #Ekkstract timestampa
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_window'] = df['timestamp'].dt.floor('min')

    # Label encode protokol i tcp_flagovi
    for col in ['protocol', 'tcp_flags']:
        df[col] = df[col].astype(str)
        df[col] = safe_label_encode(df[col], label_encoder)

    df = add_aggregate_features(df)

    if len(df) == 0:
        raise ValueError("Nema podataka nakon obrade!")

    #Skaliranje numeričkih featura
    df[feature_cols] = scaler.transform(df[feature_cols])

    return df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Učitavanje spremljenih objekta
    label_encoder = torch.load("mlp_label_encoder.pth")
    scaler = torch.load("mlp_scaler.pth")

    feature_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port',
                    'protocol', 'tcp_flags', 'hour', 'minute', 'second',
                    'unique_dst_ip_per_src', 'unique_dst_port_per_src', 'conn_attempts']

    #Inicijalizacija modela
    input_dim = len(feature_cols)
    hidden_dim = 128  #Kao u treningu
    output_dim = len(label_encoder.classes_)

    model = MLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load("mlp_model.pth"))
    model.to(device)
    model.eval()

    #Preprocesiranje test dataseta
    df_test = preprocess_test("labeled_traffic_test.csv", label_encoder, scaler, feature_cols)

    test_dataset = TrafficDataset(df_test, label_encoder, scaler, feature_cols)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
