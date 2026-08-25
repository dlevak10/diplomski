import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import ipaddress

# ------------------------------#
# 1. Dataset + Preprocessing    #
# ------------------------------#
class TrafficDataset(Dataset):
    def __init__(self, df, label_encoder, scaler, feature_cols):
        self.X = torch.tensor(scaler.transform(df[feature_cols]), dtype=torch.float32)
        self.y = torch.tensor(df['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def add_aggregate_features(df, time_window_col='time_window'):
    # Broj jedinstvenih dst_ip kojima pristupa jedan src_ip unutar vremenskog intervala
    df['unique_dst_ip_per_src'] = df.groupby([time_window_col, 'src_ip'])['dst_ip'].transform('nunique')

    # Broj jedinstvenih dst_portova kojima pristupa jedan src_ip unutar vremenskog intervala
    df['unique_dst_port_per_src'] = df.groupby([time_window_col, 'src_ip'])['dst_port'].transform('nunique')

    # Broj konekcijskih pokušaja s istim src_ip, dst_ip, dst_port u vremenskom intervalu
    df['conn_attempts'] = df.groupby([time_window_col, 'src_ip', 'dst_ip', 'dst_port'])['timestamp'].transform('count')

    return df


def preprocess(csv_path):
    df = pd.read_csv(csv_path)

    #Pretvori IP adrese u int
    for col in ['src_ip', 'dst_ip']:
        df[col] = df[col].apply(lambda x: int(ipaddress.IPv4Address(x)))

    #Timestamp -> datetime i ekstrahiraj vrijeme
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second

    #Ako nema, napravi stupac 'time_window' za agregaciju 
    df['time_window'] = df['timestamp'].dt.floor('T')  #svaki zapis zaokružen na minutu

    #Label encoding 
    for col in ['protocol', 'tcp_flags']:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df = add_aggregate_features(df, time_window_col='time_window')

    #Label encoder
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    #Feature kolone+nove agregacijske
    feature_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port',
                    'protocol', 'tcp_flags', 'hour', 'minute', 'second',
                    'unique_dst_ip_per_src', 'unique_dst_port_per_src', 'conn_attempts']

    #Normalizacija featura
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, label_encoder, scaler, feature_cols


# ----------------#
# 2. MLP Model    #
# ----------------#
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


# -------------------#
# 3. Training Petlja #
# -------------------#
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100 * correct / total


# ---------#
# 4. Main  #
# ---------#
if __name__ == "__main__":
    csv_path = "labeled_traffic_test.csv"
    df, label_encoder, scaler, feature_cols = preprocess(csv_path)

    dataset = TrafficDataset(df, label_encoder, scaler, feature_cols)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=len(feature_cols), hidden_dim=128, num_classes=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss, acc = train_model(dataloader, model, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

    #Spremanje modela, label encodear i scalera
    torch.save(model.state_dict(), "mlp_model.pth")
    torch.save(label_encoder, "mlp_label_encoder.pth")
    torch.save(scaler, "mlp_scaler.pth")

    print("Model, label encoder i scaler su spremljeni.")
