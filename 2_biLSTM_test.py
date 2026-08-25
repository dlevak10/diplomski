import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchtext.data.utils import get_tokenizer
import pandas as pd
# -----------------------------#
#1. Definicija LogDataset klase#
# -----------------------------#
class LogDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        indices = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            pad_len = self.max_len - len(indices)
            indices = torch.cat([indices, torch.zeros(pad_len, dtype=torch.long)])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return indices, label
# -----------------------------#
# 2. Definicija modela BiLSTM  #
# -----------------------------#
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 

    def forward(self, x):
        x = self.embedding(x)                     
        lstm_out, _ = self.lstm(x)               
        out = lstm_out[:, -1, :]                  
        out = self.dropout(out)
        return self.fc(out)

def prepare_test_data(csv_path, vocab, label_encoder):
    df = pd.read_csv(csv_path)

    def row_to_text(row):
        return f"timestamp: {row['timestamp']} src_ip: {row['src_ip']} dst_ip: {row['dst_ip']} " \
               f"src_port: {row['src_port']} dst_port: {row['dst_port']} protocol: {row['protocol']} " \
               f"tcp_flags: {row['tcp_flags']}"

    df['text'] = df.apply(row_to_text, axis=1)
    texts = df['text'].tolist()
    labels = label_encoder.transform(df['label'])
    tokenizer = get_tokenizer("basic_english")
    dataset = LogDataset(texts, labels, vocab, tokenizer, max_len=50)
    return dataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Učitavanje spremljenih objekta
    vocab = torch.load("bilstm_vocab.pth")
    label_encoder = torch.load("bilstm_label_encoder.pth")
    
    vocab.set_default_index(vocab["<unk>"])

    vocab_size = len(vocab)
    embed_dim = 100
    hidden_dim = 128
    num_classes = len(label_encoder.classes_)

    model = BiLSTM(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load("bilstm_model.pth", map_location=device))
    model.eval()

    #Učitavanje testnih podatka sa labelama
    test_dataset = prepare_test_data("labeled_traffic_test.csv", vocab, label_encoder)
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

    #Izračun metrika
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
