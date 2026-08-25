import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

# ------------------------------
# 1. Dataset
# ------------------------------
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

# ------------------------------
# 2. Vocabulary builder
# ------------------------------
def build_vocab(texts, tokenizer):
    def yield_tokens():
        for text in texts:
            yield tokenizer(text)
    return build_vocab_from_iterator(yield_tokens(), specials=["<unk>"])

# ------------------------------#
# 3. Podaci iz CSV-a            #
# ------------------------------#
def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    def row_to_text(row):
        return f"timestamp: {row['timestamp']} src_ip: {row['src_ip']} dst_ip: {row['dst_ip']} " \
               f"src_port: {row['src_port']} dst_port: {row['dst_port']} protocol: {row['protocol']} " \
               f"tcp_flags: {row['tcp_flags']}"

    df['text'] = df.apply(row_to_text, axis=1)
    le = LabelEncoder()
    labels = le.fit_transform(df['label'])
    texts = df['text'].tolist()
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(texts, tokenizer)
    vocab.set_default_index(vocab["<unk>"])
    dataset = LogDataset(texts, labels, vocab, tokenizer, max_len=50)
    return dataset, le, vocab

# ------------------------------#
# 4. biLSTM Model               #
# ------------------------------#
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
        out = lstm_out[:, -1, :]                  #Koristi zadnji output po vremenu
        out = self.dropout(out)
        return self.fc(out)

# -------------------------------------#
# 5. Trening petlja + spremanje modela #
# -------------------------------------#
if __name__ == "__main__":
    dataset, label_encoder, vocab = prepare_data("labeled_traffic_train.csv")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab)
    embed_dim = 100
    hidden_dim = 128
    num_classes = len(label_encoder.classes_)

    model = BiLSTM(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
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

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Accuracy: {100 * correct / total:.2f}%")

    # Spremi model
    torch.save(model.state_dict(), "bilstm_model.pth")
    print("Model je spremljen kao 'bilstm_model.pth'")
    # Spremi vokabular
    torch.save(vocab, "bilstm_vocab.pth")
    print("Vocab je spremljen kao 'bilstm_vocab.pth'")
    # Spremi label encoder
    torch.save(label_encoder, "bilstm_label_encoder.pth")
    print("LabelEncoder je spremljen kao 'bilstm_label_encoder.pth'")
