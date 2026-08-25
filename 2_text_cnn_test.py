import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer

# ----------------------------------#
# Dataset za inference (bez labela) #
# ----------------------------------#
class LogDatasetInfer(Dataset):
    def __init__(self, texts, vocab, tokenizer, max_len=50):
        self.texts = texts
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        indices = torch.tensor([self.vocab[token] if token in self.vocab else self.vocab["<unk>"] for token in tokens], dtype=torch.long)
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            pad_len = self.max_len - len(indices)
            indices = torch.cat([indices, torch.zeros(pad_len, dtype=torch.long)])
        return indices

# --------------#
# Model TextCNN #
# --------------#
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)          
        x = x.unsqueeze(1)        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# -------------------------------------#
# Funkcija za pripremu teksta iz csv-a #
# -------------------------------------#
def prepare_texts(csv_path):
    df = pd.read_csv(csv_path)
    def row_to_text(row):
        return f"timestamp: {row['timestamp']} src_ip: {row['src_ip']} dst_ip: {row['dst_ip']} " \
               f"src_port: {row['src_port']} dst_port: {row['dst_port']} protocol: {row['protocol']} " \
               f"tcp_flags: {row['tcp_flags']}"
    df['text'] = df.apply(row_to_text, axis=1)
    return df, df['text'].tolist()

# ---------------------------#
# Main inference funcija     #
# ---------------------------#
def inference(csv_path, output_csv_path, model_path, vocab_path, label_encoder_path, batch_size=8, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Učitavanje vokabulara i label encodera
    vocab = torch.load(vocab_path)
    label_encoder = torch.load(label_encoder_path)

    #Priprema tekstova
    df, texts = prepare_texts(csv_path)

    tokenizer = get_tokenizer("basic_english")
    dataset = LogDatasetInfer(texts, vocab, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #Inicijalizacija modela
    vocab_size = len(vocab)
    embed_dim = 100
    num_classes = len(label_encoder.classes_)
    model = TextCNN(vocab_size, embed_dim, num_classes).to(device)

    #Učitavanje modela
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    #Dekodiranje numeričke predikcije natrag u string labele
    pred_labels = label_encoder.inverse_transform(all_preds)

    #Dodavanje predikcije u DataFrame
    df['predicted_label'] = pred_labels

    #Spremanje u CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Predikcije su spremljene u {output_csv_path}")

# ---------#
# Main     #
# ---------#
if __name__ == "__main__":
    input_csv = "combined_sorted_logs_test.csv"
    output_csv = "tested_labeled_logs.csv"

    inference(
        csv_path=input_csv,
        output_csv_path=output_csv,
        model_path="textcnn_model.pth",
        vocab_path="vocab.pth",
        label_encoder_path="label_encoder.pth",
        batch_size=8,
        max_len=50,
    )
