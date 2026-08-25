import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F


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
        unk_index = self.vocab['<unk>'] if '<unk>' in self.vocab else 0
        indexed = [self.vocab[token] if token in self.vocab else unk_index for token in tokens]
        if len(indexed) < self.max_len:
            indexed += [0] * (self.max_len - len(indexed))  # padding
        else:
            indexed = indexed[:self.max_len]

        return torch.tensor(indexed), torch.tensor(self.labels[idx])


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1) 
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x] 
        x = torch.cat(x, 1)  
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def prepare_test_data(csv_path, vocab, label_encoder, max_len=50):
    df = pd.read_csv(csv_path)

    def row_to_text(row):
        return f"timestamp: {row['timestamp']} src_ip: {row['src_ip']} dst_ip: {row['dst_ip']} " \
               f"src_port: {row['src_port']} dst_port: {row['dst_port']} protocol: {row['protocol']} " \
               f"tcp_flags: {row['tcp_flags']}"

    df['text'] = df.apply(row_to_text, axis=1)
    labels = label_encoder.transform(df['label'])
    texts = df['text'].tolist()
    tokenizer = get_tokenizer("basic_english")
    dataset = LogDataset(texts, labels, vocab, tokenizer, max_len=max_len)
    return dataset


def evaluate_model(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds_all)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='weighted')
    return acc, precision, recall, f1


def print_metrics(name, acc, precision, recall, f1):
    print(f"=== {name} ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = torch.load("vocab.pth")
    label_encoder = torch.load("label_encoder.pth")

    # CSV s test/eval logovima
    test_csv_path = "labeled_traffic_test.csv"

    test_dataset = prepare_test_data(test_csv_path, vocab, label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    vocab_size = len(vocab)
    embed_dim = 100
    num_classes = len(label_encoder.classes_)

    #Učitavanje modela
    model_original = TextCNN(vocab_size, embed_dim, num_classes).to(device)
    model_original.load_state_dict(torch.load("textcnn_model.pth"))

    model_finetuned = TextCNN(vocab_size, embed_dim, num_classes).to(device)
    model_finetuned.load_state_dict(torch.load("textcnn_model_finetuned.pth"))

    #Evaluacija modela
    acc_o, prec_o, rec_o, f1_o = evaluate_model(model_original, test_loader, device)
    acc_f, prec_f, rec_f, f1_f = evaluate_model(model_finetuned, test_loader, device)

    #Prikaz rezultata
    print_metrics("Originalni model", acc_o, prec_o, rec_o, f1_o)
    print_metrics("Finetuned model", acc_f, prec_f, rec_f, f1_f)
