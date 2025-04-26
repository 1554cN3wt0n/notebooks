

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    return nn, np, torch


@app.cell
def _(torch):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return (dev,)


@app.cell
def _():
    vocab = {k:v for v,k in enumerate("0123456789+ ")}
    idx2char = {k:v for v,k in vocab.items()}
    vocab_size = len(vocab)
    seq_len = 7
    epochs = 30
    batch_size = 32
    return epochs, idx2char, seq_len, vocab


@app.cell
def _(idx2char, vocab):
    def encode(s):
        return [vocab[i] for i in s]
    def decode(idxs):
        return "".join([idx2char[i] for i in idxs])
    return (decode,)


@app.cell
def _(np, seq_len, torch, vocab):
    # Generate dataset
    def generate_dataset(num_samples):
        data = []
        labels = []
        for _ in range(num_samples):
            a = np.random.randint(0, 1000)
            b = np.random.randint(0, 1000)
            c = a + b
            input_str = f"{a}+{b}"
            output_str = str(c)

            # Pad with zeros to SEQ_LENGTH
            input_list = [vocab[char] for char in input_str]
            input_list += [vocab[" "]] * (seq_len - len(input_list)) #padding the input

            output_list = [vocab[digit] for digit in output_str]
            output_list += [vocab[" "]] * (seq_len - len(output_list)) #padding the output.

            data.append(input_list)
            labels.append(output_list)
        return torch.tensor(data), torch.tensor(labels)
    return (generate_dataset,)


@app.cell
def _(generate_dataset, torch):
    train_x, train_y = generate_dataset(50000)
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=32)
    return (train_dl,)


@app.cell
def _(dev, nn, torch):
    class MyModel(nn.Module):
        def __init__(self, vocab_size=12, hidden_dim=128, n_layers=1):
            super().__init__()
            self.embed = nn.Embedding(vocab_size,hidden_dim)
            self.encoder = nn.LSTM(hidden_dim,hidden_dim,n_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim,hidden_dim,n_layers, batch_first=True)
            self.proj = nn.Linear(hidden_dim,vocab_size)
        def forward(self, x):
            x = self.embed(x)
            x, (h,c) = self.encoder(x)
            x, (h,c) = self.decoder(torch.zeros(x.shape).to(dev), (h,c))
            x = self.proj(x)
            return x
    return (MyModel,)


@app.cell
def _(MyModel, dev, nn, torch):
    model = MyModel(hidden_dim=128, n_layers=1).to(dev)
    criterion = nn.CrossEntropyLoss(ignore_index=0) #ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return criterion, model, optimizer


@app.cell
def _(criterion, dev, epochs, model, optimizer, train_dl):
    model.train()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_dl:
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1,2), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / (len(train_dl))}")
    return


@app.cell
def _(decode, dev, model, seq_len, torch, vocab):
    model.eval()
    q = "199+162"
    x = [vocab[char] for char in q]
    x += [vocab[" "]] * (seq_len - len(x))
    x = model(torch.tensor(x).reshape(1,-1).to(dev)).cpu()
    decode(torch.argmax(x, axis=-1).numpy().tolist()[0]).strip()
    return


@app.cell
def _():
    # torch.save(model.state_dict(), "lstm_3digits.pt")
    return


if __name__ == "__main__":
    app.run()
