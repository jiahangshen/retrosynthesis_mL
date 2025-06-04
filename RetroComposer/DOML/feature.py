import torch
# 特征提取器（以RNN处理SMILES为例）
class SMILESFeaturizer(torch.nn.Module):
    def __init__(self, vocab_size=64, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        x = self.embed(x)  # x: [batch, seq_len]
        _, (h_n, _) = self.rnn(x)
        return h_n.squeeze(0)  # [batch, hidden_dim]
    

