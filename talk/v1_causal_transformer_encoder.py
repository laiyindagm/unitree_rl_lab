class CausalTransformerEncoder(nn.Module):
    def __init__(self, d_obs, d_model=128, n_heads=4, n_layers=2,
                 max_len=50, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_obs, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))  # 学习式位置编码
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                  # x: (B, H, d_obs)
        B, H, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :H]
        # causal mask: H×H 上三角 -inf
        mask = torch.triu(torch.full((H, H), float('-inf'),
                                     device=x.device), diagonal=1)
        h = self.blocks(h, mask=mask)      # (B, H, d_model)
        h = self.norm(h)
        # 用最后一个时刻 token 作为整段历史的总结（类似 GPT 的 last-token pooling）
        return h[:, -1]                    # (B, d_model)