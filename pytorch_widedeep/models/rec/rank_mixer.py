import torch
from torch import nn
from beartype import beartype
from jaxtyping import Float, jaxtyped


class Tokenizer(nn.Module):
    def __init__(
        self,
        group_sizes: list[int],
        embed_dims: list[int],
        model_dim: int,
        token_size: int | None = None,
    ):
        super().__init__()
        assert len(group_sizes) == len(
            embed_dims
        ), "group_sizes and embed_dims must have the same length"
        self.group_sizes = group_sizes
        self.embed_dims = embed_dims
        self.input_dim = sum([gsz * edm for gsz, edm in zip(group_sizes, embed_dims)])
        self.model_dim = model_dim
        self.token_size = (
            token_size
            if token_size is not None
            else min([gsz * edm for gsz, edm in zip(group_sizes, embed_dims)])
        )

        assert (
            self.input_dim % self.token_size == 0
        ), "sum of group_sizes must be divisible by token_size"

        self.T = self.input_dim // self.token_size

        self.proj = nn.ModuleList(
            [nn.Linear(self.token_size, model_dim) for _ in range(self.T)]
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch input_dim"]
    ) -> Float[torch.Tensor, "batch T model_dim"]:

        assert (
            x.shape[-1] == self.input_dim
        ), f"Expected input feature dimension {self.input_dim}, got {x.shape[-1]}"

        chunks = x.split(self.token_size, dim=-1)  # list of T tensors, each (B, d)
        tokens = torch.stack(
            [self.proj[i](chunks[i]) for i in range(self.T)], dim=1
        )  # (B, T, D)

        return tokens


if __name__ == "__main__":
    # Imagine 3 feature groups:
    #   - group 1: 2 categorical embeddings of dim 8  → 2*8 = 16
    #   - group 2: 1 categorical embedding  of dim 16 → 1*16 = 16
    #   - group 3: 4 categorical embeddings of dim 4  → 4*4 = 16
    # Total input_dim = 48, token_size auto-selected as min(16, 16, 16) = 16
    # → T = 48 // 16 = 3 tokens
    group_sizes = [2, 1, 4]
    embed_dims = [8, 16, 4]
    model_dim = 32
    tokenizer = Tokenizer(
        group_sizes=group_sizes, embed_dims=embed_dims, model_dim=model_dim
    )
    print(f"input_dim:  {tokenizer.input_dim}")  # 48
    print(f"token_size: {tokenizer.token_size}")  # 16
    print(f"T (tokens): {tokenizer.T}")  # 3
    B = 4
    x = torch.randn(B, tokenizer.input_dim)
    tokens = tokenizer(x)
    print(f"Input shape:  {x.shape}")  # (4, 48)
    print(f"Output shape: {tokens.shape}")  # (4, 3, 32) → (batch, T, model_dim)
