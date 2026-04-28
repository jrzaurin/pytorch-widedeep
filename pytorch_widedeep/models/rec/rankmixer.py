import math
from typing import Literal

import torch
import einops
import torch.nn.functional as F
from torch import nn
from beartype import beartype
from jaxtyping import Float, jaxtyped

from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class RankMixerTokenizer(nn.Module):
    def __init__(
        self,
        group_sizes: list[int],
        model_dim: int,
        token_size: int,
    ):
        super().__init__()

        self.group_sizes = group_sizes
        self.input_dim = sum(group_sizes)
        self.model_dim = model_dim
        self.token_size = token_size

        assert (
            self.input_dim % self.token_size == 0
        ), "sum of group_sizes must be divisible by token_size"

        self.num_tokens = self.input_dim // self.token_size

        self.proj = nn.ModuleList(
            [nn.Linear(self.token_size, model_dim) for _ in range(self.num_tokens)]
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch input_dim"]
    ) -> Float[torch.Tensor, "batch num_tokens model_dim"]:

        assert (
            x.shape[-1] == self.input_dim
        ), f"Expected input feature dimension {self.input_dim}, got {x.shape[-1]}"

        chunks = x.split(self.token_size, dim=-1)  # list of T tensors, each (B, d)
        tokens = torch.stack(
            [self.proj[i](chunks[i]) for i in range(self.num_tokens)], dim=1
        )  # (B, T, D)

        return tokens


class MultiHeadTokenMixing(nn.Module):
    def __init__(self, num_tokens: int, model_dim: int):
        super().__init__()
        # in the paper this operation is simply structural. For this they need
        # T = H, and this is what we will do here
        self.num_tokens = num_tokens
        self.num_heads = num_tokens
        self.model_dim = model_dim
        self.head_dim = model_dim // self.num_heads

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch num_tokens model_dim"]
    ) -> Float[torch.Tensor, "batch num_tokens model_dim"]:

        x = einops.rearrange(
            x, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim
        )
        x = einops.rearrange(x, "b h t d -> b h (t d)")

        return x


class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        hidden_dim = int(embed_dim * ffn_ratio)

        self.W1 = nn.Parameter(torch.empty(num_tokens, embed_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_tokens, hidden_dim, embed_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch num_tokens model_dim"]
    ) -> Float[torch.Tensor, "batch num_tokens model_dim"]:
        h = torch.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)
        out = torch.einsum("bth,thd->btd", h, self.W2) + self.b2
        return out


class SparseMoEPerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = int(model_dim * ffn_ratio)

        self.W1 = nn.Parameter(
            torch.empty(num_tokens, num_experts, model_dim, hidden_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(num_tokens, num_experts, hidden_dim))
        self.W2 = nn.Parameter(
            torch.empty(num_tokens, num_experts, hidden_dim, model_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(num_tokens, num_experts, model_dim))

        self.router = nn.Parameter(torch.empty(num_tokens, model_dim, num_experts))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.router, a=math.sqrt(5))

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch num_tokens model_dim"]
    ) -> Float[torch.Tensor, "batch num_tokens model_dim"]:

        # relu routing
        router_logits = torch.einsum("btd,tde->bte", x, self.router)
        router_weights = F.relu(router_logits)

        # Zero out non-top-K
        _, topk_idx = router_weights.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(router_weights).scatter_(-1, topk_idx, 1.0)
        router_weights = router_weights * mask

        # Normalize
        router_sum = router_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        router_weights = router_weights / router_sum  # (B, T, E)

        # Expert FFN: compute all experts, then weight-sum
        # h1: (B, T, E, hidden)
        h = torch.einsum("btd,tedh->bteh", x, self.W1) + self.b1.unsqueeze(0)
        h = F.gelu(h)
        h = self.dropout(h)

        # out_experts: (B, T, E, D)
        out_experts = torch.einsum("bteh,tehd->bted", h, self.W2) + self.b2.unsqueeze(0)

        # Weighted combination: (B, T, D)
        out = (out_experts * router_weights.unsqueeze(-1)).sum(dim=2)

        return out


class RankMixerBlock(nn.Module):

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.token_mixing = MultiHeadTokenMixing(num_tokens, embed_dim)
        self.ffn = PerTokenFFN(num_tokens, embed_dim, ffn_ratio, dropout)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch num_tokens model_dim"]
    ) -> Float[torch.Tensor, "batch num_tokens model_dim"]:

        x = x + self.token_mixing(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class MoEBlock(nn.Module):
    """Wrapper to make MoE block behave like RankMixerBlock."""

    def __init__(self, modules: nn.ModuleDict):
        super().__init__()
        self.norm1 = modules["norm1"]
        self.norm2 = modules["norm2"]
        self.token_mixing = modules["token_mixing"]
        self.ffn = modules["ffn"]

    def forward(self, x):
        x = x + self.token_mixing(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RankMixer(BaseTabularModelWithoutAttention):
    def __init__(
        self,
        *,
        column_idx: dict[str, int],
        column_groups: list[list[str]],
        cat_embed_input: list[tuple[str, int, int]] | None = None,
        cat_embed_dropout: float | None = None,
        use_cat_bias: bool | None = None,
        cat_embed_activation: str | None = None,
        continuous_cols: list[str] | None = None,
        cont_norm_layer: Literal["batchnorm", "layernorm"] | None = None,
        embed_continuous: bool | None = None,
        embed_continuous_method: (
            Literal["standard", "piecewise", "periodic"] | None
        ) = None,
        cont_embed_dim: int | None = None,
        cont_embed_dropout: float | None = None,
        cont_embed_activation: str | None = None,
        quantization_setup: dict[str, list[float]] | None = None,
        n_frequencies: int | None = None,
        sigma: float | None = None,
        share_last_layer: bool | None = None,
        full_embed_dropout: bool | None = None,
        num_tokens: int = 4,
        token_size: int = 16,
        model_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        ffn_dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super(RankMixer, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=embed_continuous,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dim=cont_embed_dim,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self._validate_input(column_groups)

        self.num_fields = len(column_idx)
        self.output_dim_ = model_dim * num_tokens

        self.column_groups = column_groups

        assert (
            num_tokens == num_heads
        ), "num_tokens must be equal to num_heads in this implementation"

        self.num_tokens = num_tokens
        self.token_size = token_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.ffn_dropout = ffn_dropout
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k

        self.tokenizer = RankMixerTokenizer(
            group_sizes=self.compute_group_sizes(),
            model_dim=model_dim,
            token_size=token_size,
        )

        # Stacked RankMixer blocks
        T = self.num_tokens
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if use_moe:
                block = nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(model_dim),
                        "norm2": nn.LayerNorm(model_dim),
                        "token_mixing": MultiHeadTokenMixing(T, model_dim),
                        "ffn": SparseMoEPerTokenFFN(
                            T, model_dim, num_experts, top_k, ffn_ratio, ffn_dropout
                        ),
                    }
                )
                self.blocks.append(MoEBlock(block))
            else:
                self.blocks.append(RankMixerBlock(T, model_dim, ffn_ratio, ffn_dropout))

        # Output norm
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        X: Float[torch.Tensor, "batch num_fields"],
    ) -> Float[torch.Tensor, "batch output_dim_"]:

        x = self._get_embeddings(X)

        # B, T, D
        x = self.tokenizer(x)

        # Forward through RankMixer blocks
        for block in self.blocks:
            x = block(x)

        x = self.output_norm(x)  # (B, T, D)

        # flatten tokens
        x = einops.rearrange(x, "b t d -> b (t d)")

        return x

    @property
    def output_dim(self) -> int:
        return self.model_dim * self.num_tokens

    def _validate_input(self, column_groups: list[list[str]]) -> None:
        _flat_groups = [col for group in column_groups for col in group]
        _cols_in_groups = set(_flat_groups)
        _cols_in_idx = set(self.column_idx.keys())
        # 1. coverage
        if _cols_in_groups != _cols_in_idx:
            missing = _cols_in_idx - _cols_in_groups
            extra = _cols_in_groups - _cols_in_idx
            raise ValueError(
                "column_groups and column_idx must reference the same columns. "
                + (f"Missing from column_groups: {missing}. " if missing else "")
                + (f"Not found in column_idx: {extra}." if extra else "")
            )
        # 2. order must match the embedding output order:
        #    cat cols in cat_embed_input order, then cont cols in continuous_cols order
        _cat_cols = [t[0] for t in self.cat_embed_input] if self.cat_embed_input else []
        _cont_cols = self.continuous_cols if self.continuous_cols else []
        _expected_order = _cat_cols + _cont_cols
        if _flat_groups != _expected_order:
            raise ValueError(
                "column_groups (flattened) must match the embedding output order: "
                "categorical columns first (in cat_embed_input order), then continuous "
                "columns (in continuous_cols order). "
                f"Expected: {_expected_order}, got: {_flat_groups}."
            )

    def compute_group_sizes(self) -> list[int]:
        col_to_flat_dim: dict[str, int] = {}
        if self.cat_embed_input is not None:
            for col, _, embed_dim in self.cat_embed_input:
                col_to_flat_dim[col] = embed_dim
        if self.continuous_cols is not None:
            cont_dim = self.cont_embed_dim if self.embed_continuous else 1
            for col in self.continuous_cols:
                col_to_flat_dim[col] = cont_dim
        return [
            sum(col_to_flat_dim[col] for col in group) for group in self.column_groups
        ]


if __name__ == "__main__":
    import numpy as np
    import torch
    import pandas as pd

    from pytorch_widedeep.preprocessing import TabPreprocessor

    # ── 1. Toy dataset ────────────────────────────────────────────────────────
    np.random.seed(42)
    N = 100
    df = pd.DataFrame(
        {
            "gender": np.random.choice(["M", "F"], N),
            "education": np.random.choice(
                ["high_school", "bachelor", "master", "phd"], N
            ),
            "age": np.random.uniform(18, 65, N),
            "income": np.random.uniform(20_000, 120_000, N),
        }
    )

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    # Explicit embed dims: gender→8, education→8
    # Each continuous col will be embedded to dim 8 by the model
    # so flat dims per group:
    #   group 1 [gender, education] : 8 + 8 = 16
    #   group 2 [age, income]       : 8 + 8 = 16  (cont_embed_dim=8)
    preprocessor = TabPreprocessor(
        cat_embed_cols=[("gender", 8), ("education", 8)],
        continuous_cols=["age", "income"],
        cols_to_scale=["age", "income"],
    )
    X = preprocessor.fit_transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print("column_idx:     ", preprocessor.column_idx)
    # → {'gender': 0, 'education': 1, 'age': 2, 'income': 3}
    print("cat_embed_input:", preprocessor.cat_embed_input)
    # → [('gender', 2, 8), ('education', 4, 8)]

    # ── 3. Column groups ──────────────────────────────────────────────────────
    # Flattened order MUST match embedding output: cats first, then conts
    # group 1 (user profile): gender + education
    # group 2 (context):      age    + income
    column_groups = [
        ["gender", "education"],
        ["age", "income"],
    ]

    # ── 4. Build RankMixer ────────────────────────────────────────────────────
    # total flat dim = 16 + 16 = 32
    # token_size = 16 → num_tokens = 32 // 16 = 2
    # num_tokens must equal num_heads: both = 2
    # model_dim = 32, head_dim = 32 // 2 = 16 ✓
    model = RankMixer(
        column_idx=preprocessor.column_idx,
        column_groups=column_groups,
        cat_embed_input=preprocessor.cat_embed_input,
        continuous_cols=preprocessor.continuous_cols,
        embed_continuous_method="standard",
        cont_embed_dim=8,
        num_tokens=2,
        token_size=16,
        model_dim=32,
        num_layers=2,
        num_heads=2,
        ffn_dropout=0.0,
        use_moe=True,
        num_experts=2,
        top_k=1,
    )

    print("\nGroup sizes:         ", model.compute_group_sizes())  # [16, 16]
    print("Tokenizer input_dim: ", model.tokenizer.input_dim)  # 32
    print("Tokenizer num_tokens:", model.tokenizer.num_tokens)  # 2

    # ── 5. Forward pass ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model(X_tensor)

    print(f"\nInput  shape: {X_tensor.shape}")  # (100, 4)
    print(f"Output shape: {out.shape}")  # (100, 64)  →  T * model_dim = 2 * 32
