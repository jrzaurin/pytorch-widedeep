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
    """Tokenizer that maps a flat embedding vector into a sequence of tokens.

    The input flat vector (concatenation of all feature embeddings) is split
    into ``T = input_dim // token_size`` non-overlapping chunks of size
    ``token_size``. Each chunk is then projected independently to ``model_dim``
    via a dedicated ``nn.Linear`` layer, yielding a token sequence of shape
    ``(B, T, model_dim)``.

    Parameters
    ----------
    group_sizes: List[int]
        List of flat feature dimensions contributed by each semantic group.
        The total input dimension is ``sum(group_sizes)``.
    model_dim: int
        Output dimension $D$ for every token after the linear projection.
    token_size: int
        Size of each flat chunk that is projected to one token. The total
        input dimension ``sum(group_sizes)`` must be divisible by
        ``token_size``.

    Attributes
    ----------
    input_dim: int
        Total input dimension, equal to ``sum(group_sizes)``.
    num_tokens: int
        Number of tokens ``T = input_dim // token_size``.
    proj: nn.ModuleList
        List of ``T`` independent linear layers mapping ``token_size`` →
        ``model_dim``.
    """

    def __init__(
        self,
        group_sizes: list[int],
        model_dim: int,
        token_size: int,
    ):
        super().__init__()

        # in their figure 1, 'input_dim' is referred as "hundreds of feature
        # embeddings"
        self.input_dim = sum(group_sizes)
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
    r"""Parameter-free multi-head token mixing operation.

    Implements the structural token mixing described in Section 3.1 of the
    RankMixer paper. The operation enforces $T = H$ (number of tokens equals
    number of heads) and achieves cross-token interaction purely through
    reshaping, with **no learnable parameters**.

    Given a token sequence $X \in \mathbb{R}^{B \times T \times D}$:

    1. Split each token's $D$-dim vector into $H = T$ heads of size
       $d = D/T$: $X \in \mathbb{R}^{B \times T \times H \times d}$.
    2. Transpose the token and head axes:
       $X \in \mathbb{R}^{B \times H \times T \times d}$.
    3. Flatten the last two dimensions: $X \in \mathbb{R}^{B \times H \times (T \cdot d)}$.

    The resulting shape $B \times T \times D$ (with $H = T$) means that
    every output token now mixes information from all input tokens.

    Parameters
    ----------
    num_tokens: int
        Number of tokens $T$ in the sequence. Must satisfy $T = H$.
    model_dim: int
        Token dimension $D$. Must be divisible by ``num_tokens`` since
        ``head_dim = model_dim // num_tokens``.

    Attributes
    ----------
    num_heads: int
        Equal to ``num_tokens`` (the paper enforces $T = H$).
    head_dim: int
        Dimension of each head: ``model_dim // num_tokens``.
    """

    def __init__(self, num_tokens: int, model_dim: int):
        super().__init__()
        # in the paper this operation is simply structural. For this they need
        # T = H, and this is what we will do here
        self.num_heads = num_tokens
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
    r"""Per-token feed-forward network with independent weights per token position.

    Unlike a shared ``nn.Linear``, this module maintains a **separate** two-layer
    MLP for each of the ``T`` token positions. This allows each token to learn
    specialised transformations suited to its semantic role (e.g., user features
    vs. item features), without any cross-token weight sharing.

    The forward pass uses ``torch.einsum`` to compute all token FFNs in a
    single batched operation:

    $$h_{b,t} = \text{GELU}(x_{b,t} W_{1,t} + b_{1,t})$$
    $$\text{out}_{b,t} = h_{b,t} W_{2,t} + b_{2,t}$$

    where $W_{1,t} \in \mathbb{R}^{D \times H}$,
    $W_{2,t} \in \mathbb{R}^{H \times D}$, and $H = \lfloor D \cdot r \rfloor$
    with expansion ratio $r$.

    Parameters
    ----------
    num_tokens: int
        Number of token positions $T$. One independent MLP is allocated
        per position.
    model_dim: int
        Token dimension $D$ (both input and output dimension).
    ff_ratio: float, default = 4.0
        Hidden-dimension expansion ratio: ``hidden_dim = int(model_dim * ffn_ratio)``.
    dropout: float, default = 0.0
        Dropout applied after the GELU activation.

    Attributes
    ----------
    W1: nn.Parameter
        First projection weights of shape ``(num_tokens, model_dim, hidden_dim)``.
    b1: nn.Parameter
        First projection biases of shape ``(num_tokens, hidden_dim)``.
    W2: nn.Parameter
        Second projection weights of shape ``(num_tokens, hidden_dim, model_dim)``.
    b2: nn.Parameter
        Second projection biases of shape ``(num_tokens, model_dim)``.
    """

    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        ff_factor: float = 4.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(model_dim * ff_factor)

        self.W1 = nn.Parameter(torch.empty(num_tokens, model_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_tokens, hidden_dim, model_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, model_dim))
        self.dropout = nn.Dropout(ff_dropout)

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
    r"""Sparse Mixture-of-Experts per-token FFN as a drop-in for `PerTokenFFN`.

    Replaces the single per-token MLP in ``PerTokenFFN`` with a group of
    ``num_experts`` expert MLPs per token position. A ReLU-based router
    selects the top-``k`` experts for each token independently, computes
    their outputs, and returns a normalised weighted sum. Experts not in the
    top-``k`` are masked out before normalisation, ensuring sparsity.

    The routing and forward computation for token position $t$ are:

    $$s_{b,t} = \text{ReLU}(x_{b,t} \, R_t), \quad s_{b,t} \in \mathbb{R}^E$$
    $$\tilde{s}_{b,t} = s_{b,t} \odot \mathbf{1}[\text{top-}k], \quad
      w_{b,t} = \tilde{s}_{b,t} / \|\tilde{s}_{b,t}\|_1$$
    $$\text{out}_{b,t} = \sum_{e=1}^{E} w_{b,t,e} \cdot \text{FFN}_{t,e}(x_{b,t})$$

    where $R_t \in \mathbb{R}^{D \times E}$ is the per-token routing matrix
    and $\text{FFN}_{t,e}$ is the $e$-th expert for token position $t$.

    Parameters
    ----------
    num_tokens: int
        Number of token positions $T$.
    model_dim: int
        Token dimension $D$ (input and output).
    num_experts: int, default = 8
        Total number of expert networks per token position.
    top_k: int, default = 2
        Number of experts activated per token per forward pass.
    ff_factor: float, default = 4.0
        Hidden-dimension expansion ratio for each expert FFN:
        ``hidden_dim = int(model_dim * ff_factor)``.
    ff_dropout: float, default = 0.0
        Dropout applied after the GELU activation inside each expert.

    Attributes
    ----------
    W1: nn.Parameter
        Expert up-projection weights, shape
        ``(num_tokens, num_experts, model_dim, hidden_dim)``.
    W2: nn.Parameter
        Expert down-projection weights, shape
        ``(num_tokens, num_experts, hidden_dim, model_dim)``.
    router: nn.Parameter
        Per-token routing matrix, shape ``(num_tokens, model_dim, num_experts)``.
    """

    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        ff_factor: float = 4.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.top_k = top_k
        hidden_dim = int(model_dim * ff_factor)

        self.W1 = nn.Parameter(
            torch.empty(num_tokens, num_experts, model_dim, hidden_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(num_tokens, num_experts, hidden_dim))
        self.W2 = nn.Parameter(
            torch.empty(num_tokens, num_experts, hidden_dim, model_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(num_tokens, num_experts, model_dim))

        self.router = nn.Parameter(torch.empty(num_tokens, model_dim, num_experts))
        self.dropout = nn.Dropout(ff_dropout)

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
    """Single RankMixer block with Pre-LayerNorm residual connections.

    Applies the two-stage mixing pipeline described in the RankMixer paper:

    1. **Token mixing** (cross-token): ``x = x + TokenMix(LayerNorm(x))``
    2. **Per-token FFN** (intra-token): ``x = x + FFN(LayerNorm(x))``

    Uses ``MultiHeadTokenMixing`` for step 1 and ``PerTokenFFN`` for step 2.
    Pre-LayerNorm placement (before each sub-layer) improves gradient flow
    in deep stacks.

    Parameters
    ----------
    num_tokens: int
        Number of token positions $T$ passed to ``MultiHeadTokenMixing`` and
        ``PerTokenFFN``.
    embed_dim: int
        Token dimension $D$.
    ff_factor: float, default = 4.0
        Hidden-dimension expansion ratio for the per-token FFN.
    ff_dropout: float, default = 0.0
        Dropout applied inside the per-token FFN.
    """

    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        ff_factor: float = 4.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.token_mixing = MultiHeadTokenMixing(num_tokens, model_dim)
        self.ffn = PerTokenFFN(num_tokens, model_dim, ff_factor, ff_dropout)

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
        self.ff = modules["ff"]

    def forward(self, x):
        x = x + self.token_mixing(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class RankMixer(BaseTabularModelWithoutAttention):
    r"""Defines a `RankMixer` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class implements the RankMixer architecture introduced in the paper
    [RankMixer: Scaling Up Ranking Models via Token Mixing](https://arxiv.org/abs/2310.09607).
    RankMixer replaces the attention mechanism of transformer-based rankers with
    a parameter-free token mixing operation, enabling cheaper and more scalable
    depth whilst retaining competitive ranking quality.

    The pipeline is:

    1. **Tokenizer** — the flat embedding vector (concatenation of categorical
       and continuous feature embeddings) is chunked into ``T`` tokens of size
       ``token_size`` and each chunk is projected to ``model_dim`` via an
       independent ``nn.Linear``.
    2. **L × RankMixer blocks** — each block applies:
        - Pre-LN multi-head token mixing (cross-token, no parameters)
        - Pre-LN per-token FFN (intra-token, token-specific parameters)
    3. **Output LayerNorm** followed by token flattening to
       shape ``(B, T × model_dim)``.

    The token mixing step (with $T = H$) achieves cross-token interaction
    purely via reshaping:

    $$\text{TokenMix}(X) =
      \text{Flatten}_{3,4}\!\left(\text{Rearrange}(X,\,
        b\,t\,(h\,d) \!\to\! b\,h\,t\,d)\right)$$

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `RankMixer` model. Required to slice the tensors. e.g.
        _{'gender': 0, 'education': 1, 'age': 2, ...}_.
    column_groups: List[List[str]]
        List of lists of column names defining the semantic feature groups.
        Each group contributes one or more consecutive tokens. The columns,
        when flattened across all groups, must appear in the same order as
        the embedding output: categorical columns first (in
        ``cat_embed_input`` order), then continuous columns (in
        ``continuous_cols`` order). e.g.
        _[['gender', 'education'], ['age', 'income']]_.
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default to 0.
    use_cat_bias: bool, Optional, default = None
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None
        Activation function for the categorical embeddings
    continuous_cols: Optional[List[str]], default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]], default = None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_. if `None`, no
        normalization layer will be used.
    embed_continuous_method: Optional[Literal["piecewise", "periodic", "standard"]], default = None
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_. The _'standard'_
        embedding method is based on the FT-Transformer implementation
        presented in the paper: [Revisiting Deep Learning Models for
        Tabular Data](https://arxiv.org/abs/2106.11959v5). The _'periodic'_
        and _'piecewise'_ methods were presented in the paper: [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556).
        Please, read the papers for details.
    cont_embed_dim: Optional[int], default = None
        Size of the continuous column embeddings. Required if
        ``embed_continuous_method`` is not `None`.
    cont_embed_dropout: Optional[float], default = None
        Dropout for the continuous embeddings. If `None`, it will default to 0.0
    cont_embed_activation: Optional[str], default = None
        Activation function for the continuous embeddings if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
        If `None`, no activation function will be applied.
    quantization_setup: Optional[Dict[str, List[float]]], default = None
        This parameter is used when the _'piecewise'_ method is used to embed
        the continuous cols. It is a dict where keys are the name of the
        continuous columns and values are lists with the boundaries for the
        quantization of the continuous_cols. If the _'piecewise'_ method is
        used, this parameter is required.
    n_frequencies: Optional[int], default = None
        This is the so called _'k'_ in [On Embeddings for Numerical Features
        in Tabular Deep Learning](https://arxiv.org/abs/2203.05556), and is
        the number of 'frequencies' used to represent each continuous column.
        If the _'periodic'_ method is used, this parameter is required.
    sigma: Optional[float], default = None
        Sigma parameter used to initialise the 'frequency weights' for the
        _'periodic'_ embedding method. If the _'periodic'_ method is used,
        this parameter is required.
    share_last_layer: Optional[bool], default = None
        If `True` the linear layer that turns the frequencies into embeddings
        will be shared across the continuous columns. Required when using
        the _'periodic'_ method.
    full_embed_dropout: bool, Optional, default = None
        If `True`, the full embedding corresponding to a column will be masked
        out/dropped. If `None`, it will default to `False`.
    num_tokens: int, default = 4
        Number of tokens $T$ produced by the tokenizer. Must equal
        ``num_heads``. Determined implicitly as
        ``input_dim // token_size``, where ``input_dim`` is the sum of
        all group sizes computed by ``compute_group_sizes``.
    token_size: int, default = 16
        Size of each flat embedding chunk projected to one token of
        dimension ``model_dim``. The total flat embedding dimension must
        be divisible by ``token_size``.
    model_dim: int, default = 32
        Dimension $D$ of every token after the tokenizer projection. Must
        be divisible by ``num_heads`` since ``head_dim = model_dim // num_tokens``.
    num_layers: int, default = 4
        Number of stacked RankMixer blocks $L$.
    num_heads: int, default = 4
        Number of heads for the token mixing operation. Must equal
        ``num_tokens`` (the paper enforces $T = H$).
    ffn_ratio: float, default = 4.0
        Hidden-dimension expansion ratio inside the per-token FFN:
        ``hidden_dim = int(model_dim * ffn_ratio)``.
    ffn_dropout: float, default = 0.1
        Dropout applied inside the per-token FFN after the GELU activation.
    use_moe: bool, default = False
        If `True`, each block uses a ``SparseMoEPerTokenFFN`` instead of the
        standard ``PerTokenFFN``.
    num_experts: int, default = 8
        Total number of expert networks per token position. Only used when
        ``use_moe=True``.
    top_k: int, default = 2
        Number of experts activated per token per forward pass. Only used
        when ``use_moe=True``.

    Attributes
    ----------
    tokenizer: RankMixerTokenizer
        Module that converts the flat embedding vector into a token sequence.
    blocks: nn.ModuleList
        Stack of ``num_layers`` ``RankMixerBlock`` or ``MoEBlock`` modules.
    output_norm: nn.LayerNorm
        Final layer normalisation applied to each token before flattening.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models.rec import RankMixer
    >>> colnames = ["a", "b", "c", "d"]
    >>> cat_embed_input = [(u, i, 8) for u, i in zip(colnames[:2], [4, 6])]
    >>> column_idx = {k: v for v, k in enumerate(colnames)}
    >>> column_groups = [["a", "b"], ["c", "d"]]
    >>> model = RankMixer(
    ...     column_idx=column_idx,
    ...     column_groups=column_groups,
    ...     cat_embed_input=cat_embed_input,
    ...     continuous_cols=["c", "d"],
    ...     embed_continuous_method="standard",
    ...     cont_embed_dim=8,
    ...     num_tokens=2,
    ...     token_size=16,
    ...     model_dim=32,
    ...     num_layers=2,
    ...     num_heads=2,
    ... )
    >>> X = torch.cat((torch.empty(5, 2).random_(4), torch.rand(5, 2)), dim=1)
    >>> out = model(X)
    """

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
        ff_factor: float = 4.0,
        ff_dropout: float = 0.1,
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

        self.column_groups = column_groups

        assert (
            num_tokens == num_heads
        ), "num_tokens must be equal to num_heads in this implementation"

        self.num_tokens = num_tokens
        self.token_size = token_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_factor = ff_factor
        self.ff_dropout = ff_dropout
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
                        "ff": SparseMoEPerTokenFFN(
                            T, model_dim, num_experts, top_k, ff_factor, ff_dropout
                        ),
                    }
                )
                self.blocks.append(MoEBlock(block))
            else:
                self.blocks.append(RankMixerBlock(T, model_dim, ff_factor, ff_dropout))

        # Output norm
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        X: Float[torch.Tensor, "batch num_fields"],
    ) -> Float[torch.Tensor, "batch output_dim"]:

        x = self._get_embeddings(X)
        x = self._rearrange_embeddings_to_groups(x)

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
        _cols_in_groups = set[str](_flat_groups)
        _cols_in_idx = set[str](self.column_idx.keys())

        if _cols_in_groups != _cols_in_idx:
            missing = _cols_in_idx - _cols_in_groups
            extra = _cols_in_groups - _cols_in_idx
            raise ValueError(
                "column_groups and column_idx must reference the same columns. "
                + (f"Missing from column_groups: {missing}. " if missing else "")
                + (f"Not found in column_idx: {extra}." if extra else "")
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

    def _rearrange_embeddings_to_groups(self, x: torch.Tensor) -> torch.Tensor:
        col_to_slice: dict[str, tuple[int, int]] = {}
        cursor = 0

        if self.cat_embed_input is not None:
            for col, _, embed_dim in self.cat_embed_input:
                col_to_slice[col] = (cursor, cursor + embed_dim)
                cursor += embed_dim

        if self.continuous_cols is not None:
            cont_dim = self.cont_embed_dim if self.embed_continuous else 1
            for col in self.continuous_cols:
                col_to_slice[col] = (cursor, cursor + cont_dim)
                cursor += cont_dim

        idx: list[int] = []
        for group in self.column_groups:
            for col in group:
                start, end = col_to_slice[col]
                idx.extend(range(start, end))

        if idx == list(range(cursor)):
            return x

        device = x.device
        perm = torch.tensor(idx, dtype=torch.long, device=device)
        return x[:, perm]
