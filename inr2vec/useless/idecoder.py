from typing import Callable, Tuple, List, Union #list and Union added for nerf2vec decoder

import torch
from einops import repeat
from torch import Tensor, nn

#import tinycudann as tcnn #addition from nerf2vec decoder
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False

from nerf.intant_ngp import _TruncExp #addtion from nerf2vec decoder

#This part is coded with the assitance of LLM
class CoordsEncoder:
    def __init__(
        self,
        encoding_type: str = 'manual',  # 'manual' or 'tcnn'
        encoding_conf: dict = None,     # required for 'tcnn'
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: Tuple[Callable, Callable] = (torch.sin, torch.cos),
    ) -> None:
        self.encoding_type = encoding_type
        self.input_dims = input_dims

        if encoding_type == 'tcnn':
            if not TCNN_AVAILABLE:
                raise ImportError("tiny-cuda-nn (tcnn) is not installed.")
            if encoding_conf is None:
                raise ValueError("You must provide `encoding_conf` for tcnn encoding.")
            self.encoder = tcnn.Encoding(input_dims, encoding_conf, seed=999)
            self.out_dim = self.encoder.n_output_dims

        elif encoding_type == 'manual':
            self.include_input = include_input
            self.max_freq_log2 = max_freq_log2
            self.num_freqs = num_freqs
            self.log_sampling = log_sampling
            self.periodic_fns = periodic_fns
            self._create_embedding_fn()
        else:
            raise ValueError(f"Unknown encoding_type '{encoding_type}'.")

    def _create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0

        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def apply_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == 'tcnn':
            return self.encoder(x)
        elif self.encoding_type == 'manual':
            return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)
#This part is coded with the assitance of LLM
    def embed(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layes_before_skip: int,
        num_hidden_layes_after_skip: int,
        out_dim: int,
    ) -> None:
        super().__init__()

        self.coords_enc = CoordsEncoder(in_dim)
        coords_dim = self.coords_enc.out_dim

        self.in_layer = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        self.skip_proj = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        before_skip = []
        for _ in range(num_hidden_layes_before_skip):
            before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(num_hidden_layes_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        self.after_skip = nn.Sequential(*after_skip)

    def forward(self, embeddings: Tensor, coords: Tensor) -> Tensor:
        # embeddings (B, D1)
        # coords (B, N, D2)
        coords = self.coords_enc.embed(coords)

        repeated_embeddings = repeat(embeddings, "b d -> b n d", n=coords.shape[1])

        emb_and_coords = torch.cat([repeated_embeddings, coords], dim=-1)

        x = self.in_layer(emb_and_coords)
        x = self.before_skip(x)

        inp_proj = self.skip_proj(emb_and_coords)
        x = x + inp_proj

        x = self.after_skip(x)

        return x.squeeze(-1)
