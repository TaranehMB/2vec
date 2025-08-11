from typing import Callable, Tuple, List, Union, Optional

import torch
from einops import repeat
from torch import Tensor, nn

# Import tcnn conditionally to make it optional
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False

# Import TruncExp conditionally
class _TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


class PositionalEncoder:
    """Traditional positional encoding with sin/cos functions."""
    def __init__(
        self,
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: Tuple[Callable, Callable] = (torch.sin, torch.cos),
    ) -> None:
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class TCNNEncoder:
    """Coordinate encoder using tiny-cuda-nn."""
    def __init__(
        self,
        encoding_conf: dict,
        input_dims: int = 3
    ) -> None:
        if not TCNN_AVAILABLE:
            raise ImportError("tinycudann is not available. Please install it or use PositionalEncoder.")
            
        self.input_dims = input_dims
        self.coords_enc = tcnn.Encoding(input_dims, encoding_conf, seed=999)
        self.out_dim = self.coords_enc.n_output_dims

    def apply_encoding(self, x):
        return self.coords_enc(x)

    def embed(self, inputs: Tensor) -> Tensor:
        result_encoding = self.apply_encoding(inputs.view(-1, self.input_dims))
        result_encoding = result_encoding.view(inputs.size()[0], inputs.size()[1], -1)
        return result_encoding


class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers_before_skip: int,
        num_hidden_layers_after_skip: int,
        out_dim: int,
        # Optional parameters for different encoder types and NeRF
        encoding_type: str = "positional",  # "positional" or "tcnn"
        encoding_conf: Optional[dict] = None,  # For TCNN
        positional_encoding_config: Optional[dict] = None,  # For positional encoding
        nerf_mode: bool = False,  # Enable NeRF-specific outputs and features
        aabb: Optional[Union[torch.Tensor, List[float]]] = None,  # For NeRF bounds
    ) -> None:
        super().__init__()

        # Fix parameter spelling inconsistency from original code
        num_hidden_layers_before_skip = num_hidden_layers_before_skip
        num_hidden_layers_after_skip = num_hidden_layers_after_skip

        # Setup coordinate encoder based on type
        if encoding_type == "tcnn":
            if not TCNN_AVAILABLE:
                raise ImportError("tinycudann is not available. Please use encoding_type='positional'")
            if encoding_conf is None:
                raise ValueError("encoding_conf must be provided when using tcnn encoder")
            self.coords_enc = TCNNEncoder(encoding_conf=encoding_conf, input_dims=in_dim)
        else:  # default to positional encoding
            pe_config = positional_encoding_config or {}
            self.coords_enc = PositionalEncoder(input_dims=in_dim, **pe_config)
        
        coords_dim = self.coords_enc.out_dim

        # NeRF-specific attributes
        self.nerf_mode = nerf_mode
        if nerf_mode:
            if aabb is None:
                raise ValueError("aabb must be provided when using nerf_mode=True")
            self.aabb = torch.tensor(aabb) if isinstance(aabb, list) else aabb
            self.in_dim = in_dim
            trunc_exp = _TruncExp.apply
            self.density_activation = lambda x: trunc_exp(x - 1)

        # Network architecture
        self.in_layer = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())
        self.skip_proj = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        before_skip = []
        for _ in range(num_hidden_layers_before_skip):
            before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(num_hidden_layers_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        self.after_skip = nn.Sequential(*after_skip)

    def forward(self, embeddings: Tensor, coords: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the implicit decoder.
        
        Args:
            embeddings: Shape (B, D1) - Batch of embeddings
            coords: Shape (B, N, D2) - Batch of coordinate sets
            
        Returns:
            If nerf_mode=False: Tensor of shape (B, N) or (B, N, out_dim)
            If nerf_mode=True: Tuple of (rgb, density) tensors
        """
        batch_size, n_coords, _ = coords.size()
        
        # Handle empty inputs for NeRF ray marching
        if n_coords == 0:
            if self.nerf_mode:
                rgb = torch.zeros([batch_size, 0, 3], device=coords.device)
                density = torch.zeros([batch_size, 0, 1], device=coords.device)
                return rgb, density
            else:
                # Return appropriate empty tensor based on output dimension
                out_shape = list(self.after_skip[-1].weight.size())
                return torch.zeros([batch_size, 0, out_shape[0]], device=coords.device).squeeze(-1)

        # Process coordinates based on mode
        if self.nerf_mode:
            # Apply AABB normalization for NeRF
            aabb_min, aabb_max = torch.split(self.aabb, self.in_dim, dim=-1)
            normalized_coords = (coords - aabb_min) / (aabb_max - aabb_min)
            selector = ((normalized_coords > 0.0) & (normalized_coords < 1.0)).all(dim=-1)
            # Encode the normalized coordinates
            encoded_coords = self.coords_enc.embed(normalized_coords)
        else:
            # Encode the raw coordinates
            encoded_coords = self.coords_enc.embed(coords)

        # Repeat embeddings to match coordinate batch size
        repeated_embeddings = repeat(embeddings, "b d -> b n d", n=encoded_coords.shape[1])
        
        # Concatenate embeddings with encoded coordinates
        emb_and_coords = torch.cat([repeated_embeddings, encoded_coords], dim=-1)
        
        # Forward pass through network
        x = self.in_layer(emb_and_coords)
        x = self.before_skip(x)
        
        inp_proj = self.skip_proj(emb_and_coords)
        x = x + inp_proj
        
        x = self.after_skip(x)
        
        # Process output based on mode
        if self.nerf_mode:
            # Split into RGB and density for NeRF
            rgb, density_before_activation = x[..., :3], x[..., 3]
            density_before_activation = density_before_activation[:, :, None]
            
            # Apply activation functions and selector mask
            density = self.density_activation(density_before_activation) * selector[..., None]
            rgb = torch.sigmoid(rgb)
            
            return rgb, density
        else:
            # Standard output
            return x.squeeze(-1)