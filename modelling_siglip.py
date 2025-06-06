from typing import Optimal, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self, 
            hidden_size=768, # dimension of each patch embedding (token vector)
            intermediate_size=3072, # dimension of the feedforward network in transformer encoder
            num_hidden_layers=12, # number of transformer encoder layers
            num_attention_heads=12, # number of attention heads of multi-head attention in transforrmer encoder
            num_channels=3, # number of channels (rgb)
            image_size=224, # size of the input image
            patch_size=16, # size of each patch in the input image
            layer_norm_eps=1e-6,  # Small constant added to denominator in layer normalization to prevent division by zero or instability when variance is very small
            attention_dropout=0.0, # dropout rate of the attentoon heads in the transformer encoder
            num_image_tokens: int=None, # number of tokens in the input image
            **kwargs # other keyword arguments
        ):
        super().__init__()
        self.hidden_size = hidden_size 
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddding(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid', 
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_positions).expand((-1, 1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloadtTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
    
class SiglipVisionMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states=hidden_states)
        return hidden_states
    

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddding(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

class SiglipVisionModel(nn.Module):

    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(sekf, pixel_values) -> Tuple:

        return self.vision_model(pixel_values=pixel_values)