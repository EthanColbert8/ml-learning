import torch
import torch.nn as nn

# Type hinting must be used so that TorchScript can correctly capture
# our model's structure and script it.
from typing import Optional

class MLPEmbedding(nn.Module):
    '''
    A Torch module that implements an MLP, to be used as an embedding for an input vector
    in a continuous space.
    '''
    def __init__(self, n_features, n_layers, d_layers, d_embed, dropout=0):
        '''
        Params:
            n_features (int): the number of input features
            n_layers (int): the number of hidden layers in the MLP
            d_layers (int): the number of nodes per hidden layer
            d_embed (int): the dimensionality of the embedding (number
                           of output features)
            dropout (float): dropout probability between hidden layers
        '''
        super().__init__()

        use_dropout = (dropout is not None) and (dropout > 0)

        layers = [nn.Linear(n_features, d_layers), nn.GELU()]
        for index in range(n_layers-1):
            layers.append(nn.Linear(d_layers, d_layers))
            layers.append(nn.GELU())
            if use_dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(d_layers, d_embed))

        self.stack = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.stack(inputs)

class DecoderLayer(nn.Module):
    '''
    An unmasked version of a Transformer decoder layer. Almost the same,
    except the self-attention layer does *NOT* do any shifting/masking.
    This allows output particles to attend to other output particles,
    regardless of where they are in the "sequence."
    '''
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        '''
        Params:
            d_model (int): the dimensionality of embedded tokens
            n_heads (int): number of attention heads for all attention layers
            dim_feedforward (int): width of the Linear layer in between
                                   attention layers
            dropout (float): dropout probability
        '''
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.multi_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        linear_layers = [
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        ]
        self.linear = nn.Sequential(*linear_layers)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, target: torch.Tensor, memory: torch.Tensor, pad_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.self_attn(target, target, target)[0]
        x = self.norm1(target + self.dropout1(x))

        x2 = self.multi_attn(x, memory, memory, key_padding_mask=pad_mask)[0]
        x2 = self.norm2(x + self.dropout2(x2))

        x3 = self.linear(x2)
        x3 = self.norm3(x2 + self.dropout3(x3))

        return x3

class Decoder(nn.Module):
    '''
    Puts together a number of our custom decoder layers to form
    a decoder block to be used in a Transformer.
    '''
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward=512, dropout=0.1, norm=None):
        '''
        Params:
            num_layers (int): the number of decoder layers in the block
            d_model (int): the dimensionality of embedded tokens
            n_heads (int): number of attention heads for all attention layers
            dim_feedforward (int): width of the Linear layer in between
                                   attention layers
            dropout (float): dropout probability
            norm (torch.nn.Module): An optional normalization layer
        '''
        super().__init__()

        layers = [DecoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)

        self.norm = norm

    def forward(self, target: torch.Tensor, memory: torch.Tensor, pad_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = target

        for layer in self.layers:
            x = layer(x, memory, pad_mask=pad_mask)
        
        if (self.norm is not None):
            x = self.norm(x)
        
        return x

class PositionAgnosticTransformer(nn.Module):
    '''
    Puts together a standard PyTorch TransformerEncoder and our custom Decoder to form
    a Transformer network. This is VERY similar to a standard Transformer, but with our
    modified decoder that doesn't do any shifting, so there isn't any dependence on the
    positioning of tokens within the sequence.
    '''
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, n_heads, dim_feedforward=512, dropout=0.1, norm=None):
        '''
        Params:
            num_encoder_layers (int):
            num_decoder_layers (int):
            d_model (int): the dimensionality of embedded tokens
            n_heads (int): number of attention heads for all attention layers
            dim_feedforward (int): width of the Linear layer in between
                                   attention layers
            dropout (float): dropout probability
            norm (torch.nn.Module): An optional normalization layer at the end of each block
        '''
        super().__init__()

        encoder_layer_example = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer_example, num_encoder_layers, norm=norm)

        self.decoder = Decoder(num_decoder_layers, d_model, n_heads, dim_feedforward, dropout, norm=norm)

    def forward(self, src: torch.Tensor, target: torch.Tensor, pad_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        mem = self.encoder(src, src_key_padding_mask=pad_mask)

        out = self.decoder(target, mem, pad_mask)

        return out

class RecoTransformer(nn.Module):
    '''
    Puts together our PositionAgnosticTransformer module with MLP embeddings for the physics objects
    and a final Linear layer to produce output, resulting in a network that takes physics objects
    as input, computes an embedding and Transformer transformation, and produces the desired
    physics objects as output.
    '''
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, n_heads, dim_feedforward=512, dropout=0.1, norm=None, dim_embeddings=None, num_layers_embeddings=2, dim_outputs=4, jet_features=5):
        '''
        Params:
            num_encoder_layers (int):
            num_decoder_layers (int):
            d_model (int): the dimensionality of embedded tokens
            n_heads (int): number of attention heads for all attention layers
            dim_feedforward (int): width of the Linear layer in between
                                   attention layers
            dropout (float): dropout probability
            norm (torch.nn.Module): An optional normalization layer at the end of each block
            dim_embeddings (int): width of hidden layers in MLP embeddings
            num_layers_embeddings (int): number of hidden layers in MLP embeddings
            dim_output (int): dimensionality of output tokens (number of
                              output features for ONE output physics object)
            jet_features (int): number of features given as input for each jet
        '''
        super().__init__()
        if (dim_embeddings is None):
            dim_embeddings = int(d_model * 1.5)

        self.jet_embedding = MLPEmbedding(jet_features, num_layers_embeddings, dim_embeddings, d_model, dropout=dropout)
        self.lepton_embedding = MLPEmbedding(5, num_layers_embeddings, dim_embeddings, d_model, dropout=dropout)
        self.met_embedding = MLPEmbedding(2, num_layers_embeddings, dim_embeddings, d_model, dropout=dropout)

        self.transformer = PositionAgnosticTransformer(num_encoder_layers, num_decoder_layers, d_model, n_heads, dim_feedforward, dropout, norm=norm)

        self.fc = nn.Linear(d_model, dim_outputs)

    def forward(self, target: torch.Tensor, jets: torch.Tensor, leptons: torch.Tensor, met: torch.Tensor, pad_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        embedded_jets = self.jet_embedding(jets)
        embedded_leptons = self.lepton_embedding(leptons)
        embedded_met = self.met_embedding(met)

        seq = torch.cat([embedded_jets, embedded_leptons, embedded_met], dim=1)

        out = self.transformer(seq, target, pad_mask)
        out = self.fc(out)

        return out

