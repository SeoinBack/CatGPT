import numpy as np
from typing import Dict, Optional

import torch
from torch import nn

def float_to_vector(
    val, embedding_size: int, vmax = 1.0
) -> torch.Tensor:
    
    vals = torch.zeros((embedding_size,))
    for i in range(0, embedding_size, 2):
        vals[embedding_size-1-i] = val / (i + 1)
        vals[embedding_size-1-(i + 1)] = -val / (i + 1)

    return vals / (vmax / 10)
    
def float_to_cyclic_vector(
    val, embedding_size, vmax = 1.0
) -> torch.Tensor:
    
    vals = torch.zeros((embedding_size,))
    for i in range(0,embedding_size,2):
        vals[i] = np.sin(2 * np.pi * val / vmax) / (i + 1)
        vals[i + 1] = np.cos(2 * np.pi * val / vmax) / (i + 1)
    return vals / (vmax / 10)


def get_cyclic_encoding(
    token: str, embedding_size: int, vmax: float = 1.0
) -> torch.Tensor:
    if embedding_size % 2 != 0:
        raise ValueError(f"Embedding size {embedding_size} cant be odd.")

    vals = torch.zeros((embedding_size,))
    if (len(token.split('.')) != 2) or (token == '.'):
        return vals

    if token[0] == '_':
        vals += float_to_vector(float(token[1:]),embedding_size,vmax)
    else:
        vals += float_to_cyclic_vector(float(token),embedding_size,vmax)
    
    return vals
class CyclicEncoding(nn.Embedding):
    """
    A nn.Embedding inspired class to generate fixed embedding vectors that represent
    numbers passed as tokens.
    NOTE: Tokens representing numbers need to follow notation _8_-1_ to represent 0.8.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vocab: Dict,
        vmax: Optional[float] = None,
        device = 'cuda',
        *args,
        **kwargs,
    ) -> None:
        """
        Constructor for FloatEmbedding; sets up the fixed embedding matrix.

        Args:
            num_embeddings (int): size of the dictionary of embeddings.
            embedding_dim (int): the size of each embedding vector
            vocab (Dict): the language dictionary with tokens as keys and indexes as
                values. Length needs to match num_embeddings
            vmax (Optional[float]): Maximal value of float, defaults to None.

        Raises:
            ValueError: if num_embeddings does not match len(vocab).
            TypeError: if neither None nor a number is passed as vmax
            ValueError: if vmax is negative.
        """

        super(CyclicEncoding, self).__init__(
            num_embeddings, embedding_dim, *args, **kwargs
        )

        if not len(vocab) == num_embeddings:
            raise ValueError(
                f"num_embeddings needs to match size of vocabulary ({num_embeddings}!={len(vocab)})"
            )

        weights = torch.zeros(num_embeddings, embedding_dim)
        for idx, (token, index) in enumerate(vocab.items()):
            assert (
                idx == index
            ), "Please sort vocab indexes in ascending order starting from 0"
            weights[idx, :] = get_cyclic_encoding(token, embedding_dim, vmax)
        weights = weights.to(device=device)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        self.vocab = vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)