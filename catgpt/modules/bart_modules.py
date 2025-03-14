import json
import logging
import math
import os
import sys
import time
import numpy as np
import torch

from dataclasses import asdict, dataclass, field

from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
#from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def shift_tokens_left(input_ids: np.array, pad_token_id: int,) -> np.ndarray:
    """
    Shift input ids one token to the left.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:,:-1] = input_ids[:,1:]
    shifted_input_ids[:,-1] = pad_token_id
    
    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

def get_attention_mask(input_ids: np.array, eos_token_id: int,) -> np.ndarray:

    attention_mask = np.ones_like(input_ids)
    eos_pos = np.argmax(input_ids == eos_token_id, axis=-1)
    mask = np.arange(input_ids.shape[-1]) > eos_pos[:, None]
    
    attention_mask[mask] = 0
    return attention_mask

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class DataCollatorForCatMLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    decoder_start_token_id: int
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        #batch = BatchEncoding(
        #    {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        #)
        
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
        )
        
        if self.tokenizer.pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )    
            
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )
        
        input_ids = batch['input_ids']
        batch['decoder_input_ids'] = input_ids
        batch['decoder_attention_mask'] = get_attention_mask(input_ids, self.tokenizer.eos_token_id)
        
        labels = shift_tokens_left(input_ids, self.tokenizer.pad_token_id)
        labels_eos_pos = np.argmax(labels == self.tokenizer.eos_token_id, axis=-1)
        labels_mask = np.arange(labels.shape[-1]) > labels_eos_pos[:, None]
        labels[labels_mask] = self.label_pad_token_id
        
        batch['labels'] = labels
        
        batch_size, input_length = input_ids.shape
        
        mask_indices = np.asarray([self.random_spans_noise_mask(input_length) for i in range(batch_size)])
        input_mask = self.create_mask_ids(mask_indices.astype(np.int8))
        
        batch['input_ids'] = self.filter_input_ids(input_ids, input_mask)
        batch['attention_mask'] = get_attention_mask(batch['input_ids'], self.tokenizer.eos_token_id)
        
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            batch,
            padding='max_length',
            max_length=self.max_length,
        )
        
        for key, value in batch.items():
            batch[key] = torch.tensor(value)

        return batch

    def create_mask_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        mask_ids = np.where(start_indices != 0, self.tokenizer.mask_token_id, start_indices)
        mask_ids -= mask_indices - start_indices

        return mask_ids

    def filter_input_ids(self, input_ids, mask_ids):
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(mask_ids != 0, mask_ids, input_ids)
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        return input_ids

    def random_spans_noise_mask(self, length):

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
