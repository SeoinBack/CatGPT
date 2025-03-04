import numpy as np
import os
import warnings
import re
import tokenizers
import json
import itertools

from typing import Any, Dict, Optional, List, Tuple
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import AddedToken

class T5TokenizerForCat(PreTrainedTokenizer):
    """
    T5Tokenizer for CatGPT
    """
    def __init__(
        self,
        vocab: List,
        extra_ids=100,
        additional_special_tokens=None,
        add_prefix_space=True,
        max_len=1024,
        **kwargs,
    ) -> None:
        
        self._extra_ids = extra_ids
        self.max_len = max_len
        self.__name__ = 'T5TokenizerForCat'

        eos_token = kwargs.pop("eos_token", "<eos>")
        bos_token = kwargs.pop("bos_token", "<bos>")
        unk_token = kwargs.pop("unk_token", "<unk>")
        pad_token = kwargs.pop("pad_token", "<pad>")
        mask_token = kwargs.pop("mask_token", "<mask>")
        special_tokens = [eos_token, bos_token, unk_token, pad_token, mask_token]
        
        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        vocab_dict = {v:i for i,v in enumerate(special_tokens+additional_special_tokens+vocab)}
        self.tokenizer = self.get_tokenizer(vocab_dict, special_tokens+additional_special_tokens)
        self.vocab = vocab
        self.special_tokens= special_tokens + additional_special_tokens
        self._extra_ids = extra_ids
        self.add_prefix_space = add_prefix_space
        
        super().__init__(
            #tokenizer_object = self.tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            add_prefix_space=add_prefix_space,
            vocab_size=self.tokenizer.get_vocab_size(),
            max_len = self.max_len,
            **kwargs,
        )

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_tokenizer(self, vocab_dict, special_tokens):
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab_dict,unk_token='<unk>'))
        
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
            tokenizers.pre_tokenizers.WhitespaceSplit(), # Remove spacing
        ])
        
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.enable_truncation(max_length=1024)
        return tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def get_sentinel_tokens(self):
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )

    def get_sentinel_token_ids(self):
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1
            
    def __getstate__(self):
        state = self.__dict__.copy()
        state['tokenizer'] = None
        return state
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        
        vocab_dict = {v: i for i, v in enumerate(self.special_tokens + self.vocab)}
        self.tokenizer = self.get_tokenizer(vocab_dict, self.special_tokens)

    def tokenize(self, text, **kwargs) -> List[str]:
        tokens = super().tokenize(text, **kwargs)
        return tokens

    def _tokenize(self, text, **kwargs):
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def _convert_token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_token(index)
        
    """
    def convert_tokens_to_string(self, tokens):
        
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.tokenizer.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
                
        out_string += self.tokenizer.decode(current_sub_tokens)
        return out_string.strip()
    """
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory")
        
        vocab_filename = (filename_prefix or "") + "vocab.txt"
        vocab_filepath = os.path.join(save_directory, vocab_filename)
        
        vocab = self.tokenizer.get_vocab()
        
        sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
        with open(vocab_filepath, "w", encoding="utf-8") as writer:
            for token, idx in sorted_vocab:
                writer.write(token + "\n")
        
        return (vocab_filepath,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path:os.PathLike,):
        pretrained_model_path = str(pretrained_model_path)
        
        config_file = 'tokenizer_config.json'
        special_tokens_file = 'special_tokens_map.json'
        vocab_file = 'vocab.txt'
        
        with open(os.path.join(pretrained_model_path,special_tokens_file), encoding='utf-8') as reader:
            special_tokens_dict = json.load(reader)
        special_tokens = list(itertools.chain.from_iterable(x if isinstance(x,list) else [x] for x in special_tokens_dict.values()))
        
        with open(os.path.join(pretrained_model_path,config_file), encoding='utf-8') as reader:
            config= json.load(reader)
        
        config.pop('vocab_size',None)
        config.pop('added_tokens_decoder',None)
        
        with open(os.path.join(pretrained_model_path,vocab_file), 'r') as fr:
            lines = [line.strip() for line in fr.readlines()]
        vocab = [line for line in lines if line not in special_tokens]

        return cls(vocab, **config)