import re
import numpy as np
from typing import Any, Dict, Optional

from transformers.trainer import Trainer
from transformers.utils import logging

import torch
from torch import nn

from catgpt.modules.encodings import CyclicEncoding
from catgpt.modules.factories import MODEL_TO_EMBEDDING_FN

logger = logging.get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_trainer_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to take out a subset of a dictionary with keys that are
    important for `CustomTrainer` but cant be passed down to `Trainer`.

    Args:
        dictionary (dict): Dict with keyword arguments for `CustomTrainer` constructor.

    Returns:
        dict: Dict with keyword arguments for `CustomTrainer` that cant be passed to
            childclass constructor (`Trainer`).
    """
    keys_to_keep = [
#        "verbose_evaluation",
        "numerical",
        "d_model",
        "vocab_size",
        "vmax",
#        "model_type",
#        "mem_len",
#        "training_logs",
#        "train_config",
#        "alternating_collator",
    ]
    keep_dict = {}
    for keep_key in keys_to_keep:
        for key, val in dictionary.items():
            if re.search(keep_key, key) is not None:
                keep_dict[key] = val
    return keep_dict


class CustomHFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        
        child_kwargs = get_trainer_dict(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k not in child_kwargs}
        super().__init__(*args, **kwargs)
        
        self.use_numerical_encodings = child_kwargs.get(
            "use_numerical_encodings", False
        )

        if self.use_numerical_encodings:
            logger.info("Attempting to use numerical encodings.")
            # self.numerical_encodings_type = child_kwargs.get(
            #     "numerical_encodings_type", "float"
            # )
            
            self.numerical_encodings_format = child_kwargs.get(
                "numerical_encodings_format", "sum"
            )
            self.numerical_encodings_dim = child_kwargs.get(
                "numerical_encodings_dim", 16
            )
            
            if self.numerical_encodings_format == "concat":

                if self.numerical_encodings_dim > child_kwargs["d_model"]:
                    raise ValueError(
                        "Numerical encoding size cant be bigger than embedding size"
                    )

                self.combine_embed = self.overwrite_embed

            elif self.numerical_encodings_format == "sum":
                self.numerical_encodings_dim = child_kwargs["d_model"]

                self.combine_embed = self.sum_embed
            
            else:
                raise ValueError(
                    f"Unknown float encoding format {self.numerical_encodings_format}."
                )

            self.numerical_encoder = CyclicEncoding(
                num_embeddings=child_kwargs['vocab_size'],
                embedding_dim=self.numerical_encodings_dim,
                vocab={k: v for k, v in sorted(self.tokenizer.vocab.items(), key=lambda item: item[1])},
                vmax=child_kwargs.get("vmax",1.),
                device=self.model.device,
            )
        
        self.model_embed = eval(
            MODEL_TO_EMBEDDING_FN[type(self.model.base_model).__name__]
        )
        self.save_attention = child_kwargs.get("save_attention", False)
        self.counter = 0
        
    def sum_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        return e + num_e

    def overwrite_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        e[:, :, -self.numerical_encodings_dim :] = num_e
        return e
        
    def save_attention(self, inputs: torch.Tensor, attention: torch.Tensor):
        """  
        Save the attention weights for the current batch.

        Args:
            inputs (torch.Tensor): input_ids
            attention (torch.Tensor): attention tensor

        """

        for idx, a in enumerate(attention):
            for i, aa in enumerate(a):
                np.save(
                    f"batch_{self.counter}_layer_{idx}_tup_{i}", aa.detach().numpy()
                )

        for i, inp in enumerate(inputs):
            tokens = self.tokenizer.convert_ids_to_tokens(inp.tolist())
            with open(f"batch_{self.counter}_sample_{i}.txt", "w") as f:
                f.write(str(tokens))
        self.counter += 1
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None        

        if self.use_numerical_encodings:
            model_inputs = inputs.copy()

            embeddings = self.model_embed(inputs["input_ids"])
            numerical_embeddings = self.numerical_encoder(inputs["input_ids"])
            embeddings = self.combine_embed(embeddings, numerical_embeddings)
            model_inputs.pop("input_ids", None)

            if not self.save_attention:
                outputs = model(inputs_embeds=embeddings, **model_inputs)
            else:
                # Attention config
                outputs = model(
                    inputs_embeds=embeddings,
                    **model_inputs,
                    output_attentions=True,
                    output_hidden_states=False,
                )
                self.save_attention(inputs["input_ids"], outputs[-1])
        else:
            outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss