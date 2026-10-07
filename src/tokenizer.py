from typing import List, Union
import torch
import tiktoken

class Tokenizer:
    """A wrapper interface around subword tokenization engines (e.g., tiktoken or HF tokenizers).

    Handles encoding strings to token IDs, decoding IDs back to text, and
    managing special tokens (<bos>, <eos>, <pad>).
    """

    def __init__(self, model_name: str = "gpt2"):
        """Initializes the underlying subword encoder and assigns special token
        IDs."""

        self.encoder = tiktoken.get_encoding(model_name)
        # Define and store special token IDs
        self.bos_id = self.encoder.eot_token
        self.eos_id = self.encoder.eot_token
        self.pad_id = self.encoder.eot_token

    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size of the tokenizer."""
        return self.encoder.n_vocab

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        allowed_special: Union[str, set] = "all",
    ) -> List[int]:
        """Encodes a string into a list of subword token IDs.

        Args:
            text: Input string to tokenize.
            add_bos: Whether to prepend the <bos> token ID.
            add_eos: Whether to append the <eos> token ID.
            allowed_special: Policy for handling special tokens in the text.

        Returns:
            List of integer token IDs.
        """
        id_list = self.encoder.encode(text, allowed_special=allowed_special)

        if add_bos:
            id_list.insert(0, self.bos_id)
        if add_eos:
            id_list.append(self.eos_id)

        return id_list

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        """Decodes a list or Tensor of token IDs back into a text string.

        Args:
            ids: List or 1D/2D PyTorch Tensor of token IDs.

        Returns:
            Decoded string.
        """
        # 1. Convert Tensor to Python list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()

        # 2. Filter or handle special tokens if required by decoder
        special_ids = {self.bos_id, self.eos_id, self.pad_id}
        clean_ids = [token_id for token_id in ids if token_id not in special_ids]

        return self.encoder.decode(clean_ids)
