import os
import numpy as np
from src.tokenizer import Tokenizer


def tokenize_text_corpus(
    text_data: str, tokenizer: Tokenizer, eos_between_documents: bool = True
) -> np.ndarray:
    """Tokenizes a raw text string into a flat 1D NumPy array of unsigned

    integers.

    Args:
        text_data: Raw text corpus string (or document sequence).
        tokenizer: Instantiated Tokenizer object.
        eos_between_documents: Whether to split text into documents and append
          EOS tokens.

    Returns:
        1D np.ndarray with dtype uint16 (or uint32 if vocab_size > 65535).
    """
    id_type = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32

    if eos_between_documents and "\n\n" in text_data:
        # Split text into individual documents
        documents = [doc.strip() for doc in text_data.split("\n\n") if doc.strip()]
        all_ids = []
        for doc in documents:
            doc_ids = tokenizer.encode(text=doc, add_bos=False, add_eos=True)
            all_ids.extend(doc_ids)
    else:
        all_ids = tokenizer.encode(text=text_data, add_bos=False, add_eos=eos_between_documents)

    return np.array(all_ids, dtype=id_type)


def prepare_train_val_splits(
    raw_text: str,
    tokenizer: Tokenizer,
    val_ratio: float = 0.1,
    output_dir: str = "data/processed",
) -> tuple[str, str]:
    """Splits raw text corpus into train and validation sets, tokenizes both,

    and writes binary uint arrays to disk.

    Args:
        raw_text: Raw input text dataset.
        tokenizer: Instantiated Tokenizer object.
        val_ratio: Fraction of dataset to allocate for validation.
        output_dir: Target directory to save train.bin and val.bin.

    Returns:
        Tuple of filepaths: (train_bin_path, val_bin_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Tokenize full corpus first
    tokenized_text = tokenize_text_corpus(text_data=raw_text, tokenizer=tokenizer)

    # 2. Split into train and val sets
    split_idx = int(len(tokenized_text) * (1 - val_ratio))
    train_tokens = tokenized_text[:split_idx]
    val_tokens = tokenized_text[split_idx:]

    # 3. Write binary files
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")

    train_tokens.tofile(train_bin_path)
    val_tokens.tofile(val_bin_path)

    return train_bin_path, val_bin_path