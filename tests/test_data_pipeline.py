import os
import numpy as np
import pytest
import torch
from data.prepare_data import prepare_train_val_splits, tokenize_text_corpus
from src.tokenizer import Tokenizer

def test_tokenizer_encode_decode_roundtrip():
    tokenizer = Tokenizer(model_name="gpt2")

    input_text = "A quick brown fox jumps over the lazy dog"
    input_tokens = tokenizer.encode(text=input_text, add_bos=False, add_eos=False)
    # Assert token IDs is a list of integers
    assert isinstance(input_tokens[0], int)

    output_text = tokenizer.decode(input_tokens)
    # Assert decoded string contains original sample text
    assert output_text == input_text

def test_tokenizer_special_tokens():
    tokenizer = Tokenizer(model_name="gpt2")

    input_text = "A quick brown fox jumps over the lazy dog"
    input_tokens = tokenizer.encode(text=input_text, add_bos=True, add_eos=True)

    # Assert first token matches tokenizer's bos_id (or expected token)
    assert input_tokens[0] == tokenizer.bos_id
    # Assert last token matches tokenizer's eos_id (or expected token)
    assert input_tokens[-1] == tokenizer.eos_id

def test_tokenize_text_corpus_dtype():
    tokenizer = Tokenizer(model_name="gpt2")

    input_text = "A quick brown fox jumps over the lazy dog"
    input_tokens = tokenize_text_corpus(text_data=input_text, tokenizer=tokenizer)

    # Assert returned object is a np.ndarray
    assert isinstance(input_tokens, np.ndarray)
    # Assert array dtype is np.uint16 (for standard GPT vocabularies)
    assert input_tokens.dtype == np.uint16

def test_prepare_train_val_splits_file_creation(tmp_path):
    tokenizer = Tokenizer(model_name="gpt2")

    input_text = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."

    VAL_RATIO = 0.1
    train_bin_path, val_bin_path = prepare_train_val_splits(
        raw_text=input_text,
        tokenizer=tokenizer,
        val_ratio = VAL_RATIO,
        output_dir=tmp_path
    )

    # Assert train.bin and val.bin exist on disk
    assert os.path.exists(train_bin_path)
    assert os.path.exists(val_bin_path)

    train_tokens = np.fromfile(train_bin_path, dtype=np.uint16)
    val_tokens = np.fromfile(val_bin_path, dtype=np.uint16)
    # Assert loaded numpy arrays are non-empty
    assert train_tokens.size > 0
    assert val_tokens.size > 0