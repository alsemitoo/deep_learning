# compare_tokenizers.py
# Unified experiment runner: Byte-level vs HF-BPE vs Custom-BPE
#
# Usage examples:
#   python compare_tokenizers.py --mode all
#   python compare_tokenizers.py --mode hf_bpe
#
# Modes: "all", "byte", "hf_bpe", "custom_bpe"
#

import os
import math
import json
import time
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tqdm import tqdm

# ---------------------------
# Experiment configuration
# ---------------------------
VOCAB_SIZE = 10000 # From 4K to 10K to accommodate Chinese characters
BLOCK_SIZE = 512
BATCH_SIZE = 16
N_LAYERS = 6
N_HEADS = 8
N_EMBD = 256
D_FF = 1024
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 15000            # change to larger for real training
EVAL_INTERVAL = 250
EVAL_ITERS = 30             # number of batches to estimate loss (keeps eval fast)
NUM_SAMPLES = 15000          # streaming samples from wikipedia (tune as needed)
MAX_CHARS = 10_000_000
SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "compare_runs"
MODES = "all"               # options: "all", "byte", "hf_bpe", "custom_bpe"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(SEED)

# ---------------------------
# Text Cleaning Utilities
# ---------------------------
import re

def clean_whitespace(text: str):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    return text.strip()

def remove_empty_parens(text: str):
    return re.sub(r'\(\s*\)', '', text)

def remove_wiki_markup(text: str):
    # headings like == Title ==
    text = re.sub(r'={2,}.*?={2,}', ' ', text)
    # templates {{...}}
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    # HTML tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    # file links
    text = re.sub(r'\[\[File:.*?\]\]', ' ', text)
    # remove leftover [[ ]]
    text = text.replace('[[', ' ').replace(']]', ' ')
    return text

def collapse_punctuation(text: str):
    return re.sub(r'([.,!?;:])\1+', r'\1', text)

def is_garbage(line: str):
    letters = sum(c.isalpha() for c in line)
    punct = sum(c in ".,!?;:()[]{}" for c in line)
    return punct > letters * 2

def clean_text(text: str):
    text = remove_wiki_markup(text)
    text = remove_empty_parens(text)
    text = collapse_punctuation(text)
    text = clean_whitespace(text)
    return text

# ---------------------------
# Tokenizers
# ---------------------------

# 1) Byte-level tokenizer
class ByteLevelTokenizer:
    """
    A standard byte-level tokenizer that maps all 256 bytes to IDs 0-255.
    Special tokens are mapped to IDs starting from 256.
    """
    def __init__(self, max_length: int = 512):
        # The base vocabulary size covers all 256 possible byte values
        self.base_vocab_size = 256
        self.max_length = max_length

    def _byte_to_id(self, byte_value: int) -> int:
        """Map a byte value (0-255) directly to its ID (0-255)"""
        return byte_value

    def _id_to_byte(self, token_id: int) -> int:
        """Map an ID (0-255) back to its byte value (0-255)"""
        return token_id

    def decode(self, token_sequence: list[int]) -> str:
        """Convert token IDs back to bytes, then decode bytes to text."""
        byte_values = []
        for token_id in token_sequence:
            # Only process IDs that correspond to actual bytes (0-255)
            if token_id < self.base_vocab_size:
                byte_values.append(self._id_to_byte(token_id))

        try:
            # Convert the list of byte values back to a bytes object, then decode as UTF-8
            return bytes(byte_values).decode('utf-8', errors='replace')
        except Exception:
            return ""

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes the entire text into a single tensor, strictly using
        byte IDs 0-255, and omitting BOS/EOS for the sliding-window LM task.
        """
        # Convert text to a sequence of byte values (0-255)
        byte_values = list(text.encode('utf-8'))
        
        # Map each byte value to its token ID (0-255)
        token_sequence = [self._byte_to_id(b) for b in byte_values]

        return torch.tensor(token_sequence, dtype=torch.long)

# 2) HuggingFace tokenizers BPE (standard)
def train_hf_bpe_tokenizer(texts: List[str], vocab_size: int = VOCAB_SIZE, save_path: str = None):
    # Tokenizers library bpe trainer expects a list of files OR pre-tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        initial_alphabet=[chr(i) for i in range(32, 127)]
    )

    # Write temporary file list
    tmp_file = os.path.join(SAVE_DIR, "hf_bpe_training.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    tokenizer.train([tmp_file], trainer)
    # post-processor to add BOS/EOS tokens automatically during encode
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                        ("[EOS]", tokenizer.token_to_id("[EOS]"))]
    )
    if save_path:
        tokenizer.save(save_path)
    return tokenizer

# 3) Custom BPE
class CustomBPETokenizer:
    """
    Pure BPE tokenizer with GPT-2 style pre-tokenization,
    consistent merges, no lowercase, no UNK in practice,
    and stable encode/decode symmetry.
    """

    def __init__(self, vocab_size=8000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.pad_token  = "<PAD>"
        self.bos_token  = "<BOS>"
        self.eos_token  = "<EOS>"
        self.unk_token  = "<UNK>"

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id
        }
        self.id_to_token = {}

        self.merges = []
        self.trained = False
        self.cache = {}

        # Regex to capture Chinese/Unicode characters
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+""",
            re.UNICODE
        )

    def _tokenize_text(self, text):
        """GPT-2 style token splitting with Ġ for leading spaces."""
        tokens = re.findall(self.pat, text)
        out = []
        for t in tokens:
            if t.startswith(" "):
                out.append("Ġ" + t[1:])
            else:
                out.append(t)
        return out

    def _get_word_freq(self, texts):
        freq = Counter()
        for t in texts:
            freq.update(self._tokenize_text(t))
        return freq

    def train(self, texts, verbose=True):
        if verbose:
            print(f"Training BPE (vocab={self.vocab_size})")

        # 1. Pre-tokenize
        freq = self._get_word_freq(texts)

        # 2. Initialize splits: each token split into characters
        splits = {tok: list(tok) for tok in freq.keys()}

        # 3. Add initial characters to vocab
        for tok in freq:
            for ch in splits[tok]:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        # 4. Learn merges
        pbar = tqdm(total=self.vocab_size - len(self.vocab), disable=not verbose)
        while len(self.vocab) < self.vocab_size:

            # Count all adjacent pairs
            pair_freq = Counter()
            for tok, chars in splits.items():
                f = freq[tok]
                for a, b in zip(chars, chars[1:]):
                    pair_freq[(a, b)] += f

            if not pair_freq:
                break

            best = max(pair_freq, key=pair_freq.get)
            A, B = best
            merged = A + B

            self.merges.append((A, B))
            self.vocab[merged] = len(self.vocab)

            # Apply merge to all tokens
            new_splits = {}
            for tok, chars in splits.items():
                i = 0
                new = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == A and chars[i+1] == B:
                        new.append(merged)
                        i += 2
                    else:
                        new.append(chars[i])
                        i += 1
                new_splits[tok] = new

            splits = new_splits
            pbar.update(1)

        pbar.close()

        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.trained = True
        self.cache = {}

    def _apply_bpe(self, token):
        """Apply merges to a single token."""
        if token in self.cache:
            return self.cache[token]

        chars = list(token)

        for A, B in self.merges:
            merged = A + B
            i = 0
            new = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == A and chars[i+1] == B:
                    new.append(merged)
                    i += 2
                else:
                    new.append(chars[i])
                    i += 1
            chars = new

        self.cache[token] = chars
        return chars

    def encode(self, text, add_special_tokens=True):
        if not self.trained:
            raise ValueError("Train tokenizer first")

        tokens = self._tokenize_text(text)

        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)

        for tok in tokens:
            for sub in self._apply_bpe(tok):
                ids.append(self.vocab.get(sub, self.unk_token_id))

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids[: self.max_length]

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for i in ids:
            if skip_special_tokens and i in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ):
                continue
            tokens.append(self.id_to_token.get(i, ""))

        text = "".join(tokens)
        return text.replace("Ġ", " ").strip()

    def encode_lm_data(self, text, chunk_size=50000):
        """Chunked encoding without truncation."""
        all_ids = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            toks = self._tokenize_text(chunk)
            for t in toks:
                for sub in self._apply_bpe(t):
                    all_ids.append(self.vocab.get(sub, self.unk_token_id))
        return torch.tensor(all_ids, dtype=torch.long)

# ---------------------------
# Simple GPT model (same for all experiments)
# ---------------------------
class GPTConfig:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.block_size = BLOCK_SIZE
        self.n_layers = N_LAYERS
        self.n_heads = N_HEADS
        self.n_embd = N_EMBD
        self.d_ff = D_FF
        self.dropout = DROPOUT

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.n_embd // config.n_heads
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).bool())

    def forward(self, x):
        B,T,C = x.shape
        q = self.Wq(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = self.Wk(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = self.Wv(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.mask[:T,:T], float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.out(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.n_embd),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B,T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device)).unsqueeze(0)
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# ---------------------------
# Data loading / batching
# ---------------------------
def load_streaming_data(lang="en", num_samples=NUM_SAMPLES, max_chars=MAX_CHARS):
    print(f"Streaming {lang} Wikipedia...")
    # Load specific language config
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    
    texts = []
    total = 0
    
    for i, ex in enumerate(dataset):
        if total >= max_chars: break
        
        raw_text = ex.get("text", "")
        
        # Chinese specific cleaning (Chinese has no spaces, different punctuation)
        if lang == "zh":
            # Remove simple whitespace but keep text
            cleaned = re.sub(r'\s+', '', raw_text) 
            if len(cleaned) < 50: continue
        else:
            # Existing English cleaning
            if len(raw_text.split()) < 50: continue
            cleaned = clean_text(raw_text)
            
        texts.append(cleaned)
        total += len(cleaned)
        
    return texts

def get_batch(train_data: torch.Tensor, val_data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    max_start = len(train_data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough data for block size")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------
# Training / evaluation
# ---------------------------
def estimate_loss(model, train_data, val_data, config, device, loss_fn):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(train_data if split=="train" else val_data, val_data, BATCH_SIZE, BLOCK_SIZE, device)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            losses.append(loss.item())
        avg = sum(losses)/len(losses)
        out[split] = avg
        out[f"{split}_ppl"] = math.exp(min(avg, 20))
    model.train()
    return out

def calculate_bpc(loss, total_tokens, total_chars):
    """
    Converts CrossEntropyLoss to Bits-Per-Character.
    loss: The raw validation loss (nats)
    total_tokens: Number of tokens in the validation set
    total_chars: Number of characters in the original validation text
    """
    total_nats = loss * total_tokens
    total_bits = total_nats / math.log(2)
    bpc = total_bits / total_chars
    return bpc

def run_experiment(mode: str, texts: List[str], run_name: str):
    """
    mode: "byte", "hf_bpe", "custom_bpe"
    texts: list of raw article strings to train tokenizers on
    run_name: directory basename to save results
    """
    print(f"\n=== Running experiment: {mode} ===")
    run_dir = os.path.join(SAVE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Consistent model config for all runs
    config = GPTConfig()
    config.vocab_size = VOCAB_SIZE
    config.block_size = BLOCK_SIZE

    # --- 1. Tokenization & Compression Ratio ---
    full_text_str = " ".join(texts)
    total_chars = len(full_text_str)
    
    if mode == "byte":
        tokenizer = ByteLevelTokenizer(max_length=BLOCK_SIZE)
        data_tensor = tokenizer.encode(full_text_str)
        vocab_size = 256 # The same for all byte-level-tokenized languages
        def encode_prompt(p):
            return torch.tensor(tokenizer.encode(p), dtype=torch.long).unsqueeze(0)

    elif mode == "hf_bpe":
        hf_save = os.path.join(run_dir, "hf_bpe.json")
        hf_tokenizer = train_hf_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE, save_path=hf_save)
        all_tokens = []
        for t in texts:
            enc = hf_tokenizer.encode(t)
            ids = enc.ids
            if len(ids) >= 1 and ids[0] == hf_tokenizer.token_to_id("[BOS]"): ids = ids[1:]
            if len(ids) >= 1 and ids[-1] == hf_tokenizer.token_to_id("[EOS]"): ids = ids[:-1]
            all_tokens.extend(ids)
        data_tensor = torch.tensor(all_tokens, dtype=torch.long)
        tokenizer = hf_tokenizer
        vocab_size = len(tokenizer.get_vocab())
        def encode_prompt(p):
            enc = tokenizer.encode(p)
            return torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0)

    elif mode == "custom_bpe":
        custom = CustomBPETokenizer(vocab_size=VOCAB_SIZE, max_length=BLOCK_SIZE)
        custom.train(texts, verbose=True)
        data_tensor = custom.encode_lm_data(full_text_str)
        tokenizer = custom
        vocab_size = custom.vocab_size
        def encode_prompt(p):
            ids = custom.encode(p, add_special_tokens=True)
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    else:
        raise ValueError("Unknown mode")

    # Calculate Compression Metrics
    total_tokens = len(data_tensor)
    tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
    print(f"Compression: {total_tokens} tokens / {total_chars} chars = {tokens_per_char:.4f} tokens/char")

    # train/val split
    n = int(0.9 * len(data_tensor))
    train_data = data_tensor[:n].to(torch.long)
    val_data = data_tensor[n:].to(torch.long)
    
    # Calculate val chars (approximate) for BPC
    val_chars = int(total_chars * 0.1)
    val_tokens = len(val_data)

    print(f"Data tokens: total={len(data_tensor)} train={len(train_data)} val={len(val_data)} vocab_size={vocab_size}")

    # Create model
    config.vocab_size = vocab_size
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Track metrics
    results = {
        "mode": mode,
        "vocab_size": vocab_size,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "tokens_per_char": tokens_per_char, # <--- Metric 1: Compression Ratio
        "val_data_tokens": val_tokens,      # <--- Fix for KeyError
        "val_chars_approx": val_chars,
        "train_losses": [],
        "val_losses": [],
        "normalized_train": [],
        "normalized_val": []
    }

    print("Training model...")
    pbar = tqdm(range(MAX_ITERS+1), desc=f"train-{mode}")
    for it in pbar:
        if it % EVAL_INTERVAL == 0:
            est = estimate_loss(model, train_data, val_data, config, DEVICE, loss_fn)
            train_loss = est["train"]
            val_loss = est["val"]
            norm_train = train_loss - math.log(vocab_size)
            norm_val = val_loss - math.log(vocab_size)
            
            results["train_losses"].append(train_loss)
            results["val_losses"].append(val_loss)
            results["normalized_train"].append(norm_train)
            results["normalized_val"].append(norm_val)
            
            # Calculate BPC live
            bpc = calculate_bpc(val_loss, val_tokens, val_chars)
            
            print(f"\nIter {it}: val_loss={val_loss:.4f} BPC={bpc:.4f}")
            pbar.set_postfix({"vr": f"{val_loss:.3f}", "bpc": f"{bpc:.3f}"})

        xb, yb = get_batch(train_data, val_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        logits = model(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # final eval
    final = estimate_loss(model, train_data, val_data, config, DEVICE, loss_fn)
    results["final_val_loss"] = final["val"]
    
    # --- 2. Inference Speed ---
    print("\nMeasuring Inference Speed...")
    model.eval()
    prompts = ["The future of", "Artificial intelligence will", "Climate change is"]
    
    total_gen_tokens = 0
    total_time = 0.0
    
    for p in prompts:
        try:
            enc = encode_prompt(p).to(DEVICE)
            start_t = time.time() # <--- Start Timer
            
            # Generate 50 tokens
            gen_len = 50
            sample = model.generate(enc, max_new_tokens=gen_len, temperature=0.8, top_k=40)[0]
            
            end_t = time.time()   # <--- End Timer
            
            # Calculate speed
            duration = end_t - start_t
            total_time += duration
            total_gen_tokens += gen_len
            
            decoded = tokenizer.decode(sample.tolist()) if hasattr(tokenizer, "decode") else tokenizer.decode(sample.tolist())
            print(f"Prompt: '{p}' -> {decoded[:100]}...")
            
        except Exception as e:
            print(f"Could not generate for prompt '{p}': {e}")

    # Calculate average speed
    if total_time > 0:
        tokens_per_sec = total_gen_tokens / total_time
        results["inference_speed_tokens_sec"] = tokens_per_sec # <--- Metric 2: Inference Speed
        print(f"Inference Speed: {tokens_per_sec:.2f} tokens/sec")
    else:
        results["inference_speed_tokens_sec"] = 0

    # save results
    fname = os.path.join(run_dir, f"results_{mode}.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {fname}")

    return results

# ---------------------------
# Main runner
# ---------------------------
def main():
    languages = ["en", "zh"]
    modes = ["byte", "custom_bpe", "hf_bpe"] 
    
    for lang in languages:
        texts = load_streaming_data(lang=lang, num_samples=NUM_SAMPLES, max_chars=MAX_CHARS)
        
        for mode in modes:
            run_name = f"{lang}_{mode}"
            print(f"--- Starting Run: {run_name} ---")
            
            results = run_experiment(mode, texts, run_name)
            
            # --- 3. BPC Performance (Final Calculation) ---
            # Now safe because 'val_data_tokens' is in results
            results['final_bpc'] = calculate_bpc(
                results['final_val_loss'], 
                results['val_data_tokens'], 
                results['val_chars_approx']
            )
            
            # Overwrite file with final BPC included
            with open(os.path.join(SAVE_DIR, f"final_{run_name}.json"), "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved final results to final_{run_name}.json (BPC: {results['final_bpc']:.4f})")

if __name__ == "__main__":
    main()
