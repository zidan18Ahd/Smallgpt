# TinyStories GPT (from scratch) — PyTorch

Train a tiny GPT-style language model **from scratch** on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset using:

* **Hugging Face `datasets`** for streaming and preprocessing
* **`tiktoken` (GPT‑2 encoding)** for tokenization
* **Pure PyTorch** Transformer (attention + MLP blocks)
* **Memory‑mapped** binary datasets (`train.bin`, `validation.bin`) for fast sampling
* Mixed precision (FP16/BF16) with **`torch.amp.autocast`** and **GradScaler**
* **Warmup + cosine decay** scheduler and **AdamW** optimizer with weight decay

> This README mirrors the exact pipeline used in the shared code/logs. It is meant to be copy‑paste runnable inside a notebook or a single Python file.

---

## 1) Environment & Requirements

Tested with the following (based on your logs):

* Python 3.11
* `datasets==4.0.0` (upgraded from 3.6.0)
* `tiktoken==0.9.0`
* `pyarrow>=15.0.0`
* `torch==2.6.0` (CUDA 12.x build in your logs)
* `numpy==1.26.4` (compatible with `tiktoken`/PyTorch)
* `pandas`, `tqdm`

**Install (typical notebook environment):**

```bash
pip install -U datasets tiktoken tqdm pyarrow pandas
# PyTorch: select the right command for your CUDA/CPU from https://pytorch.org/get-started/locally/
```

> ⚠️ **Dependency notes** (seen in your output):
>
> * Some environments preinstall GPU libraries (e.g., `nvidia-cublas-cu12 12.5.x`) that **don’t match** the exact versions PyTorch expects. If you want to avoid CUDA mismatch errors quickly, install the **CPU‑only** PyTorch build or ensure your CUDA component versions match the wheel you install.
> * `gcsfs` may require an exact `fsspec` pin. If you don’t use GCS, you can ignore `gcsfs` warnings or install a matching version (`pip install 'gcsfs==2025.3.0'`) when needed.

---

## 2) Project Structure

After running the preprocessing script below, you’ll have:

```
.
├── train.bin              # uint16 memmap of all training token ids
├── validation.bin         # uint16 memmap of all validation token ids
├── best_model_params.pt   # best checkpoint by val loss (saved during training)
└── README.md
```

---

## 3) Data: Load + Tokenize + Pack to .bin

```python
from datasets import load_dataset
import tiktoken, os, numpy as np
from tqdm.auto import tqdm

ds = load_dataset("roneneldan/TinyStories")
enc = tiktoken.get_encoding("gpt2")

# Map raw text → token ids (no special tokens)
def process(example):
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}

if not os.path.exists("train.bin"):
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
    )

    # Write memmaps for fast random slicing during training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16  # GPT‑2 vocab (50257) < 2**16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
```

**Why memmap?** You can sample any contiguous `block_size` window without loading the entire dataset into RAM.

---

## 4) Model: Tiny GPT (from scratch)

Core components (LayerNorm, CausalSelfAttention w/ FlashAttention path when available, MLP, Block, GPT) with **weight tying** and standard init. The config used in your run:

```python
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F, math

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
            )
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        return logits[:, [-1], :], None  # only last token when generating

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

**Default config** (from your run):

```python
config = GPTConfig(
    vocab_size=50257,  # GPT‑2 tokenizer size
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True,
)
model = GPT(config)
```

---

## 5) Batching from memmaps

```python
import numpy as np, torch

batch_size = 32
block_size = 128

def get_batch(split):
    data = np.memmap('train.bin' if split == 'train' else 'validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if torch.cuda.is_available():
        x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
    else:
        x, y = x.to('cpu'), y.to('cpu')
    return x, y
```

---

## 6) Training loop

```python
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from contextlib import nullcontext

learning_rate = 1e-3
max_iters = 20000
warmup_steps = 1000
min_lr = 5e-4
eval_iters = 500
batch_size = 32
block_size = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if device == 'cuda' else 'cpu'
dtype = 'bfloat16' if (device == 'cuda' and torch.cuda.is_bf16_supported()) else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)

scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch('train' if split=='train' else 'validation')
                with ctx:
                    logits, loss = model(X.to(device), Y.to(device))
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

best_val_loss = float('inf')
best_model_params_path = 'best_model_params.pt'
train_loss_hist, val_loss_hist = [], []

gradient_accumulation_steps = 32

for step in range(max_iters):
    if step % eval_iters == 0 and step != 0:
        losses = estimate_loss(model)
        print(f"Step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}; lr={optimizer.param_groups[0]['lr']:.5f}")
        train_loss_hist.append(losses['train'].item())
        val_loss_hist.append(losses['val'].item())
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_params_path)

    X, Y = get_batch('train')
    X, Y = X.to(device), Y.to(device)

    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    # Only step optimizer & scheduler when we actually update weights
    do_step = ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == max_iters)
    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
```

**Observed losses** (your run excerpt):

```
Step 1000:  train 5.1067, val 5.1030
Step 5000:  train 2.9833, val 2.9839
Step 10000: train 2.4224, val 2.4259
Step 15000: train 2.1876, val 2.1857
Step 19500: train 2.0745, val 2.0828
```

> 💡 **Tip:** The scheduler warning you saw ("`lr_scheduler.step()` before `optimizer.step()`") happens when the scheduler is advanced on steps where the optimizer didn’t step (due to gradient accumulation). The loop above calls `scheduler.step()` **only when** we call `optimizer.step()`.

---

## 7) Inference: Generate Tiny Stories

```python
import torch, tiktoken
enc = tiktoken.get_encoding('gpt2')

# Load best checkpoint
ckpt = torch.load('best_model_params.pt', map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
model.to(device)

# Helper: encode/decode

def encode(text):
    return torch.tensor([enc.encode_ordinary(text)], dtype=torch.long, device=device)

def decode(ids):
    return enc.decode(ids)

# Prompt and generate
prompt = "Once upon a time,"
idx = encode(prompt)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=128, temperature=0.8, top_k=200)

text = decode(out[0].tolist())
print(text)
```

**Sampling knobs**

* `temperature < 1.0` → more deterministic; `> 1.0` → more diverse
* `top_k` limits the candidate pool per step (e.g., 50–200 is common)

---

## 8) Reproducibility

```python
import torch, random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

For exact reproducibility across GPUs/hardware, consider setting

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 9) Tips, Gotchas & Troubleshooting

* **CUDA component mismatch** (logs show cuBLAS/CUPTI/etc. versions differ from what PyTorch expects):

  * Quick fix: use **CPU‑only** PyTorch wheel if GPU isn’t required.
  * Or install a PyTorch wheel that matches your CUDA runtime (see the official selector).
* **`fsspec`/`gcsfs` conflicts**: If `gcsfs` is present, it may require `fsspec==2025.3.2`. Either pin both to matching versions or uninstall `gcsfs` if unused.
* **Scheduler warning**: Only call `scheduler.step()` on iterations that also call `optimizer.step()` (see training loop above).
* **OOM or slow training**:

  * Reduce `n_layer`, `n_head`, `n_embd`, `block_size`, or increase gradient accumulation.
  * Use BF16 if supported: `torch.cuda.is_bf16_supported()`.
* **Tokenizer limits**: We store ids as `uint16` since GPT‑2 vocab size 50,257 < 2^16.
* **Memmap safety**: Recreate the memmap in each batch (as done) to avoid file handle leaks in some notebook runtimes.

---

## 10) Configuration Summary

| Key          | Value                                                       |
| ------------ | ----------------------------------------------------------- |
| `vocab_size` | 50257 (GPT‑2)                                               |
| `block_size` | 128                                                         |
| `n_layer`    | 6                                                           |
| `n_head`     | 6                                                           |
| `n_embd`     | 384                                                         |
| `dropout`    | 0.1                                                         |
| Optimizer    | AdamW (`betas=(0.9, 0.95)`, `weight_decay=0.1`, `eps=1e-9`) |
| Scheduler    | 1k warmup → cosine to `min_lr=5e-4` over 20k steps          |
| Precision    | BF16 if supported, else FP16 with GradScaler                |
| Batch size   | 32 tokens windows (128) × grad accum 32                     |

---

## 11) License & Attribution

* Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (credit to the authors)
* This repository’s GPT implementation is educational/minimal and inspired by common tiny‑GPT patterns.

If you use this README or code, please consider citing the TinyStories dataset and linking back to Hugging Face.
