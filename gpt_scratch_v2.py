import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 64 # how many individual sequences to train on in parallel
block_size = 128 # the number of tokens in the sequence
max_iters = 5000
eval_interval = max_iters // 10
lr = 3e-4
eval_iters = 500
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.25
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

data_names = ["shakespeare.txt", "harry-potter.txt"]
data_endpoints = ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", ""]
data_n = 1

if not os.path.exists("data-" + data_names[data_n]):
	os.system(f'curl {data_endpoints[data_n]} -o data-{data_names[data_n]}')

with open(f"data-{data_names[data_n]}", "r") as file:
	text = file.read()

# all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# make mappings
s_to_i = {s:i for i,s in enumerate(chars)}
i_to_s = {i:s for i,s in enumerate(chars)}
# make encode and decode functions
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda x: ''.join([i_to_s[i] for i in x])

# train and test splits
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# get random batch function
def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,)) # choose batch_size starting indices at random
	x = torch.stack([data[i:i+block_size] for i in ix]) # construct the input sequence
	y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # construct the target sequence
	x, y = x.to(device), y.to(device)
	return x, y

# estimate loss
@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			x, y = get_batch(split)
			_, loss = model(x, y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

class Head(nn.Module):
	""" one head of self-attention """
	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x) # (B, T, C)
		q = self.query(x) # (B, T, C)
		# compute scaled dot-product attention
		wei = q @ k.transpose(-2, -1) + C**-0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		wei = F.softmax(wei, dim=-1) # (B, T, T)
		wei = self.dropout(wei)
		# perform weighted aggregation of values
		v = self.value(x) # (B, T, C)
		out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
		return out

class MultiHeadAttention(nn.Module):
	""" multiple heads of self-attention in parallel """
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.proj(out)
		out = self.dropout(out)
		return out

class FeedForward(nn.Module):
	""" a simple linear layer follewed by a non-linearity """
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout)
		)
	
	def forward(self, x):
		return self.net(x)

class Block(nn.Module):
	""" Transformer block: communication followed by computation """
	def __init__(self, n_embd, n_heads):
		super().__init__()
		head_size = n_embd // n_heads
		self.sa_heads = MultiHeadAttention(n_heads, head_size)
		self.ffwd = FeedForward(n_embd)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)
	
	def forward(self, x):
		x = x + self.sa_heads(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x

# define model
class BigramLanguageModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
		self.ln_f = nn.LayerNorm(n_embd) # final layer norm
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def forward(self, idx, targets=None):
		B, T = idx.shape
		
		# idx and targets are both of shape (B, T)
		token_emb = self.token_embedding_table(idx) # (B, T, C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
		x = token_emb + pos_emb # (B, T, C)
		x = self.blocks(x) # (B, T, C)
		x = self.ln_f(x) # (B, T, C)
		logits = self.lm_head(x) # (B, T, vocab_size)
		
		# if we don't have targets, skip loss calculation
		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx is of shape [batch_size, block_size], array of indices in the current context
		# max_new_tokens is the maximum number of new tokens to generate
		for _ in range(max_new_tokens):
			# crop idx to last block_size tokens
			idx_cond = idx[:, -block_size:]
			# get predictions for the next token
			logits, _ = self(idx_cond) # loss is not needed for generation (hence _)
			# only look at the last time step
			logits = logits[:, -1, :] # becomes shape [batch_size, vocab_size]
			# use softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # shape [batch_size, vocab_size]
			# sample from the distribution
			next_token = torch.multinomial(probs, num_samples=1) # shape [batch_size, 1]
			# append to the context
			idx = torch.cat((idx, next_token), dim=1) # shape [batch_size, block_size+1]
		return idx

model = BigramLanguageModel()
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
	
	# evaluate loss every eval_interval iterations
	if iter % eval_interval == 0 or iter == max_iters - 1:
		losses = estimate_loss()
		print(f'Step {iter} | Train loss: {losses["train"]:.4f} | Val loss: {losses["val"]:.4f}')
	
	# sample random batch
	xb, yb = get_batch('train')

	# evaluate loss
	_, loss = model(xb, yb) # can throw out logits here because we only need the loss
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()

# generate some text
idx = torch.zeros((1, 1), dtype=torch.int64, device=device)
generation = m.generate(idx, max_new_tokens=5000)[0]
print(decode(generation.tolist())[:300])

# make file and save generation to it
with open(f"gen-{data_names[data_n]}", "w") as file:
	file.write(decode(generation.tolist()))