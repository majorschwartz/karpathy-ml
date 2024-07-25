import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = max_iters // 10
lr = 1e-2
eval_iters = 200
n_embd = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

if not os.path.exists("shakespeare.txt"):
	os.system('curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o shakespeare.txt')

with open("shakespeare.txt", "r") as file:
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
	pass

# define model
class BigramLanguageModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def forward(self, idx, targets=None):
		B, T = idx.shape
		# idx and targets are both of shape [batch_size, block_size]
		token_emb = self.token_embedding_table(idx) # shape [batch_size, block_size, n_embd]
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # shape [block_size, n_embd]
		x = token_emb + pos_emb # shape [batch_size, block_size, n_embd]
		logits = self.lm_head(x) # shape [batch_size, block_size, vocab_size]
		
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
			# get predictions for the next token
			logits, _ = self(idx) # loss is not needed for generation (hence _)
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
	if iter % eval_interval == 0:
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
generation = m.generate(idx, max_new_tokens=300)[0]
print(decode(generation.tolist()))