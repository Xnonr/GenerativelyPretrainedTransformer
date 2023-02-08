# Importing Dependencies
import torch
import torch.nn as nn
import wget

from os.path import exists
from torch.nn import functional

# Hyperparameters & Constants
batch_size = 64
block_size = 256
#device = 'cuda' if torch.cuda.is_available() else 'cpu' # CUDA
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # Mac M1
dropout = 0.2
evaluation_interval = 500
evaluation_iterations = 200
learning_rate = 3e-4
maximum_iterations = 5000
number_of_embedding_dimensions = 384
number_of_heads = 6
number_of_layers = 6
tiny_shakespeare_dataset_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# Setting Seed
torch.manual_seed(1337)

# Printing Device
print(f'Running on: {device}')

# Dowloading Tiny Shakespeare Dataset
if exists('input.txt') == False:
    print('Downloading the Tiny Shakespeare dataset.')
    tiny_shakespeare_dataset = wget.download(tiny_shakespeare_dataset_url)
else:
    print('The Tiny Shakespeare dataset has already been downloaded.')

# Reading Dataset
with open('input.txt', 'r', encoding = 'utf-8') as file:
    dataset_text = file.read()

# Gathering Dataset Details
unique_characters = sorted(list(set(dataset_text)))
vocabulary_size = len(unique_characters)

# Constructing Character Integer Mappings
string_to_integer = {character:integer for integer, character in enumerate(unique_characters)}
integer_to_string = {integer:character for integer, character in enumerate(unique_characters)}

# Constructing Encoder & Decoder
encoder = lambda string: [string_to_integer[character] for character in string]
decoder = lambda list_of_integers: ''.join([integer_to_string[integer] for integer in list_of_integers])

# Enconding Dataset
encoded_dataset = torch.tensor(encoder(dataset_text), dtype = torch.long)

# Designating Training & Testing Splits
split_index = int(0.9 * len(encoded_dataset))
training_data = encoded_dataset[:split_index]
validation_data = encoded_dataset[split_index:]

# Batched Dataset Loading
def get_batch(split):
    data = training_data if split == 'training' else validation_data
    index = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in index])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in index])
    x, y = x.to(device), y.to(device)

    return x, y

# Loss Estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['training', 'validation']:
        losses = torch.zeros(evaluation_iterations)
        
        for k in range(evaluation_iterations):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()

    model.train()
    
    return out

# Single Head Self Attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(number_of_embedding_dimensions, head_size, bias = False)
        self.query = nn.Linear(number_of_embedding_dimensions, head_size, bias = False)
        self.value = nn.Linear(number_of_embedding_dimensions, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, time, channels = x.shape
        keys = self.key(x)
        queries = self.query(x)
        weights = queries @ keys.transpose(-2, -1) * channels ** -0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        weights = functional.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        values = self.value(x)
        output = weights @ values

        return output

# Multiple Head Self Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, number_of_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(number_of_heads)])
        self.projection = nn.Linear(number_of_embedding_dimensions, number_of_embedding_dimensions)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim = -1)
        output = self.dropout(self.projection(output))

        return output

# Multi Layer Perceptron
class FeedForward(nn.Module):
    def __init__(self, number_of_embedding_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_embedding_dimensions, 4 * number_of_embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4 * number_of_embedding_dimensions, number_of_embedding_dimensions),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# Transformer Block
class Block(nn.Module):
    def __init__(self, number_of_embedding_dimensions, number_of_heads):
        super().__init__()
        head_size = number_of_embedding_dimensions // number_of_heads
        self.self_attention = MultiHeadAttention(number_of_heads, head_size)
        self.feed_forward = FeedForward(number_of_embedding_dimensions)
        self.layer_normalization1 = nn.LayerNorm(number_of_embedding_dimensions)
        self.layer_normalization2 = nn.LayerNorm(number_of_embedding_dimensions)
    
    def forward(self, x):
        x = x + self.self_attention(self.layer_normalization1(x))
        x = x + self.feed_forward(self.layer_normalization2(x))
        
        return x

# Highly Simplified Bigram Language Neural Network Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, number_of_embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, number_of_embedding_dimensions)
        
        self.blocks = nn.Sequential(*[Block(number_of_embedding_dimensions, number_of_heads = number_of_heads) for _ in range(number_of_layers)])
        
        self.layer_normalization = nn.LayerNorm(number_of_embedding_dimensions)
        self.language_modeling_head = nn.Linear(number_of_embedding_dimensions, vocabulary_size)
        
    def forward(self, index, targets = None):
        batch, time = index.shape
        token_embeddings = self.token_embedding_table(index)
        position_embeddings = self.position_embedding_table(torch.arange(time, device = device))
        token_id_position_embeddings = token_embeddings + position_embeddings
        token_id_position_embeddings = self.blocks(token_id_position_embeddings)
        token_id_position_embeddings = self.layer_normalization(token_id_position_embeddings)
        logits = self.language_modeling_head(token_id_position_embeddings)

        if targets is None:
            loss = None
        
        else:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_condition = index[:, -block_size:]
            logits, loss = self(index_condition)
            logits = logits[:, -1, :]
            probabilities = functional.softmax(logits, dim = -1)
            index_next = torch.multinomial(probabilities, num_samples = 1)
            index = torch.cat((index, index_next), dim = 1)
            
        return index

# Constructing Neural Network Model
model = BigramLanguageModel(vocabulary_size)
model = model.to(device)

# Constructing PyTorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# Model Training Loop
for iteration in range(maximum_iterations):
    
    if iteration % evaluation_interval == 0:
        losses = estimate_loss()
        print(f"Step {iteration}: Training Loss = {losses['training']:.4f} | Validation Loss = {losses['validation']:.4f}")
    
    xb, yb = get_batch('training')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# Generating Predictions Via Model
context = torch.zeros((1, 1), dtype = torch.long, device = device)

# Printing Results
#print(decoder(model.generate(context, max_new_tokens = 10000)[0].tolist()))

open('results.txt', 'w').write(decoder(model.generate(context, max_new_tokens = 10000)[0].tolist()))