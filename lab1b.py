import comet_ml
COMET_API_KET = "HTd95MIewu28w3RQOKl6FJg62"
import torch
import torch.nn as nn
import torch.optim as optim

import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from IPython.display import Audio
from tqdm import tqdm
from scipy.io.wavfile import write

songs = mdl.lab1.load_training_data()

#example_song = songs[15]
# print("\nExample song: ")
# print(example_song)

#mdl.lab1.play_song(example_song)
## Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

#Find all unique character in the joined string
vocab = sorted(set(songs_joined))
#print("There are", len(vocab), "unique characters in the dataset")

### Define numerical representation of text ###
# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
# we can evaluaate char2idx["d"]
char2idx = {u: i for i, u in enumerate(vocab)}

# Create a mapping form indices to characters. This is
# the inverse of char2idx and allows us to convert back
# from unique index to the character in our vocabulary.

idx2char = np.array(vocab)

# print('{')
# for char, _ in zip(char2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
# print('  ...\n')

### Vectorize the songs string ###

# '''TODO: Write a function to convert the all songs string to a vectorized
#     (i.e., numeric) representation. Use the appropriate mapping
#     above to convert from vocab characters to the corresponding indices.

#   NOTE: the output of the `vectorize_string` function
#   should be a np.array with `N` elements, where `N` is
#   the number of characters in the input string


def vectorize_string(string):
    return np.array([char2idx[char] for char in string])


vectorized_songs = vectorize_string(songs_joined)

# print('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# # check that vectorized_songs is a numpy array
# assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"

### Batch definition to create training examples ###
def get_batch(vectorized_songs, seq_lenght, batch_size):
    # the lenght of the vectorized songs string
    n = vectorized_songs.shape[0]-1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_lenght, batch_size)

    '''TODO: construct a list of input sequences for the training batch'''
    # input_batch = vectorized_songs[idx - n: idx]
    input_batch = [vectorized_songs[i : i + seq_lenght] for i in idx]

    #output_batch = vectorized_songs[idx - n + 1: idx + 1]
    output_batch = [vectorized_songs[i + 1: i + seq_lenght + 1] for i in idx]

    # Convert the input and output batches to tensors
    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)

    return x_batch, y_batch

# Perform some simple tests to make sure your batch function is working properly


# test_args = (vectorized_songs, 10, 2)
# x_batch, y_batch = get_batch(*test_args)
# assert x_batch.shape == (2, 10), "x_batch shape is incorrect"
# assert y_batch.shape == (2, 10), "y_batch shape is incorrect"
# print("Batch function works correctly!")

# x_batch, y_batch = get_batch(vectorized_songs, seq_lenght=5, batch_size=1)

# for i, (input_idx, target_idx) in enumerate(zip(x_batch[0], y_batch[0])):
#     print("Step {:3d}".format(i))
#     print(" input: {} ({:s})".format(input_idx, repr(idx2char[input_idx.item()])))
#     print(" expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx.item()])))

### Defining the RNN Model ###

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # Define each of the network layers
        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        '''TODO: Layer 2: LSTM with hidden_size `hidden_size`. note: number of layers defaults to 1.
         Use the nn.LSTM() module from pytorch.'''
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)

        '''TODO: Layer 3: Linear (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size.'''
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state with zeros
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)

        x = x.permute(1, 0, 2)
        out, state = self.lstm(x, state)
        out = out.permute(1, 0, 2)
        out = self.fc(out)
        return out if not return_state else (out, state)


vocab_size = len(vocab)
# embedding_dim = 256
# hidden_size = 1024
# batch_size = 8

device = torch.device("cpu")

#model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)

# print(model)

# # test the model with some sample data
# x, y = get_batch(vectorized_songs, seq_lenght=100, batch_size=32)
# x = x.to(device)
# y = y.to(device)

# pred = model(x)
# print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
# print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

# sampled_indices = torch.multinomial(torch.softmax(pred[0], dim=-1),num_samples=1)
# sampled_indices = sampled_indices.squeeze(-1).cpu().numpy()
# sampled_indices

# print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

### Defining the loss function ###

# '''TODO: define the compute_loss function to compute and return the loss between
#     the true labels and predictions (logits). '''

cross_entropy = nn.CrossEntropyLoss()  # instantiates the function

def compute_loss(labels, logits):
    """
    Inputs:
        labels: (batch_size, sequence_length)
        logits: (batch_size, sequence_length, vocab_size)

    Output:
        loss: scalar cross entropy loss over the batch and sequence length
    """

    # Batch the labels so that the shape of the labels should be (B * L,)
    batched_labels = labels.view(-1)

    ''' TODO: Batch the logits so that the shape of the logits should be (B * L, V) '''
    batched_logits = logits.view(-1, logits.size(-1)) # TODO

    '''TODO: Compute the cross-entropy loss using the batched  next characters and predictions'''
    loss = cross_entropy(batched_logits, batched_labels) # TODO
    return loss

### compute the loss on the predictions from the untrained model from earlier. ###
#y.shape  # (batch_size, sequence_length)
#pred.shape  # (batch_size, sequence_length, vocab_size)

'''TODO: compute the loss using the true next characters from the example batch
    and the predictions from the untrained model several cells above'''
#example_batch_loss = compute_loss(y, pred) # TODO

# print(f"Prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")
# print(f"scalar_loss:      {example_batch_loss.mean().item()}")


### Hyperparameter setting and optimization ###

vocab_size = len(vocab)

# Model parameters:
params = dict(
    num_training_iterations = 3000, #Increase this to train longer
    batch_size = 4, # Experiment between 1 and 64
    seq_length = 100, # Experiment between 50 and 500
    learning_rate = 5e-3,
    embedding_dimm = 64,
    hidden_size = 128, # Experiment beetwen 1 and 2048
)

# Checkpoint Location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

### Create a Comet experiment to track out training run ###

def create_experiment():
    # end any prior experiments
    if 'experiment' in locals():
        experiment.end()

    #initiate the comet experiment for tracking
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KET,
        project_name="6S191_Lab1_Part2")
    
    # Log our hyperparameters, defined above, to the experiment
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()

    return experiment

### Define optimizer and training operation ###

'''TODO: instantiate a new LSTMModel model for training using the hyperparameters
    created above.'''
model = LSTMModel(
    vocab_size=vocab_size,
    embedding_dim=params["embedding_dimm"],
    hidden_size=params["hidden_size"]
)

model.to(device)

'''TODO: instantiate an optimizer with its learning rate.
  Checkout the PyTorch website for a list of supported optimizers.
  https://pytorch.org/docs/stable/optim.html
  Try using the Adam optimizer to start.'''
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_step(x, y):
    # Set the model's mode to train
    model.train()

    #Zero gradients for every step
    optimizer.zero_grad()

    #Forward pass
    y_hat = model(x)

    #Compute the loss
    loss = compute_loss(y, y_hat)

    loss.backward()

    optimizer.step()

    return loss

##################
# Begin training!#
##################

# history = []
# plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
# #experiment = create_experiment()

# if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
# for iter in tqdm(range(params["num_training_iterations"])):

#     # Grab a batch and propagate it through the network
#     x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])

#     # Convert numpy arrays to PyTorch tensors
#     x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
#     y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

#     # Take a train step
#     loss = train_step(x_batch, y_batch)

#     # Log the loss to the Comet interface
#     experiment.log_metric("loss", loss.item(), step=iter)

#     # Update the progress bar and visualize within notebook
#     history.append(loss.item())
#     plotter.plot(history)

#     # Save model checkpoint
#     if iter % 100 == 0:
#         torch.save(model.state_dict(), checkpoint_prefix)

# Save the final trained model
#torch.save(model.state_dict(), checkpoint_prefix)
#experiment.flush()

def generate_text(model, start_string, generation_length=1000):
    # Ustaw model w tryb ewaluacji (nie uczymy go)
    model.eval()

    # Zamień startowy string na wektor indeksów (używając wcześniej zdefiniowanego słownika)
    input_idx = np.array([char2idx[s] for s in start_string], dtype=np.int64)
    input_idx = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)

    # Zainicjalizuj stan ukryty LSTM
    state = model.init_hidden(batch_size=1, device=device)

    # Bufor na wygenerowany tekst
    text_generated = []

    tqdm._instances.clear()  # czyści progress bary, żeby się nie duplikowały

    for i in tqdm(range(generation_length)):
        # Przekaż dane przez model
        predictions, state = model(input_idx, state, return_state=True)

        # Weź tylko ostatnią predykcję z sekwencji (bo model przewiduje kolejny znak)
        predictions = predictions[:, -1, :]

        # Przekształć logits w rozkład prawdopodobieństwa
        probs = torch.softmax(predictions, dim=-1)

        # Wylosuj kolejny znak wg rozkładu prawdopodobieństwa
        next_idx = torch.multinomial(probs, num_samples=1).item()

        # Zamień indeks z powrotem na znak
        next_char = idx2char[next_idx]

        # Dodaj znak do wygenerowanego tekstu
        text_generated.append(next_char)

        # Zaktualizuj wejście (kolejny krok LSTM dostaje ostatni przewidziany znak)
        input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return start_string + ''.join(text_generated)


generated_text = generate_text(model, start_string="X:1\nT:Generated Tune\nM:4/4\nL:1/8\nK:C\n", generation_length=1000)

### Play back generated songs ###

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

waveform = mdl.lab1.play_song(generated_songs[0])
Audio(waveform)
#os.system("timidity example.mid -Ow -o example.wav")


    # If its a valid song (correct syntax), lets play it!
    # if waveform:
    #     print("Generated song", i)
    #     ipythondisplay.display(waveform)

    #     numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
    #     wav_file_path = f"output_{i}.wav"
    #     write(wav_file_path, 88200, numeric_data)
