# Importing necessary libraries
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Transformer
import math
import numpy as np

###### Settings ######

# Language to convert to (only have one set to true)
engToFren = False
frenToEng = True

# Training parameters
epochs = 600
learning_rate = 0.001
p_dropout = 0.3
examplesToPrint = 5 # How many example input/output to print after validation
nhead = 2
num_layers = 1

# How many epochs before it will save training loss to log. 0 To save every epoch.
# It saves every x epochs including the first one.
saveToLogDelay = 20

# Create a file for saving outputs
training_log_file = open('graphs/language_seq_to_seq/Sequence_to_sequence_prediction_'+ str(time.time()) +'.log', 'w')

# Save parameters to log
training_log_file.write(f'Sequence to sequence prediction' + '\n\n')
training_log_file.flush()
training_log_file.write(f'Epoches: {epochs}, Learning Rate: {learning_rate}, Dropout: {p_dropout}, Examples To Print: {examplesToPrint}' + '\n')
training_log_file.flush()
training_log_file.write(f'number of heads: {nhead}, number of layers: {num_layers}' + '\n')
training_log_file.flush()
training_log_file.write(f'Translating from English to French: {engToFren}, Translating from French To English: {frenToEng}' + '\n')
training_log_file.flush()


### Dataset and loader provided by class ###

# Vocabulary class to handle mapping between words and numerical indices
class Vocabulary:
    def __init__(self):
        # Initialize dictionaries for word to index and index to word mappings
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.word_count = {}  # Keep track of word frequencies
        self.n_words = 3  # Start counting from 3 to account for special tokens

    def add_sentence(self, sentence):
        # Add all words in a sentence to the vocabulary
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        # Add a word to the vocabulary
        if word not in self.word2index:
            # Assign a new index to the word and update mappings
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            # Increment word count if the word already exists in the vocabulary
            self.word_count[word] += 1

def tokenize_and_pad(sentences, vocab, max_length):
    # Calculate the maximum sentence length for padding
    #max_length = max(len(sentence.split(' ')) for sentence in sentences) + 2  # +2 for SOS and EOS tokens
    tokenized_sentences = []
    for sentence in sentences:
        # Convert each sentence to a list of indices, adding SOS and EOS tokens
        tokens = [vocab.word2index["<SOS>"]] + [vocab.word2index[word] for word in sentence.split(' ')] + [vocab.word2index["<EOS>"]]
        # Pad sentences to the maximum length
        padded_tokens = tokens + [vocab.word2index["<PAD>"]] * (max_length - len(tokens))
        tokenized_sentences.append(padded_tokens)
    return torch.tensor(tokenized_sentences, dtype=torch.long)

# Custom Dataset class for English to French sentences
class EngFrDataset(Dataset):
    def __init__(self, pairs):
        self.eng_vocab = Vocabulary()
        self.fr_vocab = Vocabulary()
        self.pairs = []

        # Process each English-French pair
        for eng, fr in pairs:
            self.eng_vocab.add_sentence(eng)
            self.fr_vocab.add_sentence(fr)
            self.pairs.append((eng, fr))

        # Separate English and French sentences
        self.eng_sentences = [pair[0] for pair in self.pairs]
        self.fr_sentences = [pair[1] for pair in self.pairs]
        
        # Tokenize and pad sentences
        max_length = max(len(sentence.split(' ')) for sentence in self.eng_sentences) + 2  # +2 for SOS and EOS tokens
        temp = max(len(sentence.split(' ')) for sentence in self.fr_sentences) + 2  # +2 for SOS and EOS tokens

        if(temp > max_length):
            max_length = temp

        self.eng_tokens = tokenize_and_pad(self.eng_sentences, self.eng_vocab, max_length)
        self.fr_tokens = tokenize_and_pad(self.fr_sentences, self.fr_vocab, max_length)

        # Define the embedding layers for English and French
        self.eng_embedding = torch.nn.Embedding(self.eng_vocab.n_words, 100)  # Embedding size = 100
        self.fr_embedding = torch.nn.Embedding(self.fr_vocab.n_words, 100)    # Embedding size = 100

    def __len__(self):
        # Return the number of sentence pairs
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get the tokenized and padded sentences by index
        eng_tokens = self.eng_tokens[idx]
        fr_tokens = self.fr_tokens[idx]
        # Lookup embeddings for the tokenized sentences
        eng_emb = self.eng_embedding(eng_tokens)
        fr_emb = self.fr_embedding(fr_tokens)
        return eng_tokens, fr_tokens, eng_emb, fr_emb

# Sample dataset of English-French sentence pairs
english_to_french = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle français couramment"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent des musées"),
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
    ("He sings beautifully", "Il chante magnifiquement"),
    ("They swim in the pool", "Ils nagent dans la piscine"),
    ("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
    ("She teaches English at school", "Elle enseigne l'anglais à l'école"),
    ("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
    ("He paints landscapes", "Il peint des paysages"),
    ("They laugh at the joke", "Ils rient de la blague"),
    ("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
    ("She runs in the park", "Elle court dans le parc"),
    ("We travel by train", "Nous voyageons en train"),
    ("He writes a letter", "Il écrit une lettre"),
    ("They read books at the library", "Ils lisent des livres à la bibliothèque"),
    ("The baby cries", "Le bébé pleure"),
    ("She studies hard for exams", "Elle étudie dur pour les examens"),
    ("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
    ("He fixes the car", "Il répare la voiture"),
    ("They drink coffee in the morning", "Ils boivent du café le matin"),
    ("The sun sets in the evening", "Le soleil se couche le soir"),
    ("She dances at the party", "Elle danse à la fête"),
    ("We play music at the concert", "Nous jouons de la musique au concert"),
    ("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
    ("They study French grammar", "Ils étudient la grammaire française"),
    ("The rain falls gently", "La pluie tombe doucement"),
    ("She sings a song", "Elle chante une chanson"),
    ("We watch a movie together", "Nous regardons un film ensemble"),
    ("He sleeps deeply", "Il dort profondément"),
    ("They travel to Paris", "Ils voyagent à Paris"),
    ("The children play in the park", "Les enfants jouent dans le parc"),
    ("She walks along the beach", "Elle se promène le long de la plage"),
    ("We talk on the phone", "Nous parlons au téléphone"),
    ("He waits for the bus", "Il attend le bus"),
    ("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
    ("The stars twinkle at night", "Les étoiles scintillent la nuit"),
    ("She dreams of flying", "Elle rêve de voler"),
    ("We work in the office", "Nous travaillons au bureau"),
    ("He studies history", "Il étudie l'histoire"),
    ("They listen to the radio", "Ils écoutent la radio"),
    ("The wind blows gently", "Le vent souffle doucement"),
    ("She swims in the ocean", "Elle nage dans l'océan"),
    ("We dance at the wedding", "Nous dansons au mariage"),
    ("He climbs the mountain", "Il gravit la montagne"),
    ("They hike in the forest", "Ils font de la randonnée dans la forêt"),
    ("The cat meows loudly", "Le chat miaule bruyamment"),
    ("She paints a picture", "Elle peint un tableau"),
    ("We build a sandcastle", "Nous construisons un château de sable"),
    ("He sings in the choir", "Il chante dans le chœur")
]

print("Initializing dataset")
# Initialize the dataset and DataLoader
dataset = EngFrDataset(english_to_french)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
print("Dataset loaded")

# Input and output size are based on the total number of possible words in each language
if engToFren:
    input_size = dataset.eng_vocab.n_words
    output_size = dataset.fr_vocab.n_words
elif frenToEng:
    input_size = dataset.fr_vocab.n_words
    output_size = dataset.eng_vocab.n_words

# Max sentence length. It doesn't matter if the number is higher than the sentence with 
# the max length, as it will pad the end. It just can not be shorter.
hidden_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using Device: {device}')

training_log_file.write(f'Input Size: {input_size}, Hidden Size: {hidden_size}, Output Size: {output_size}' + '\n')
training_log_file.flush()
training_log_file.write(f'Using Device: {device}' + '\n')
training_log_file.flush()



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded)
        #output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return transformer_output[:,-1,:]
    
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers,num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, target_tensor):
        # Initialize decoder input with SOS_token
        #decoder_input = torch.zeros(1, target_tensor.size(0), dtype=torch.long, device=device)
        
        # Embed decoder input and apply positional encoding
        embedding = self.embedding(target_tensor)
        embedding = self.pos_encoder(embedding)

        # Decode using the transformer decoder
        decoder_output = self.transformer_decoder(embedding, encoder_output)

        # Apply linear layer and softmax
        output = self.softmax(self.linear(decoder_output))

        return output


class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self.encoder_embedding = nn.Embedding(input_size, hidden_size)
        self.decoder_embedding = nn.Embedding(output_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
    def forward(self, input_tensor, output_tensor):
        encoded_input = self.encoder_embedding(input_tensor)
        encoded_input = self.pos_encoder(encoded_input)

        encoded_output = self.decoder_embedding(output_tensor)
        encoded_output = self.pos_encoder(encoded_output)

        return encoded_input, encoded_output
    
# Code from the pytorch tutorial found here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = p_dropout, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.unsqueeze(1) # add the batch size of 1 into the middle
        x = x + self.pe[:x.size(0)]
        x = x.squeeze(1)
        #return self.dropout(x)
        return x

class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.encode = Encoding()
        self.t = Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_layers,num_decoder_layers=num_layers,dropout=p_dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input, output):
        encoded_input, encoded_output = self.encode(input, output)
        decoder_output = self.t(encoded_input, encoded_output)
        out = self.softmax(self.linear(decoder_output))
        return out


#encoder = Encoder().to(device)
#decoder = Decoder().to(device)
transformer = Trans().to(device)

# Save how many parameters are in the model
total_params = sum(p.numel() for p in transformer.parameters())
print(f'Total number of parameters in the model: {total_params}')
training_log_file.write(f'Number of parameters in model: {total_params}' + '\n')
training_log_file.flush()

# Initializing optimizers for both encoder and decoder with Stochastic Gradient Descent (SGD)
#encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
transformer_optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)

def train(input_tensor, target_tensor):
    # Clear gradients for optimizers
    #encoder_optimizer.zero_grad()
    #decoder_optimizer.zero_grad()
    transformer_optimizer.zero_grad()


    # Encode input sequence
    #encoder_output = encoder(input_tensor)gue

    # Decode the input sequence using the decoder
    #decoder_output = decoder(encoder_output, target_tensor)
    decoder_output = transformer(input_tensor, target_tensor)
    #print(guesses_index[0,1])
    # Create a tensor where each element in the final dimension contains the max value
    #guesses = max_values.unsqueeze(-1).expand_as(decoder_output)
    #target_tensor = target_tensor.type(torch.LongTensor).to(device)
    # Calculate loss
    loss = criterion(decoder_output, target_tensor)

    # Backpropagation
    loss.backward()

    # Update encoder and decoder parameters
    #encoder_optimizer.step()
    #decoder_optimizer.step()
    transformer_optimizer.step()

    return loss.item()


def evaluate_and_show_examples():
    # Switch model to evaluation mode
    transformer.eval()
    
    total_loss = 0
    correct_predictions = 0
    print('Starting Evaluation' + '\n')
    print('Example Predictions And Targets' + '\n')
    training_log_file.write('Starting Evaluation\n\nExample Predictions And Targets' + '\n')
    training_log_file.flush()  

    start_time = time.time()
    # No gradient calculation
    with torch.no_grad():
        for i, (input_tensor, target_tensor, _, _) in enumerate(dataloader):
            # If doing french to english, swap input and target
            if frenToEng:
                temp = input_tensor
                input_tensor = target_tensor
                target_tensor = temp

            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            
            #encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            loss = 0

            # Encoding step
            #for ei in range(input_length):
            #    encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

            if frenToEng:
                SOS_token = dataset.eng_vocab.word2index['<SOS>']
                EOS_token = dataset.eng_vocab.word2index['<EOS>']
                PAD_token = dataset.eng_vocab.word2index['<PAD>']
            elif engToFren:
                SOS_token = dataset.fr_vocab.word2index['<SOS>']
                EOS_token = dataset.fr_vocab.word2index['<EOS>']
                PAD_token = dataset.fr_vocab.word2index['<PAD>']
            
            # Decoding step
            #decoder_input = torch.tensor(SOS_token, device=device)
            
            #decoder_hidden = encoder_hidden

            

            #for di in range(target_length-1):
            #    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            #    topv, topi = decoder_output.topk(1)
            #    predicted_indices.append(topi.item())
            #    decoder_input = topi.squeeze().detach()

             #   loss += criterion(decoder_output, target_tensor[di+1].unsqueeze(0))
             #   if decoder_input.item() == EOS_token:
                    #break
            decoder_output = transformer(input_tensor, target_tensor)
            loss = criterion(decoder_output, target_tensor)

            # Find the maximum value along the last dimension
            guesses_prob, guesses_index = torch.max(decoder_output, dim=-1)

            predicted_indices = guesses_index.tolist()

            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            # Remove the <SOS> and <PAD> Characters since the loop ends before adding them
            #temp = [e for e in target_tensor.tolist() if e not in (SOS_token, EOS_token, PAD_token)]
            temp = target_tensor.tolist()
            if predicted_indices == temp:
                correct_predictions += 1
            # Print the prediction and target in either english or french based on settings
            if(i < examplesToPrint):
                print('Prediction: ')
                training_log_file.write('Prediction: ')
                training_log_file.flush()  
                for token in predicted_indices:
                    if(token != EOS_token and token != SOS_token and token != PAD_token):
                        if(engToFren):
                            print(f'{dataset.fr_vocab.index2word[token]}', end = ' ')
                            training_log_file.write(f'{dataset.fr_vocab.index2word[token]}' + ' ')
                            training_log_file.flush()  
                        elif(frenToEng):
                            print(f'{dataset.eng_vocab.index2word[token]}', end = ' ')
                            training_log_file.write(f'{dataset.eng_vocab.index2word[token]}' + ' ')
                            training_log_file.flush()  
                print('\nTarget: ')
                training_log_file.write('\nTarget: ')
                training_log_file.flush()  
                for token in target_tensor.tolist():
                    if(token != EOS_token and token != SOS_token and token != PAD_token):
                        if(engToFren):
                            print(f'{dataset.fr_vocab.index2word[token]}', end = ' ')
                            training_log_file.write(f'{dataset.fr_vocab.index2word[token]}' + ' ')
                            training_log_file.flush()  
                        elif(frenToEng):
                            print(f'{dataset.eng_vocab.index2word[token]}', end = ' ')
                            training_log_file.write(f'{dataset.eng_vocab.index2word[token]}' + ' ')
                            training_log_file.flush()   
                print('\n')
                training_log_file.write('\n\n')
                training_log_file.flush()  
    # Print overall evaluation results
    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}, Time: {total_time} Seconds')
    training_log_file.write(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}, Time: {total_time} Seconds' + '\n')
    training_log_file.flush()

# Negative Log Likelihood Loss function for calculating loss 
criterion = nn.CrossEntropyLoss()

# Used to store stats over training
train_loss_list = []

print(f"\nStarting training for {epochs} epochs")
training_log_file.write(f'\n\nStarting training for {epochs} epochs' + '\n')
training_log_file.flush()
total_time = 0
# Training loop
for epoch in range(epochs):
    start_time = time.time() # Get start time of epoch
    total_loss = 0

    print(f"Starting epoch {epoch + 1} / {epochs}")
    if(saveToLogDelay == 0 or (epoch+1) == epochs or epoch % saveToLogDelay == 0):
        training_log_file.write(f'Starting epoch {epoch + 1} / {epochs}' + '\n')
        training_log_file.flush()

    for input_tensor, target_tensor, _, _ in dataloader:
        # Move tensors to the correct device
        input_tensor = input_tensor[0].to(device)
        target_tensor = target_tensor[0].to(device)
        
        #while(input_tensor.size(0) > target_tensor.size(0)):
        #    input_tensor = torch.cat((input_tensor, torch.zeros(1).to(device)), dim = 0)

        #while(target_tensor.size(0) > input_tensor.size(0)):
        #    input_tensor = torch.cat((target_tensor, torch.zeros(1).to(device)), dim = 0)

        # If translating english to french, keep input and target tensors.
        # If translating french to english, swap the input and target tensors.
        if engToFren:
            loss = train(input_tensor, target_tensor)
        elif frenToEng:
            loss = train(target_tensor, input_tensor)
        total_loss += loss
        # Get the predictions 
    # Print loss every 10 epochs
    end_time = time.time() # Get end time of epoch
    elapsed_time = end_time - start_time # Get elapsed time of epoch
    total_time += elapsed_time

    # Save the loss from this epoch
    train_loss_list.append(total_loss) # Save epoch loss in list

    print(f'Epoch {epoch+1} / {epochs}, Loss: {total_loss}, Time: {elapsed_time} seconds')
    if(saveToLogDelay == 0 or (epoch+1) == epochs or epoch % saveToLogDelay == 0):
        training_log_file.write(f'Epoch {epoch+1} / {epochs}, Loss: {total_loss}, Time: {elapsed_time} seconds' + '\n')
        training_log_file.flush()

print(f'Training Completed in {total_time} seconds')
training_log_file.write(f'Training Completed in {total_time} seconds' + '\n')
training_log_file.flush()

plt.plot(train_loss_list, label = 'Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss VS Epoch')
plt.legend()
plt.savefig('./graphs/language_seq_to_seq/Sequence_to_sequence_prediction.png')
plt.close()

print("Training Loss Graph Generated")

# save Model
torch.save(transformer, 'model/transformer_sequence_to_sequence_prediction.pt')

evaluate_and_show_examples()





