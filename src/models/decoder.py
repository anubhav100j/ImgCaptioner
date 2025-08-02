import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    """
    LSTM-based decoder that generates captions from image features.
    Uses attention mechanism to focus on relevant image regions.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        """
        Args:
            embed_size (int): Size of word embeddings
            hidden_size (int): Size of hidden state in LSTM
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        
        # Store hyperparameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
    
    def forward(self, features, captions, lengths=None):
        """
        Forward pass through the decoder.
        
        Args:
            features (torch.Tensor): Image features from encoder (batch_size, embed_size)
            captions (torch.Tensor): Ground truth captions (batch_size, max_length)
            lengths (list): Lengths of the captions for packing
            
        Returns:
            torch.Tensor: Output logits (batch_size, max_length, vocab_size)
        """
        batch_size = features.size(0)
        
        # Embed the captions
        embeddings = self.embed(captions)  # (batch_size, max_length, embed_size)
        
        # Initialize hidden state and cell state using image features
        h, c = self.init_hidden(features)  # (num_layers, batch_size, hidden_size)
        
        # Prepare input for LSTM (shifted right)
        inputs = embeddings[:, :-1]  # Remove <EOS> token
        
        if lengths is not None:
            # Sort inputs by decreasing lengths
            lengths = [l - 1 for l in lengths]  # Subtract 1 because we removed <EOS>
            lengths = torch.tensor(lengths, dtype=torch.long, device=features.device)
            sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
            inputs = inputs[sort_idx]
            features = features[sort_idx]
            
            # Pack padded sequence
            inputs = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
            )
        
        # Forward pass through LSTM
        lstm_out, (h, c) = self.lstm(inputs, (h, c))
        
        if lengths is not None:
            # Unpack packed sequence
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            
            # Unsort the outputs
            _, unsort_idx = sort_idx.sort()
            lstm_out = lstm_out[unsort_idx]
        
        # Apply attention
        attention_weights = self.attention(
            torch.cat([lstm_out, features.unsqueeze(1).expand_as(lstm_out)], dim=2)
        )
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context = (lstm_out * attention_weights).sum(dim=1)
        
        # Final output layer
        outputs = self.fc(context)
        
        return outputs
    
    def init_hidden(self, features):
        """Initialize hidden state and cell state using image features."""
        # Use a linear layer to project image features to initial hidden state
        h = features.unsqueeze(0).expand(self.lstm.num_layers, -1, -1)
        c = torch.zeros_like(h)
        return h, c
    
    def sample(self, features, max_length=20, temperature=1.0):
        """Generate captions for given image features using greedy search."""
        batch_size = features.size(0)
        
        # Initialize hidden state and cell state
        h, c = self.init_hidden(features)
        
        # Start with <SOS> token
        inputs = torch.ones(batch_size, 1, dtype=torch.long, device=features.device) * 1  # 1 = <SOS>
        
        # Store generated words
        sampled_ids = []
        
        for i in range(max_length):
            # Embed the input tokens
            embeddings = self.embed(inputs)  # (batch_size, 1, embed_size)
            
            # Forward pass through LSTM
            lstm_out, (h, c) = self.lstm(embeddings, (h, c))
            
            # Apply attention
            attention_weights = self.attention(
                torch.cat([lstm_out, features.unsqueeze(1).expand_as(lstm_out)], dim=2)
            )
            attention_weights = F.softmax(attention_weights / temperature, dim=1)
            
            # Apply attention weights
            context = (lstm_out * attention_weights).sum(dim=1)
            
            # Predict next word
            outputs = self.fc(context)
            _, predicted = outputs.max(1)
            
            # Store the predicted word
            sampled_ids.append(predicted.unsqueeze(1))
            
            # Next input is the predicted word
            inputs = predicted.unsqueeze(1)
            
            # Stop if all sequences have generated <EOS> token (2 = <EOS>)
            if (predicted == 2).all():
                break
        
        # Concatenate the sampled words
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, max_length)
        
        return sampled_ids


if __name__ == "__main__":
    # Test the decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy inputs
    batch_size = 32
    embed_size = 512
    hidden_size = 512
    vocab_size = 10000
    max_length = 20
    
    # Initialize the decoder
    decoder = DecoderLSTM(embed_size, hidden_size, vocab_size).to(device)
    
    # Create dummy features and captions
    features = torch.randn(batch_size, embed_size).to(device)
    captions = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)
    lengths = [max_length] * batch_size
    
    # Forward pass
    outputs = decoder(features, captions, lengths)
    print(f"Input features shape: {features.shape}")
    print(f"Input captions shape: {captions.shape}")
    print(f"Output logits shape: {outputs.shape}")
    
    # Test sampling
    sampled_ids = decoder.sample(features)
    print(f"Sampled captions shape: {sampled_ids.shape}")
