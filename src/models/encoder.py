import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Encoder module that extracts image features using a pre-trained CNN.
    Uses ResNet-50 as the base model by default.
    """
    def __init__(self, embed_size, train_encoder=False):
        """
        Args:
            embed_size (int): Dimension of the output embedding
            train_encoder (bool): Whether to fine-tune the CNN encoder
        """
        super(EncoderCNN, self).__init__()
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]  # Remove the last fc layer
        self.cnn = nn.Sequential(*modules)
        
        # Freeze the parameters if not training the encoder
        if not train_encoder:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Add a learnable affine layer to transform features to the embedding space
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        # Initialize weights
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        """
        Forward pass through the encoder.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Image features of shape (batch_size, embed_size)
        """
        # Extract features from the CNN
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.cnn(images)  # (batch_size, 2048, 1, 1)
        
        # Flatten the features
        features = features.reshape(features.size(0), -1)  # (batch_size, 2048)
        
        # Project to the embedding space
        features = self.linear(features)  # (batch_size, embed_size)
        features = self.bn(features)
        
        return features


if __name__ == "__main__":
    # Test the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy input
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Initialize the encoder
    embed_size = 512
    encoder = EncoderCNN(embed_size).to(device)
    
    # Forward pass
    features = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Encoder training mode: {encoder.training}")
    
    # Test with training mode
    encoder.train()
    features_train = encoder(dummy_input)
    print(f"Features in training mode: {features_train.requires_grad}")
    
    # Test with evaluation mode
    encoder.eval()
    with torch.no_grad():
        features_eval = encoder(dummy_input)
    print(f"Features in eval mode: {features_eval.requires_grad}")
