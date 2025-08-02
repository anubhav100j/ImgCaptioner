import os
import torch
from torchvision import transforms
from PIL import Image
import yaml
import argparse

from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM
from data.dataset import ImageCaptionDataset
from utils.checkpoint import load_model


def load_image(image_path, transform=None):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image


def generate_caption(image_path, encoder, decoder, vocab, device, max_length=20):
    """Generate a caption for the given image."""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = load_image(image_path, transform)
    image = image.unsqueeze(0).to(device)
    
    # Generate caption
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode the image
        features = encoder(image)
        
        # Generate caption using greedy search
        sampled_ids = decoder.sample(features, max_length=max_length)
        
        # Convert word_ids to words
        sampled_ids = sampled_ids[0].cpu().numpy()
        
        # Convert word IDs to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<EOS>':
                break
            if word not in ['<SOS>', '<PAD>', '<UNK>']:
                sampled_caption.append(word)
        
        sentence = ' '.join(sampled_caption)
        
    return sentence


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate captions for an image')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='models/model_final.pth.tar', 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to the config file')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Maximum length of the generated caption')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary (assuming we have a dataset to get the vocab from)
    # In a real scenario, you should save the vocab with the model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    
    # For demo purposes, we'll create a dummy dataset to get the vocab
    # In practice, you should save the vocab with the model
    transform = None  # We'll handle transform in the generate_caption function
    dataset = ImageCaptionDataset(
        root_dir=config['train_image_dir'],
        captions_file=config['train_annotation_file'],
        transform=transform
    )
    vocab = dataset.vocab
    
    # Initialize models
    encoder = EncoderCNN(
        embed_size=config['embed_size'],
        train_encoder=config['train_encoder']
    ).to(device)
    
    decoder = DecoderLSTM(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=len(vocab),
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Load the trained model parameters
    encoder, decoder = load_model(encoder, decoder, args.model, device)
    
    # Generate caption
    caption = generate_caption(
        args.image, encoder, decoder, vocab, device, args.max_length
    )
    
    print(f"Generated caption: {caption}")


if __name__ == '__main__':
    main()
