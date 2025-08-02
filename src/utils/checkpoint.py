import os
import torch

def save_checkpoint(encoder, decoder, optimizer, epoch, save_dir, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        epoch: Current epoch
        save_dir: Directory to save the checkpoint
        filename: Checkpoint filename
    """
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save checkpoint
    torch.save(state, os.path.join(save_dir, filename))
    print(f"Checkpoint saved to {os.path.join(save_dir, filename)}")


def load_checkpoint(encoder, decoder, optimizer, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        encoder: Encoder model with loaded weights
        decoder: Decoder model with loaded weights
        optimizer: Optimizer with loaded state
        epoch: The epoch from which to resume training
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    
    return encoder, decoder, optimizer, epoch


def load_model(encoder, decoder, checkpoint_path, device='cpu'):
    """
    Load only the model weights (without optimizer state).
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        encoder: Encoder model with loaded weights
        decoder: Decoder model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    
    return encoder, decoder
