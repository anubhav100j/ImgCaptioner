import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM
from data.dataset import get_loader
from utils import save_checkpoint, load_checkpoint

def train():
    # Load configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model directory
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(config['model_dir'], 'logs'))
    
    # Load data loaders
    train_loader = get_loader(
        root_folder=config['train_image_dir'],
        annotation_file=config['train_annotation_file'],
        transform=None,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )
    
    val_loader = get_loader(
        root_folder=config['val_image_dir'],
        annotation_file=config['val_annotation_file'],
        transform=None,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )
    
    # Get vocabulary size from the dataset
    vocab_size = len(train_loader.dataset.vocab)
    
    # Initialize models
    encoder = EncoderCNN(
        embed_size=config['embed_size'],
        train_encoder=config['train_encoder']
    ).to(device)
    
    decoder = DecoderLSTM(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=vocab_size,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    
    if config['train_encoder']:
        params += list(encoder.cnn.parameters())
    
    optimizer = optim.Adam(params, lr=config['learning_rate'])
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if config['resume']:
        encoder, decoder, optimizer, start_epoch = load_checkpoint(
            encoder, decoder, optimizer, config['checkpoint']
        )
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        # Train for one epoch
        train_loss = train_epoch(
            train_loader, encoder, decoder, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss = validate(val_loader, encoder, decoder, criterion, device, epoch, config)
        
        # Log to tensorboard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_checkpoint(
                encoder, decoder, optimizer, epoch, config['model_dir'], 
                f'model_epoch_{epoch+1}.pth.tar'
            )
        
        # Print progress
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save final model
    save_checkpoint(
        encoder, decoder, optimizer, config['num_epochs'] - 1, 
        config['model_dir'], 'model_final.pth.tar'
    )
    
    # Close tensorboard writer
    writer.close()


def train_epoch(loader, encoder, decoder, criterion, optimizer, device, epoch, config):
    """Train for one epoch."""
    encoder.train()
    decoder.train()
    
    losses = []
    
    for images, captions in tqdm(loader, desc=f'Epoch {epoch + 1} [Train]'):
        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        
        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate loss
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            captions[:, 1:].contiguous().view(-1)
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if config['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(encoder.parameters()),
                config['grad_clip']
            )
        
        optimizer.step()
        
        # Track loss
        losses.append(loss.item())
    
    return sum(losses) / len(losses)


def validate(loader, encoder, decoder, criterion, device, epoch, config):
    """Validate the model."""
    encoder.eval()
    decoder.eval()
    
    losses = []
    
    with torch.no_grad():
        for images, captions in tqdm(loader, desc=f'Epoch {epoch + 1} [Val]'):
            # Move to device
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                captions[:, 1:].contiguous().view(-1)
            )
            
            # Track loss
            losses.append(loss.item())
    
    return sum(losses) / len(losses)


if __name__ == '__main__':
    train()
