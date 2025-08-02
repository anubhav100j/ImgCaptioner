import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

class Vocabulary:
    """Vocabulary class for mapping words to indices and vice versa."""
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_freq = defaultdict(int)
        self.n_words = 4  # Count SOS, EOS, PAD, UNK

    def build_vocabulary(self, sentence_list):
        """Build vocabulary from list of sentences."""
        word_freq = defaultdict(int)
        for sentence in tqdm(sentence_list, desc="Building vocabulary"):
            for word in word_tokenize(sentence.lower()):
                word_freq[word] += 1
        
        self.word_freq = word_freq
        
        # Add words that meet the frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

    def numericalize(self, text):
        """Convert text to list of word indices."""
        tokenized_text = word_tokenize(text.lower())
        return [
            self.word2idx[token] if token in self.word2idx else self.word2idx['<UNK>']
            for token in tokenized_text
        ]

    def __len__(self):
        return self.n_words


class ImageCaptionDataset(Dataset):
    """PyTorch Dataset for loading image-caption pairs."""
    def __init__(self, root_dir, captions_file, transform=None, vocab=None, freq_threshold=5):
        """
        Args:
            root_dir (string): Directory with all the images.
            captions_file (string): Path to the file with captions.
            transform (callable, optional): Optional transform to be applied on the images.
            freq_threshold (int): Minimum frequency threshold for words to be included in vocabulary.
        """
        self.root_dir = root_dir
        self.transform = transform or self.default_transform()
        
        # Read and process captions
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        # Initialize or use existing vocabulary
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            all_captions = [item['caption'] for item in self.captions_data]
            self.vocab.build_vocabulary(all_captions)
        else:
            self.vocab = vocab
        
        # Convert captions to numerical form
        self.captions = []
        self.image_paths = []
        for item in self.captions_data:
            img_path = os.path.join(root_dir, item['image'])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                caption = [self.vocab.word2idx['<SOS>']]
                caption += self.vocab.numericalize(item['caption'])
                caption.append(self.vocab.word2idx['<EOS>'])
                self.captions.append(torch.tensor(caption))
    
    @staticmethod
    def default_transform():
        """Default image transformations."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        caption = self.captions[idx]
        
        return image, caption
    
    def get_vocab(self):
        """Returns the vocabulary object."""
        return self.vocab


def get_loader(root_folder, annotation_file, transform, batch_size=32, 
               shuffle=True, num_workers=8, vocab=None, **kwargs):
    """Returns a data loader for the dataset."""
    dataset = ImageCaptionDataset(root_folder, annotation_file, transform, vocab=vocab, **kwargs)
    
    # Create data loader
    pad_idx = dataset.vocab.word2idx['<PAD>']
    
    def collate_fn(batch):
        """Custom collate function to handle variable length captions."""
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        
        # Pad captions to the same length
        captions = [item[1] for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=pad_idx
        )
        
        return images, captions
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
