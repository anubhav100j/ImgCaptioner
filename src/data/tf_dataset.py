import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Vocabulary:
    def __init__(self, freq_threshold=5, num_words=None):
        self.freq_threshold = freq_threshold
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<UNK>")
        self.word_index = None
        self.index_word = None

    def build_vocabulary(self, sentence_list):
        self.tokenizer.fit_on_texts(sentence_list)
        self.word_index = self.tokenizer.word_index
        self.index_word = {v: k for k, v in self.word_index.items()}

    def numericalize(self, text):
        return self.tokenizer.texts_to_sequences([text])[0]

    def __len__(self):
        return len(self.tokenizer.word_index) + 1  # +1 for padding


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


def make_tf_dataset(image_folder, captions_file, batch_size=32, buffer_size=1000, max_length=30, vocab=None):
    # Load captions
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    image_paths = []
    captions = []
    for item in captions_data:
        image_paths.append(tf.strings.join([image_folder, '/', item['image']]))
        captions.append(item['caption'])

    # Build or use vocabulary
    if vocab is None:
        vocab = Vocabulary()
        vocab.build_vocabulary(captions)

    # Convert captions to sequences
    cap_seqs = [vocab.numericalize(c) for c in captions]
    cap_seqs = pad_sequences(cap_seqs, maxlen=max_length, padding='post')

    # Create tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices((image_paths, cap_seqs))

    def _map_func(img_path, cap_seq):
        img = load_image(img_path)
        return img, cap_seq

    ds = ds.map(_map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, vocab

# Example usage:
# ds, vocab = make_tf_dataset('data/raw/train2017', 'data/annotations/captions_train2017.json')
# for images, captions in ds.take(1):
#     print(images.shape, captions.shape)
