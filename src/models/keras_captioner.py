import tensorflow as tf
from tensorflow.keras import layers, Model

class ImageCaptioner(Model):
    def __init__(self, vocab_size, embedding_dim=256, units=512, max_length=30):
        super().__init__()
        # Encoder: Pretrained CNN (ResNet50)
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')
        self.cnn_encoder = base_model
        self.fc = layers.Dense(embedding_dim, activation='relu')
        
        # Decoder
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.fc_out = layers.Dense(vocab_size)
        self.max_length = max_length

    def call(self, inputs, training=False):
        img_tensor, cap_seq = inputs
        # Encode image
        img_features = self.cnn_encoder(img_tensor, training=False)
        img_features = self.fc(img_features)
        img_features = tf.expand_dims(img_features, 1)  # (batch, 1, embedding_dim)
        
        # Embed captions
        cap_emb = self.embedding(cap_seq)
        
        # Concatenate image features as first input to LSTM
        lstm_input = tf.concat([img_features, cap_emb], axis=1)
        lstm_out, _, _ = self.lstm(lstm_input)
        output = self.fc_out(lstm_out)
        return output

    def generate_caption(self, img_tensor, tokenizer, max_length=None):
        if max_length is None:
            max_length = self.max_length
        # Encode image
        img_features = self.cnn_encoder(img_tensor, training=False)
        img_features = self.fc(img_features)
        img_features = tf.expand_dims(img_features, 1)
        
        # Start with <SOS> token (assumed to be 1)
        caption = [1]
        for _ in range(max_length):
            cap_seq = tf.expand_dims(caption, 0)
            cap_emb = self.embedding(cap_seq)
            lstm_input = tf.concat([img_features, cap_emb], axis=1)
            lstm_out, _, _ = self.lstm(lstm_input)
            output = self.fc_out(lstm_out)
            predicted_id = tf.argmax(output[:, -1, :], axis=-1).numpy()[0]
            if predicted_id == 2:  # <EOS>
                break
            caption.append(predicted_id)
        return tokenizer.sequences_to_texts([caption[1:]])[0]  # skip <SOS>
