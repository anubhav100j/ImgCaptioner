import tensorflow as tf
from src.data.tf_dataset import make_tf_dataset
from src.models.keras_captioner import ImageCaptioner

# Config
IMAGE_DIR = 'data/raw/train2017'
CAPTIONS_FILE = 'data/annotations/captions_train2017.json'
BATCH_SIZE = 32
MAX_LENGTH = 30
EPOCHS = 5

# Load data
train_ds, vocab = make_tf_dataset(
    image_folder=IMAGE_DIR,
    captions_file=CAPTIONS_FILE,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)

# Model
model = ImageCaptioner(vocab_size=len(vocab), embedding_dim=256, units=512, max_length=MAX_LENGTH)

# Compile
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss_fn)

# Train
model.fit(train_ds, epochs=EPOCHS)

# Save model
model.save('models/keras_captioner.h5')
