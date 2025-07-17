import numpy as np
from tensorflow import keras
from mnist_loader import load_images, load_labels

archive_path = r'C:\Users\user\Downloads\archive'  # <- CHANGE made here

# Load data
X_train = load_images(f'{archive_path}\\train-images.idx3-ubyte')
y_train = load_labels(f'{archive_path}\\train-labels.idx1-ubyte')
X_test = load_images(f'{archive_path}\\t10k-images.idx3-ubyte')
y_test = load_labels(f'{archive_path}\\t10k-labels.idx1-ubyte')

# Preprocess
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, validation_split=0.1, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

model.save('mnist_cnn_model.h5')

