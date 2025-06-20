import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

img_size = (100, 100)
batch_size = 8
epochs = 20
train_dir = "dataset"

# Augmentation + preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Base MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base_model.trainable = True  
for layer in base_model.layers[:-20]: # congelar todas excepto últimas 20 camadas convolucionais
    layer.trainable = False
# Topo customizado
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Class weights (ajustar se necessário)
class_weight = {0: 1.0, 1: 2.0}  # mais peso à classe do aluno

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('modelo/modelo.h5', save_best_only=True)
]

model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, class_weight=class_weight)
