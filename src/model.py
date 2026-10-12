from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_model(patch_size, n_components, n_classes, dropout_rate=0.3, learning_rate=0.001, clip_value=0.5):
    model = Sequential([
        Input(shape=(patch_size, patch_size, n_components)),
        Conv2D(32, (4, 4), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation="relu"),
        Dropout(dropout_rate),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        Dropout(dropout_rate),
        Flatten(), Dense(128, activation="relu"), Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax")])
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipvalue=clip_value),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
