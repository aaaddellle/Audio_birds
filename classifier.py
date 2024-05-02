import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2


def sequential():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 1), name='Layer1'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name='Layer2'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", name='Layer3'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(units=256, activation="relu", name="hidden_layer1")) #sigmoid - hidden layer
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=128, activation="relu", name="layer2"))
    model.add(BatchNormalization())
    model.add(Dense(units=10, activation="softmax", name="layer3"))
    #epochs = 10

    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    feature_extraction_model = Model(inputs=model.input, outputs=model.get_layer('Layer3').output)
    return model, feature_extraction_model

model, feature_extraction_model = sequential()
