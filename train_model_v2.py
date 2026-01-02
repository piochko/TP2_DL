import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.keras

# 1. Chargement et Découpage des données (Ex 1)
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

x_train = x_train.reshape(54000, 784).astype("float32") / 255.0
x_val = x_val.reshape(6000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0

# 2. Configuration
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
L2_REG = 0.001

optimizers_to_test = {
    'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': 'rmsprop',
    'Adam': 'adam'
}

# 3. Boucle d'entraînement
for opt_name, optimizer in optimizers_to_test.items():
    with mlflow.start_run(run_name=f"Optimizer_{opt_name}"):
        
        mlflow.log_param("optimizer_name", opt_name)
        mlflow.log_param("l2_reg", L2_REG)

        # Construction du modèle - VERSION STRICTE TP2
        model = keras.Sequential([
            keras.layers.Input(shape=(784,)),
            
            # Dropout APRES la couche d'entrée
            keras.layers.Dropout(DROPOUT_RATE), 
            
            keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(L2_REG)),
            keras.layers.BatchNormalization(), #
            keras.layers.Activation('relu'),
            
            # Dropout AVANT la couche de sortie
            keras.layers.Dropout(DROPOUT_RATE), 
            
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\n--- Entraînement : {opt_name} ---")
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val), #
            verbose=1
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.keras.log_model(model, f"model_{opt_name}")

print("\nTerminé.")