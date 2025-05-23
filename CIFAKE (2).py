import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix , accuracy_score
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import cv2
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D







# Function to load CIFAKE dataset from a regular folder
def load_cifake_data(folder_path='/home/netweb/Documents/CIFAKE/archive', img_size=(32, 32)):
    print("Loading CIFAKE dataset from folder...")

    # Define train and test directories
    train_dir = os.path.join(folder_path, 'train')
    test_dir = os.path.join(folder_path, 'test')

    # Count images
    train_real = os.listdir(os.path.join(train_dir, 'REAL'))
    train_fake = os.listdir(os.path.join(train_dir, 'FAKE'))
    test_real = os.listdir(os.path.join(test_dir, 'REAL'))
    test_fake = os.listdir(os.path.join(test_dir, 'FAKE'))

    print("Number of real images in training set:", len(train_real))
    print("Number of fake images in training set:", len(train_fake))
    print("Number of real images in testing set:", len(test_real))
    print("Number of fake images in testing set:", len(test_fake))

    # Create data generators
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=len(train_real) + len(train_fake),
        class_mode='binary',
        shuffle=False
    )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=len(test_real) + len(test_fake),
        class_mode='binary',
        shuffle=False
    )

    # ✅ Correct way to get data from DirectoryIterator
    X_train_real_fake, y_train_real_fake = next(train_generator)
    X_test_real_fake, y_test_real_fake = next(test_generator)

    print(f"Loaded {X_train_real_fake.shape[0]} training images and {X_test_real_fake.shape[0]} test images")

    return X_train_real_fake, y_train_real_fake, X_test_real_fake, y_test_real_fake

# ✅ Call the modified function (pass your actual unzipped CIFAKE path here)
X_train, y_train, X_test, y_test = load_cifake_data('/home/netweb/Documents/CIFAKE/archive')










def create_feature_extractor(num_filters, num_layers):
    model = Sequential()
    model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    for _ in range(num_layers - 1):
        model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(1, (1, 1), activation='sigmoid'))
    model.add(GlobalAveragePooling2D())
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def benchmark_cnn_extractors(X_train, y_train, X_test, y_test, filter_list=[16, 32, 64, 128], layer_list=[1, 2, 3]):
    loss_matrix = []
    precision_matrix = []
    recall_matrix = []
    accuracy_matrix = []
    f1_matrix = []

    best_loss = float('inf')
    best_model_info = None

    for filters in filter_list:
        row_loss, row_prec, row_rec, row_acc, row_f1 = [], [], [], [], []
        for layers in layer_list:
            print(f"\n🔧 Training CNN with {filters} filters and {layers} layers...")
            model = create_feature_extractor(filters, layers)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

            loss, _ = model.evaluate(X_test, y_test, verbose=0)
            y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            row_loss.append(round(loss, 3))
            row_prec.append(round(precision, 3))
            row_rec.append(round(recall, 3))
            row_acc.append(round(acc, 3))
            row_f1.append(round(f1, 3))

            if loss < best_loss:
                best_loss = loss
                best_model_info = (filters, layers)

        loss_matrix.append(row_loss)
        precision_matrix.append(row_prec)
        recall_matrix.append(row_rec)
        accuracy_matrix.append(row_acc)
        f1_matrix.append(row_f1)

    layer_headers = [str(l) for l in layer_list]
    filter_rows = [str(f) for f in filter_list]

    print("\n📊 TABLE 3. Validation Loss:")
    print(pd.DataFrame(loss_matrix, index=filter_rows, columns=layer_headers))

    print("\n📊 TABLE 4. Validation Precision:")
    print(pd.DataFrame(precision_matrix, index=filter_rows, columns=layer_headers))

    print("\n📊 TABLE 5. Validation Recall:")
    print(pd.DataFrame(recall_matrix, index=filter_rows, columns=layer_headers))

    print("\n📊 Validation Accuracy:")
    print(pd.DataFrame(accuracy_matrix, index=filter_rows, columns=layer_headers))

    print("\n📊 Validation F1 Score:")
    print(pd.DataFrame(f1_matrix, index=filter_rows, columns=layer_headers))

    print(f"\n✅ Best model based on LOSS: {best_model_info[0]} filters, {best_model_info[1]} layers (Loss = {best_loss:.3f})")
    return best_model_info


best_filter, best_layer = benchmark_cnn_extractors(X_train, y_train, X_test, y_test)
print(best_filter,"  ",best_layer)










# ✅ Build base CNN for reuse
def build_conv_base(filters, layers):
    # Define input shape explicitly
    inputs = Input(shape=(32, 32, 3))

    # First convolution block
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    # Additional convolution blocks
    for _ in range(layers - 1):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

    # Create and return the model
    return Model(inputs=inputs, outputs=x)

# ✅ Attach dense head to CNN
def build_dense_model(conv_base, dense_units, dense_layers):
    x = Flatten()(conv_base.output)
    for _ in range(dense_layers):
        x = Dense(dense_units, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=conv_base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ✅ Benchmark CNN + Dense models, print tables & save best model by F1 score
def benchmark_dense_models(X_train, y_train, X_test, y_test, base_filters, base_layers, save_dir='saved_models'):
    os.makedirs(save_dir, exist_ok=True)

    dense_units_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    dense_layers_list = [1, 2, 3]

    results = {
        'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []
    }

    best_f1 = -1
    best_model_info = None
    best_model_path = None

    for units in dense_units_list:
        row_loss, row_acc, row_prec, row_rec, row_f1 = [], [], [], [], []
        for layers in dense_layers_list:
            print(f"\n🔧 Training model: {units} units | {layers} layers")

            conv_base = build_conv_base(base_filters, base_layers)
            model = build_dense_model(conv_base, units, layers)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

            loss, _ = model.evaluate(X_test, y_test, verbose=0)
            y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            row_loss.append(round(loss, 3))
            row_acc.append(round(acc, 3))
            row_prec.append(round(precision, 3))
            row_rec.append(round(recall, 3))
            row_f1.append(round(f1, 3))

            # ✅ Save best model by F1 Score
            if f1 > best_f1:
                best_f1 = f1
                best_model_info = (units, layers)
                best_model_path = os.path.join(save_dir, f"best_model_{units}u_{layers}l_f1_{round(f1, 3)}.h5")
                model.save(best_model_path)

        results['Loss'].append(row_loss)
        results['Accuracy'].append(row_acc)
        results['Precision'].append(row_prec)
        results['Recall'].append(row_rec)
        results['F1 Score'].append(row_f1)

    # 🧾 Metric tables
    layer_headers = [str(l) for l in dense_layers_list]
    unit_rows = [str(u) for u in dense_units_list]

    print("\n📊 Validation Loss:")
    print(pd.DataFrame(results['Loss'], index=unit_rows, columns=layer_headers))

    print("\n📊 Validation Accuracy:")
    print(pd.DataFrame(results['Accuracy'], index=unit_rows, columns=layer_headers))

    print("\n📊 Validation Precision:")
    print(pd.DataFrame(results['Precision'], index=unit_rows, columns=layer_headers))

    print("\n📊 Validation Recall:")
    print(pd.DataFrame(results['Recall'], index=unit_rows, columns=layer_headers))

    print("\n📊 Validation F1 Score:")
    print(pd.DataFrame(results['F1 Score'], index=unit_rows, columns=layer_headers))

    print(f"\n✅ Best Dense Model (by F1): {best_model_info[0]} units, {best_model_info[1]} layers (F1 Score = {best_f1:.3f})")
    print(f"📁 Model saved at: {best_model_path}")

    return best_model_path


best_model_path = benchmark_dense_models(X_train, y_train, X_test, y_test, best_filter, best_layer)




