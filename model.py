# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IGlI6gyJAs8uPekJiy6TuD4eveYwyAHm
"""

import tensorflow as tf
from tensorflow.keras import layers

def resnet_block(inputs, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU()(x)
    return x

def build_model():
    # Fully Connected Neural Network 1
    input_fc1 = layers.Input(shape=(4,))
    x = layers.Dense(16)(input_fc1)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.Reshape((8, 8, 1))(x)
    x = layers.Softmax()(x)
    x = layers.Lambda(lambda x: x * 30.3)(x)
    output_fc1 = x

    # RESNET v1 1
    #input_resnet1 = layers.Input(shape=(8, 8, 1))
    input_resnet1 = output_fc1
    x = resnet_block(input_resnet1, filters=64)
    x = resnet_block(x, filters=64)
    output_resnet1 = x

    # RESNET v1 2
    #input_resnet2 = layers.Input(shape=(8, 8, 64))
    input_resnet2 = output_resnet1 
    x = resnet_block(input_resnet2, filters=64)
    x = resnet_block(x, filters=64)
    output_resnet2 = x

    # Fully Connected Neural Network 2
    #input_fc2 = layers.Input(shape=(32, 32, 64))
    input_fc2 = output_resnet2
    x = layers.Flatten()(input_fc2)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    output_fc2 = layers.Dense(72)(x)

    # Build the model
    model = tf.keras.Model(
        inputs=[input_fc1],
        outputs=[output_fc2]
    )
    return model

# Create the model
model = build_model()

# Print the model summary
model.summary()

import numpy as np
from sklearn import preprocessing

# Specify the file path
file_path = 'inputx.dat'  # Path to the .dat file

# Read the file and extract the 2nd and 3rd columns
data = np.loadtxt(file_path)
input_1 = data[:, 1]
input_2 = data[:, 2]

# Split column_2 into two columns
columnx_1_new = []
columnx_2_new = []

for i in range(len(input_1)):
    if i % 2 == 0:
        columnx_1_new.append(input_1[i])
    else:
        columnx_2_new.append(input_1[i])

# Split column_2 into two columns
columny_1_new = []
columny_2_new = []

for i in range(len(input_2)):
    if i % 2 == 0:
        columny_1_new.append(input_2[i])
    else:
        columny_2_new.append(input_2[i])

input_1_n = preprocessing.normalize([columnx_1_new])
input_2_n = preprocessing.normalize([columny_1_new])
input_3_n = preprocessing.normalize([columnx_2_new])
input_4_n = preprocessing.normalize([columny_2_new])

# Reshape the normalized arrays
reshaped_column_1 = input_1_n.reshape((-1, 1))
reshaped_column_2 = input_2_n.reshape((-1, 1))
reshaped_column_3 = input_3_n.reshape((-1, 1))
reshaped_column_4 = input_4_n.reshape((-1, 1))


# Combine the columns into a single input array
x_data = np.concatenate((reshaped_column_1, reshaped_column_2, reshaped_column_3, reshaped_column_4), axis=1)
x_data = np.array(x_data)
# Print the arrays
#print("2nd Column:")
print(x_data)
#print("3rd Column:")
#print(input_2_n)

#import numpy as np

# Specify the file path
file_path = 'inputy.dat'  # Path to the .dat file

# Read the file and extract the 2nd and 3rd columns
data = np.loadtxt(file_path)
output_1 = data[:, 1]
output_2 = data[:, 2]


output_1_n = preprocessing.normalize([output_1])
output_2_n = preprocessing.normalize([output_2])

# Reshape the normalized arrays
reshapedy_column_2 = output_1_n.reshape((-1, 1))
reshapedy_column_3 = output_2_n.reshape((-1, 1))


# Combine the columns into a single input array
y_data_2 = np.concatenate((reshapedy_column_2, reshapedy_column_3), axis=1)
#y_data_2 = np.repeat(y_data_2, len(output_1) // len(y_data_2), axis=0)

# Print the arrays
#print("2nd Column:")
#print(y_data)
#print("3rd Column:")
#print(output_2)

def convert_matrix(original_matrix, m):
    original_rows, original_cols = len(original_matrix), len(original_matrix[0])
    new_matrix = [[0] * m for _ in range(72)]  # Create an empty 72xm matrix
    
    for row in range(72):
        for col in range(m):
            original_row = row % original_rows
            original_col = col % original_cols
            new_matrix[row][col] = original_matrix[original_row][original_col]
    
    return new_matrix


y_data = convert_matrix(y_data_2, 150001)
y_data = np.array(y_data)
y_data = y_data.reshape((-1, 1))
print(y_data)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(x_data, y_data, batch_size=32, epochs=100, validation_split=0.2)


model.save('model.h5')
# Evaluate the model
#loss, mae = model.evaluate(x_test, [y_test] * 4)

# Make predictions
#predictions = model.predict(x_test)

