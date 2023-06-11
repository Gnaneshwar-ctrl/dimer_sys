import numpy as np

input_filename = 'inputx.dat'
output_filename = 'inputx_n.dat'

# Initialize minimum and maximum values for normalization
min_col3 = float('inf')
max_col3 = float('-inf')
min_col4 = float('inf')
max_col4 = float('-inf')

# Read the input file and find the minimum and maximum values
with open(input_filename, 'r') as input_file:
    for line in input_file:
        columns = line.strip().split()
        if len(columns) == 4:
            col3 = float(columns[2])
            col4 = float(columns[3])
            min_col3 = min(min_col3, col3)
            max_col3 = max(max_col3, col3)
            min_col4 = min(min_col4, col4)
            max_col4 = max(max_col4, col4)

# Normalize the values and write to the output file
with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
    for line in input_file:
        columns = line.strip().split()
        if len(columns) == 4:
            col3 = float(columns[2])
            col4 = float(columns[3])
            normalized_col3 = (col3 - min_col3) / (max_col3 - min_col3)
            normalized_col4 = (col4 - min_col4) / (max_col4 - min_col4)
            output_file.write(f'{columns[0]} {columns[1]} {normalized_col3} {normalized_col4}\n')




input_filename = 'inputy.dat'
output_filename = 'inputy_n.dat'

# Initialize minimum and maximum values for normalization
min_col3 = float('inf')
max_col3 = float('-inf')
min_col4 = float('inf')
max_col4 = float('-inf')

# Read the input file and find the minimum and maximum values
with open(input_filename, 'r') as input_file:
    for line in input_file:
        columns = line.strip().split()
        if len(columns) == 4:
            col3 = float(columns[2])
            col4 = float(columns[3])
            min_col3 = min(min_col3, col3)
            max_col3 = max(max_col3, col3)
            min_col4 = min(min_col4, col4)
            max_col4 = max(max_col4, col4)

# Normalize the values and write to the output file
with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
    for line in input_file:
        columns = line.strip().split()
        if len(columns) == 4:
            col3 = float(columns[2])
            col4 = float(columns[3])
            normalized_col3 = (col3 - min_col3) / (max_col3 - min_col3)
            normalized_col4 = (col4 - min_col4) / (max_col4 - min_col4)
            output_file.write(f'{columns[0]} {columns[1]} {normalized_col3} {normalized_col4}\n')



input_filename = 'inputx_n.dat'
data_array = []

with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()
    for i in range(0, len(lines), 2):
        line1 = lines[i].strip().split()
        line2 = lines[i+1].strip().split()
        
        if len(line1) == 4 and len(line2) == 4:
            data_array.append([float(line1[2]), float(line1[3]), float(line2[2]), float(line2[3])])

data_array = np.array(data_array)

#print(data_array)

x_train = data_array


input_filename = 'inputy_n.dat'
data_array = []

with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()
    for i in range(0, len(lines), 36):
        subarray = []
        for j in range(i, i+36):
            line = lines[j].strip().split()
            if len(line) == 4:
                subarray.extend([float(line[2]), float(line[3])])
        data_array.append(subarray)

data_array = np.array(data_array)

#print(data_array)

y_train = data_array


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

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

model.save('model.h5')
