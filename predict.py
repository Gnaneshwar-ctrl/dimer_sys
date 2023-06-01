from tensorflow.keras.models import load_model
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
# Load the saved model
loaded_model = load_model('model.h5')

#new_data = [0.00249351, 0.00091206, 0.00298935, 0.00090879]
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
new_data_reshaped = np.array(x_data)



#new_data_reshaped = np.reshape(new_data, (1, -1))
# Perform prediction on new data
#predictions = loaded_model.predict(new_data_reshaped)
predictions = [0,1]
print(new_data_reshaped)
print(predictions)
# Create a MinMaxScaler object for inverse transformation
scaler = MinMaxScaler()

# Specify the file path
file_path = 'inputy_text.dat'  # Path to the .dat file

# Read the file and extract the 2nd and 3rd columns
data = np.loadtxt(file_path)
output_1 = data[:, 1]
output_2 = data[:, 2]


#original1_shape = output_1.shape
#original2_shape = output_2.shape

reshaped_column_1_y = output_1.reshape((-1, 1))
reshaped_column_2_y = output_2.reshape((-1, 1))

# Combine the columns into a single input array
#y_data_2 = np.concatenate((output_1, output_2), axis=0).reshape(-1, 1)
y_data_2 = np.concatenate((reshaped_column_1_y, reshaped_column_2_y), axis=1)
###print(y_data_2)
#y_shape = np.concatenate((output_1, output_2), axis=0)
#original_shape = y_shape.shape
# Fit the scaler on the original data
#scaler.fit(y_data_2)

print(y_data_2)
#print(y_shape)

#actual_predictions = scaler.inverse_transform(predictions)

# Reshape the actual predictions to match the original shape
#actual_predictions = actual_predictions.reshape((-1, 2))

#actual_output_1_reshape = actual_predictions.reshape(original_shape)

# Extract the actual values from the columns
##actual_output_1 = actual_predictions[:, 0]
##actual_output_2 = actual_predictions[:, 1]


#actual_output_1_reshape = actual_output_1.reshape(original1_shape)
#actual_output_2_reshare = actual_output_2.reshape(original2_shape)

# ###print the predictions
#####print(actual_output_1)
#####print(actual_output_2)
###print(predictions)
#scaler.fit(normalized_data)
####print(actual_predictions, len(actual_predictions))


###print(predictions.shape)

import math


# Determine the dimensions of the box
box_size = 6  # Size of the box
grid_size = 8  # Size of the grid

# Calculate the grid spacing
spacing = box_size / (grid_size - 1)

# Create an empty 8x8 grid
#grid_points = np.zeros((grid_size, grid_size), dtype=int)
grid_points = []

for atom in y_data_2:
    x, y = atom
    grid_x = int(x / spacing)
    grid_y = int(y / spacing)
    grid_points.append([grid_x, grid_y]) 

#print(grid_points)


# Calculate the coordinates of the center of each grid point
grid_centers = []
for i in range(grid_size):
    for j in range(grid_size):
        x_center = i * spacing
        y_center = j * spacing
        grid_centers.append([x_center, y_center])


#grid_points_variable = grid_points.tolist()

print(grid_centers)
grid_centers = np.array(grid_centers)
print(grid_centers.shape)

def fn(x, y, x_new, y_new, sigma):
        r = x - x_new
        ri = y - y_new
        exp = math.sqrt((r**2 + ri**2))
        exponent = -((exp) ** 2) / (2 * sigma ** 2)
        result = math.exp(exponent)
        return result

Z_total = np.array([])
sigma = 0.4

for i in range(0,64):
#    new_var = predictions[i]
    new_var = grid_centers[i] 
    x = new_var[0]
    y = new_var[1]
    Z_y = 0

    for m in range(0,36):
        ram = (i*36 + m)
        #r = complex(x, y)
        x_new = y_data_2[ram][0]
        y_new = y_data_2[ram][0]
        #r_i = complex(x_new, y_new) 
        q = fn(x, y, x_new, y_new, sigma)
        q = float(q)
        Z_y = Z_y + q
        
    ####print(Z_y)
    #Z_total = np.array([arr for arr in Z_total] + [arr for arr in Z_y])
    Z_total = np.append(Z_total, Z_y)

Z_total = np.array(Z_total) 

print(Z_total)

filename = "log_data.txt"

for i in range(0,64):
    var = Z_total[i]
    with open(filename, 'a') as file:
        file.write(str(var))
        file.write("\n") 


import matplotlib.pyplot as plt

# Assuming you have a 64-element array
data = Z_total
data = np.arange(64).reshape((8, 8))

# Plotting the heatmap
#plt.imshow(data, cmap='hot', interpolation='nearest')
#plt.colorbar()  # Optional, add a colorbar
#plt.show()

print(y_data_2.shape)

split_data = np.reshape(y_data_2, (100001, 36, 2))

print(split_data[0].shape)
print(split_data[0])

x = split_data[0][:, 0]
y = split_data[0][:, 1]


dimer = [[4.76314, 1.73205 ], [5.62917,   1.73205  ]]
dimer = np.array(dimer)
x_d = dimer[:, 0]
y_d = dimer[:, 1]
# Plot the atoms
plt.scatter(x, y)
plt.scatter(x_d, y_d)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Atom Positions')
plt.show()

#fig, axs = plt.subplots(8, 8, figsize=(10, 10))

# Iterate over each subplot and plot the corresponding data
#for i in range(8):
#    for j in range(8):
#        index = i * 8 + j
#        axs[i, j].plot(split_data[index][:, 0], split_data[index][:, 1])
#        axs[i, j].set_title(f'Grid {index+1}')

# Adjust the spacing between subplots
#plt.tight_layout()

# Show the plot
#plt.show()

# Plotting the heatmap
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()  # Optional, add a colorbar
plt.show()