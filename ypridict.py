from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('model.h5')


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

x_test = data_array
x_data = x_test.reshape((-1, 2, 2))

#print(x_test)
#print(x_data)

predictions = loaded_model.predict(x_test)

#print(predictions)

data = np.array(predictions)

# Reshape the array
y_data = data.reshape((-1, 36, 2))

# Print the new array
#print(new_data)



import matplotlib.pyplot as plt

def compute_energy(configuration):
    # Define parameters of the interaction potentials for each particle type
    epsilon1 = 10.0  # Depth of the potential well for particle type 1
    sigma1 = 1.0  # Distance at which the potential is zero for particle type 1
    epsilon2 = 0.001  # Depth of the potential well for particle type 2
    sigma2 = 1.0  # Distance at which the potential is zero for particle type 2

    # Initialize the total energy
    energy = 0.0

    # Split the configuration array into particle 1 and particle 2 arrays
    particle1 = configuration[:2]
    particle2 = configuration[2:]

    # Compute the energy between particle 1 and particle 2
    for i in range(len(particle1)):
        for j in range(len(particle2)):
            # Compute the distance between the particles
            dx = particle2[j][0] - particle1[i][0]  # x-coordinate difference
            dy = particle2[j][1] - particle1[i][1]  # y-coordinate difference
            distance = np.sqrt(dx**2 + dy**2)

            # Calculate the interaction energy based on the distance and particle types
            if i == 0 and j == 0:
                # Particle 1 with itself
                epsilon = epsilon1
                sigma = sigma1
            else:
                # Particle 2 with itself or Particle 1 with Particle 2
                epsilon = epsilon2
                sigma = sigma2

            # Calculate the Lennard-Jones interaction energy
            lj_energy = 4 * epsilon * ((sigma / distance)**12 - (sigma / distance)**6)

            # Add the energy contribution to the total energy
            energy += lj_energy

    return energy



# Concatenate x and y arrays vertically
configurations = np.concatenate((x_data, y_data), axis=1)
configurations *= 6
# Print the combined array
#print(configurations)


energies = []
for config in configurations:
    #print(config)
    energy = compute_energy(config)
    #print(energy)
    energies.append(energy)

#print(energies)


# Step 3: Convert the energies list to a numpy array
energies = np.array(energies)

# Step 4: Plot a histogram
plt.hist(energies, bins=120, range=[0, 500], label='Energy Distribution')  # Adjust the number of bins and add a label
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.title('Energy Distribution')

# Step 5: Add a legend
plt.legend()

plt.show()

# Step 4: Plot a histogram
plt.hist(energies, bins=120, range=[0, 5], label='Energy Distribution')  # Adjust the number of bins and add a label
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.title('Energy Distribution')

# Step 5: Add a legend
plt.legend()

plt.show()

# Step 4: Plot a histogram
plt.hist(energies, bins=120, range=[0, 0.1], label='Energy Distribution')  # Adjust the number of bins and add a label
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.title('Energy Distribution')

# Step 5: Add a legend
plt.legend()

plt.show()