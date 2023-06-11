import numpy as np
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



# Read inputx_n.dat file
inputx_file = 'inputx_n.dat'
inputx_data = np.loadtxt(inputx_file)

# Read inputy_n.dat file
inputy_file = 'inputy_n.dat'
inputy_data = np.loadtxt(inputy_file)

# Determine the number of particles based on the length of the data
num_particles_x = 2  # Assuming 2 lines per configuration in inputx_n.dat
num_particles_y = 36  # Assuming 36 lines per configuration in inputy_n.dat

# Determine the number of configurations based on the length of inputx_data
num_configurations = len(inputx_data) // num_particles_x

# Create the configurations array
configurations = np.empty((num_configurations, num_particles_x + num_particles_y, 2))  # Initialize an empty array

# Iterate over each configuration
for i in range(num_configurations):
    # Extract x coordinates from inputx_data
    x_coordinates = inputx_data[i * num_particles_x: (i + 1) * num_particles_x, 2:]  # Extract x coordinates for the current configuration
    
    # Extract y coordinates from inputy_data
    y_coordinates = inputy_data[i * num_particles_y: (i + 1) * num_particles_y, 2:]  # Extract y coordinates for the current configuration
    
    # Create the configuration array with x and y coordinates
    configuration = np.concatenate((x_coordinates, y_coordinates), axis=0)
    
    configurations[i] = configuration

configurations *= 6

# Use the configurations array in your compute_energy function or any other calculations

# Example: Print the configurations array
# print(configurations)

# print(configurations.shape)


# for i in range(num_configurations):
#     min_values = np.min(configurations[i], axis=0)
#     max_values = np.max(configurations[i], axis=0)
#     print("Configuration", i+1)
#     print("Minimum values:", min_values)
#     print("Maximum values:", max_values)


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