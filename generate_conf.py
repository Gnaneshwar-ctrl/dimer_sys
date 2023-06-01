import math

def generate_coordinates(num_atoms, box_size):
    # Calculate the number of atoms per row/column
    atoms_per_side = math.ceil(math.sqrt(num_atoms))
    
    # Calculate the spacing between atoms
    spacing = box_size / atoms_per_side
    
    # Calculate the offset to center the atoms in the box
    offset = (box_size - (atoms_per_side - 1) * spacing) / 2
    
    # Generate the coordinates
    coordinates = []
    for i in range(atoms_per_side):
        for j in range(atoms_per_side):
            x = offset + i * spacing
            y = offset + j * spacing
            coordinates.append((x, y))
    
    return coordinates

def place_dimer(coordinates, box_size):
    # Calculate the center of the box
    center = box_size / 2
    
    # Calculate the spacing between dimer atoms
    spacing = box_size / math.sqrt(len(coordinates))
    
    # Calculate the offset to center the dimer atoms
    offset = spacing / 2
    
    # Add the dimer atoms
    dimer_coordinates = coordinates.copy()
    dimer_coordinates.append((center - offset, center))
    dimer_coordinates.append((center + offset, center))
    
    return dimer_coordinates

# Parameters
num_atoms = 36
box_size = 6.0

# Generate coordinates for atoms
coordinates = generate_coordinates(num_atoms, box_size)

# Place dimer atoms
dimer_coordinates = place_dimer(coordinates, box_size)

# Save coordinates as LAMMPS input data file
with open('atoms.data', 'w') as file:
    # Write header
    file.write('LAMMPS data file\n\n')
    file.write(f'{len(dimer_coordinates)} atoms\n')
    file.write('2 atom types\n')
    file.write('\n')
    
    # Write box dimensions
    file.write(f'0.0 {box_size} xlo xhi\n')
    file.write(f'0.0 {box_size} ylo yhi\n')
    file.write('0.0 1.0 zlo zhi\n')
    file.write('\n')
    
    # Write atom data
    file.write('Masses\n\n')
    file.write('1 1.0\n')
    file.write('2 1.0\n')
    file.write('\n')
    file.write('Atoms\n\n')
    
    # Write atom coordinates
    for i, (x, y) in enumerate(dimer_coordinates):
        atom_type = 2 if i >= len(coordinates) else 1
        file.write(f'{i+1} {atom_type} {x} {y} 0.0\n')
