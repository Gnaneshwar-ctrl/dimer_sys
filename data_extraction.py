input_filename = 'dump.xy'
output_x_filename = 'inputx.dat'
output_y_filename = 'inputy.dat'

with open(input_filename, 'r') as input_file, \
        open(output_x_filename, 'w') as output_x_file, \
        open(output_y_filename, 'w') as output_y_file:
    is_atoms_section = False  # Flag to indicate if the atoms section has been reached
    for line in input_file:
        line = line.strip()
        if line.startswith('ITEM: ATOMS'):
            # Reached the section containing atom data
            is_atoms_section = True
            continue
        elif is_atoms_section:
            # Split the line into columns
            columns = line.split()
            if len(columns) == 4:
                # Check the third column for the desired value
                if columns[1] == '1':
                    # Write to inputx.dat
                    output_x_file.write(line + '\n')
                elif columns[1] == '2':
                    # Write to inputy.dat
                    output_y_file.write(line + '\n')

print("Data extraction completed.")        
