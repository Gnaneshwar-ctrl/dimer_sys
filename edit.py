# Specify the file paths
input_file_path = 'dump.metal.xyz'  # Path to the input file
output_file_path = 'inputy.dat'  # Path to the output file

# Read the input file and capture lines starting with '1'
lines_to_save = []
with open(input_file_path, 'r') as input_file:
    for line in input_file:
        if line.startswith('2'):
            lines_to_save.append(line)

# Save the captured lines to the output file
with open(output_file_path, 'w') as output_file:
    output_file.writelines(lines_to_save)
