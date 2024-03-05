import os

# Set the directory containing the files to be renamed
directory = 'models/augmented_taxi2/'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if '03_02_sensitivity_tc2_jk_mlf' in filename:
        # Create the new filename by replacing 'lfh' with 'mlf'
        new_filename = filename.replace('mlf', 'fixed')
        
        # Create the full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed "{filename}" to "{new_filename}"')