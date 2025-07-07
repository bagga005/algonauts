import os
import sys

if len(sys.argv) != 2:
    print("Usage: python add_prefix.py /path/to/directory")
    sys.exit(1)

directory = sys.argv[1]

if not os.path.isdir(directory):
    print(f"Error: '{directory}' is not a directory.")
    sys.exit(1)

def add_prefix_to_files(target_dir, prefix):
    for filename in os.listdir(target_dir):
        filepath = os.path.join(target_dir, filename)
        # Only rename files, not directories
        if os.path.isfile(filepath) and not filename.startswith(prefix):
            new_filename = prefix + filename
            new_filepath = os.path.join(target_dir, new_filename)
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filepath} -> {new_filepath}")

# Add prefix to files in the main directory
add_prefix_to_files(directory, "movie10_")

# Add prefix to files in each immediate subdirectory
for item in os.listdir(directory):
    subdir_path = os.path.join(directory, item)
    if os.path.isdir(subdir_path):
        add_prefix_to_files(subdir_path, "movie10_")
