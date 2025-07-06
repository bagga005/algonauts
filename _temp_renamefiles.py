import os
import sys

if len(sys.argv) != 2:
    print("Usage: python rename_files.py /path/to/directory")
    sys.exit(1)

directory = sys.argv[1]

# Check if the provided path is a directory
if not os.path.isdir(directory):
    print(f"Error: '{directory}' is not a directory.")
    sys.exit(1)

# Process files in the given directory
for filename in os.listdir(directory):
    full_path = os.path.join(directory, filename)
    if os.path.isfile(full_path):
        new_name = filename.replace('task-', '').replace('video', '')
        if new_name != filename:
            new_path = os.path.join(directory, new_name)
            os.rename(full_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
