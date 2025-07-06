import os

# Get all files in the current directory
for filename in os.listdir('.'):
    # Only process files (not directories)
    if os.path.isfile(filename):
        # Remove 'task-' and 'video' from filename
        new_name = filename.replace('task-', '').replace('video', '')
        # Only rename if the name actually changes
        if new_name != filename:
            os.rename(filename, new_name)
            print(f"Renamed: {filename} -> {new_name}")