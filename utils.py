import os


def rename_files(directory, prefix):
    counter = 701
    for filename in os.listdir(directory):
        # Ignore directories
        if os.path.isdir(os.path.join(directory, filename)):
            continue

        # Get file extension
        file_ext = os.path.splitext(filename)[1]

        # New filename with prefix and counter
        new_filename = f"{prefix}_{counter}{file_ext}"

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        counter += 1


# Example usage:
directory = "/home/faiaz/Documents/motion_planning_datasets-master/single_bugtrap/validation"
prefix = "validation"
rename_files(directory, prefix)
