import os

# Specify the folder path to be traversed, set this section according to your save path
root_directory = ""
append_string = "_arc_margin"  # The string to be added

# Traverse all folders under the specified path
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)

    # Check if it is a folde
    if os.path.isdir(folder_path):
        # New folder name
        new_folder_name = folder_name - append_string
        new_folder_path = os.path.join(root_directory, new_folder_name)

        # Rename the folder
        os.rename(folder_path, new_folder_path)
        print(f"Renamed '{folder_path}' to '{new_folder_path}'")

print("All folders have been renamed.")
