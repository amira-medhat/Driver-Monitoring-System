# from roboflow import Roboflow

# rf = Roboflow(api_key="SnqcHeeqiN7aknrWqHCo")  # Replace with your API key
# # Load workspace and project
# workspace = rf.workspace("driver-monitoring-system")  # Ensure correct name
# project = workspace.project("driver-activity")  # Ensure correct project name

# # List available versions
# print("Available versions:", project.versions())

# # Select the latest available version dynamically
# latest_version = project.versions()[-1]  # Get the last (latest) version
# dataset = project.version(latest_version).dataset()  # Load dataset

# # Upload images
# dataset.upload(
#     folder=r"C:\Users\Amira\state-farm-distracted-driver-detection\imgs\train",
#     num_workers=8,
# )

# print("Upload completed successfully!")

# Install
$ pip install roboflow

# Authenticate
$ roboflow login

# Import
$ roboflow import -w driver-monitoring-system -p driver-activity r"C:\Users\Amira\state-farm-distracted-driver-detection\imgs\train"