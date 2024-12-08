import test_resnet18

for batch in test_resnet18.test_loader:
    print(len(batch))  # Number of items in the batch
    print(type(batch))  # Type of the batch (e.g., tuple, list)
    break
