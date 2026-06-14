# here we are going to go modula using the magic command %% writefile
""""
Contains functionnalities for creatring PyTorch DataLoaders's fore image classification
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       transform: transforms.Compose, 
                       batch_size: int, num_workers: int=NUM_WORKERS-4):
    """
    Takes in a training directory and testing directory path and turns them into PyTroch Dataset and DataLoader
    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvison transform to perform on training and testing data
        batch_size: Number of sample per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_name is a list of the target classes.
        Example usage:
            train_dataloader, test_dataloader, classnames = create_dataloaders(...)
    """
    # Create simple transform
    data_transform = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=data_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")


    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True, 
                                pin_memory=True, 
                                prefetch_factor=2) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                shuffle=False) # don't usually need to shuffle testing data

    # Get class names as a list
    class_names = train_data.classes
    class_names

    return train_dataloader, test_dataloader, class_names
