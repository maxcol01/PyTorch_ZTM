"""
Trains a PyTorch image classification model using device agnostic code
"""
import os
import torch
from torchvision import transforms
import datasetup, engine, model, utils
from timeit import default_time as timer
#setup hyperparameters

NUM_EPOCH = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# setup directories
train_dir = "data/pizza_steak_suchi/train"
train_dir = "data/pizza_steak_suchi/test"


# device agnostic
device = "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# create DataLoaders and get class names
train_dataloader, test_dataloader, class_names = datasetup.create_dataloaders(train_dir=train_dir, 
                                                                              test_dir=test_dir, 
                                                                              transforms=data_transform, batch_size = BATCH_SIZE)

# Create model
# 
model = model.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

# Set up loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())

# Start training 
start_time = timer()
engine.train(model=model, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            epochs=NUM_EPOCHS, 
            device=device)
end_time = timers()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# save the model to file
utils.save_model(model=model, target="models", model_name=".....pth")
