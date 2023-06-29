import torch 
import torch.nn  
import torch.optim
import torch.utils.data.DataLoader

# Create an instance of the ResNet18 model with 100 output classes
model = resnet34(num_classes=100)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Lists to store the loss and accuracy for each epoch
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []
predicted_labels = []

# Train the model
for epoch in range(200):
    train_loss = 0.0
    train_correct = 0 # count the number of correct predictions
    train_total = 0 # count the total number of samples
    model.train()

    # Iterate over the training dataset
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute the training loss and accuracy
    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_accuracy = 100 * train_correct / train_total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Compute the validation loss and accuracy
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_correct = 0 # count the number of correct predictions
        valid_total = 0 # count the total number of samples

        # Iterate over the validation dataset
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
            if epoch == 199:
                predicted_labels += predicted.tolist()


        # Compute the validation loss and accuracy
        epoch_valid_loss = valid_loss / len(test_loader.dataset)
        epoch_valid_accuracy = 100 * valid_correct / valid_total
        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_accuracy)

    # Print the training and validation metrics for each epoch
    print(f'Epoch {epoch+1}/{200}:')
    print(f'Training Loss: {epoch_train_loss:.4f} | Training Accuracy: {epoch_train_accuracy:.2f}%')
    print(f'Validation Loss: {epoch_valid_loss:.4f} | Validation Accuracy: {epoch_valid_accuracy:.2f}%')