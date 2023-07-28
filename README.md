## How to Use Model and Data Parallelism in PyTorch

How to use two types of parallelism in PyTorch: model parallelism and data parallelism. Some of the advantages and disadvantages of each approach, and provide some examples and tips on how to implement them in your code.

### What is Model Parallelism?

Model parallelism is a technique that splits a large model into smaller parts and assigns each part to a different device. This way, you can train or run a model that is too big to fit into a single device’s memory, or leverage the specialized capabilities of different devices, such as tensor cores or mixed precision.

For example, suppose you have a model that consists of 8 layers and you have 4 gpus. You can use model parallelism to place the first 2 layers in device 0 and the second 2 layers in device 1 and the third 2 layers in device 2 and the last 2 layers in device 3, and then connect them with some communication mechanism, such as PyTorch’s torch.distributed package.

### To use data parallelism in PyTorch, you need to do the following steps:

- Define your model as a subclass of torch.nn.Module and implement the forward method.

- Create an instance of the model and move it to the device that will host the master copy of the parameters.

- Create an optimizer and a loss function as usual.

- Perform the forward, backward, and update steps as usual.

Here is an example of how to use data parallelism in PyTorch:
```
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')

class my_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(in_features = 12, out_features = 256).to(device0)
        self.layer2 = nn.Linear(in_features = 256, out_features = 128).to(device0)
        self.layer3 = nn.Linear(in_features = 128, out_features = 64).to(device1)
        self.layer4 = nn.Linear(in_features = 64, out_features = 8).to(device1)
        self.layer5 = nn.Linear(in_features = 8, out_features = 64).to(device2)
        self.layer6 = nn.Linear(in_features = 64, out_features = 128).to(device2)
        self.layer7 = nn.Linear(in_features = 128, out_features = 256).to(device3)
        self.layer8 = nn.Linear(in_features = 256, out_features = 12).to(device3)

    def forward(self, inp):
        x = self.relu(self.layer1(inp.to(device0)))
        x = self.relu(self.layer2(x.to(device0)))
        x = self.relu(self.layer3(x.to(device1)))
        x = self.relu(self.layer4(x.to(device1)))
        x = self.relu(self.layer5(x.to(device2)))
        x = self.relu(self.layer6(x.to(device2)))
        x = self.relu(self.layer7(x.to(device3)))
        x = self.layer8(x.to(device3))
        return (x)


# Build and define your model
model = my_model()

# Define the Loss function and the Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters())

# Specify the number of epochs
epochs = 10

# Instantiate lists to save the loss values during the training process
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # Training mode
    model.train()

    train_loss = 0

    for train_x, train_y in train_loader:

        train_x, train_y = train_x.to(device), train_y.to(device)
        # At start of each Epoch
        optimizer.zero_grad()

        # Feedforward
        y_pred = model(train_x)

        # Calculate the loss function
        loss = loss_fn(y_pred, train_y.to(y_pred.device))
        train_loss += loss

        # Do the back probagation and update the parameters
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)

    # Evaluation mode
    model.eval()

    loss_test = 0

    with torch.inference_mode():

        for test_x, test_y in test_loader:

            test_x, test_y = test_x.to(device), test_y.to(device)

            # Feedforward again for the evaluation phase
            y_pred_test = model(test)

            # Calculate the loss for the test dataset
            loss_test += loss_fn(test.to(y_pred_test.device), y_pred_test)

        loss_test /= len(test_loader)

    # Append loss values for the training process
    train_loss_values.append(train_loss)          # .cpu().detach().numpy()
    test_loss_values.append(loss_test)
    epoch_count.append(epoch)
    print(f"Epoch : {epoch + 1} | train_Loss: {train_loss} | test_Loss: {loss_test}")
```
### Advantages and Disadvantages of Model Parallelism

Both model parallelism and data parallelism have their own advantages and disadvantages, depending on the use case and the characteristics of the model and the data.

#### Some of the advantages of model parallelism are:

- It allows you to train or run models that are too large to fit into a single device’s memory.

- It can reduce the communication overhead by minimizing the amount of data that needs to be transferred between devices.

#### Some of the disadvantages of model parallelism are:

- It can introduce additional complexity and overhead in designing and implementing the model architecture and the communication mechanism.

- It can increase the synchronization cost by requiring frequent gradient exchanges between devices.

- It can limit the scalability by depending on the number of sub-models or layers that can be parallelized.

### Conclusion

- How to use model parallelism in PyTorch. 

- Discussed some of the advantages and disadvantages of each approach.

- Provided some examples and tips on how to implement in your code.
