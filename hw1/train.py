import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import wandb
wandb.init(project="aipro-hw1-cifar10")


batch_size = 128
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainset = torchvision.datasets.CIFAR10(root = './data',train = True,download = True,transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train = False,download = True,transform = transform)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size = batch_size,shuffle = False,num_workers = 0)

dataiter = iter(trainloader)
images,labels = next(dataiter)


class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this assignment.")

device = torch.device("cuda")
print("device:", device)


model = LeNet().to(device)
images = images.to(device)
labels = labels.to(device)


print("model device:", next(model.parameters()).device)
print("images device:", images.device)
print("labels device:", labels.device)



criterion = nn.CrossEntropyLoss()


learning_rate = 0.01
momentum = 0.0

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
    momentum=momentum
)


epoches = 10

for epoch in range(epoches):

    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for i , (images,labels) in enumerate(trainloader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total+=labels.size(0)
        correct +=(predicted == labels).sum().item()
        wandb.log({"loss":loss.item()})


    epoch_loss = running_loss/len(trainloader)
    epoch_accuracy = 100*correct/total
    wandb.log({
        "epoch":epoch+1,
        "epoch_loss":epoch_loss,
        "train_accuracy":epoch_accuracy
        })
    

    print(
            f"Epoch [{epoch+1}/{epoches}],"
            f"Loss: {epoch_loss:.4f}"
            f"Accuracy: {epoch_accuracy:.2f}"
        )



model.eval()

test_loss = 0.0
correct = 0
total = 0
class_correct = [0] * len(testset.classes)
class_total = [0] * len(testset.classes)

with torch.no_grad():
    for images,labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        test_loss += loss.item()

        _,predicted = torch.max(outputs.data,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, prediction in zip(labels, predicted):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

test_loss = test_loss/len(testloader)
test_accuracy = 100 * correct / total
class_accuracies = {
    class_name: 100 * class_correct[class_index] / class_total[class_index]
    for class_index, class_name in enumerate(testset.classes)
}

print(
    f"Test Loss: {test_loss:.4f}"
    f"Test Accuracy: {test_accuracy:.2f}"
)

print("Per-class test accuracy:")
for class_name, class_accuracy in class_accuracies.items():
    print(f"{class_name}: {class_accuracy:.2f}%")

wandb.log({
    "test_loss":test_loss,
    "test_accuracy":test_accuracy,
    **{f"test_accuracy/{class_name}": class_accuracy
       for class_name, class_accuracy in class_accuracies.items()}
    })




wandb.finish()


