import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data',train=True,
        download=True,transform=transform)
    test_set=datasets.FashionMNIST('./data', train=False,
        transform=transform)
    dataset = train_set if training else test_set
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 64, 
        shuffle = True if training else False 
        
    )
    return data_loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(), 
        nn.Linear(64, 10)
        )

    return model



def train_model(model, train_loader, criterion, T):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Learning rate scheduler (optional)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(T):
        model.train()
        total_loss = 0 
        correct = 0 
        total_samples = 0 
        
        # Optionally update learning rate with scheduler
        # scheduler.step()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
        
        accuracy = 100. * correct / total_samples
        avg_loss = total_loss / total_samples
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total_samples} ({accuracy:.2f}%) Loss: {avg_loss:.3f}')



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    total_loss = 0
    correct = 0 
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    
    accuracy = 100. * correct / total_samples
    avg_loss = total_loss / total_samples
    
    if show_loss:
        print(f'Average loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')   


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        logits = model(test_images[index].unsqueeze(0))
        prob = F.softmax(logits, dim=1)
        
        top3_prob, top3_classes = torch.topk(prob, 3, dim = 1)  
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        # Display the results
        for i in range(3):
            class_idx = top3_classes[0][i].item()
            class_name = class_names[class_idx]
            probability = top3_prob[0][i].item() * 100  # Convert to percentage
            print(f'{class_name}: {probability:.2f}%')



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
