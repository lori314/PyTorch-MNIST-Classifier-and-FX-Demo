import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class MyModule(nn.Module):
    def __init__(self) -> None:
        super(MyModule,self).__init__()
        self.layer1=nn.Linear(28*28,128)
        self.layer2=nn.Linear(128,64)
        self.layer3=nn.Linear(64,10)
    
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=self.layer3(x)
        return x
    

model=MyModule()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化处理
])

# 下载并加载训练数据集
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载并加载测试数据集
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device=device)

num_epochs=100
for epoch in range(num_epochs):
    running_loss=0.0
    for i,(images,labels) in enumerate(trainloader):
        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}')

torch.save(model.state_dict(), 'model_state_dict.pth')