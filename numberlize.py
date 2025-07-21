import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image

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
# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小为28x28
    transforms.ToTensor(),        # 将图片转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化处理
])

# 加载图像
image_path = "test9.png"  # 图像路径
image = Image.open(image_path).convert('L')  # 转换为灰度图

# 应用预处理
image = transform(image).unsqueeze(0)  # 增加批次维度

# 加载模型
model = MyModule()  # 之前训练的模型结构
model.load_state_dict(torch.load('model_state_dict.pth',weights_only=True))
model.eval()

# 预测
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# 输出预测结果
print(f"Predicted digit: {predicted.item()}")