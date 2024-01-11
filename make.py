import torch
from torchviz import make_dot
from model.new_net import U_Net

# 确保你的UNet模型类已经定义，并且导入成功
model = U_Net(21)  # 假设你的UNet类已经定义好了

# 创建一个随机输入张量来代表你的输入图像
x = torch.randn(1, 3, 320, 320)  # 你的输入图像大小可能需要调整

# 使用make_dot来生成可视化图
vis_graph = make_dot(model(x), params=dict(model.named_parameters()))

# 输出图像到文件
vis_graph.render('./unet_model', format='png')