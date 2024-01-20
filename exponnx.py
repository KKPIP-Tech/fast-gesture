import torch
import torch.onnx

# 确保 model 和 optimizer 已经被定义和加载
# 假设 'model' 是您的 PyTorch 模型
# 加载模型
model.load_state_dict(torch.load(save_path + '/last.pt')['model'])

# 将模型设置为评估模式
model.eval()

# 定义一个 dummy 输入，这取决于您的模型
# 例如，如果您的模型接受 1x3x224x224 的输入，则可以创建一个相应的张量
# 这里的尺寸 (1, 3, 224, 224) 应该根据您的模型进行调整
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

# 设置 ONNX 文件的保存路径
onnx_save_path = "path_to_save_model/model.onnx"

# 导出模型
torch.onnx.export(model,               # 模型
                  dummy_input,         # 模型输入的 dummy 张量
                  onnx_save_path,      # ONNX 模型的保存路径
                  export_params=True,  # 带有训练参数的模型
                  opset_version=10,    # ONNX 版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names = ['input'],   # 输入名
                  output_names = ['output'], # 输出名
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 批量大小动态化
                                'output' : {0 : 'batch_size'}})

print(f"模型已导出到 {onnx_save_path}")

