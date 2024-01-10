import torch
from torchviz import make_dot
from model.net import HandGestureNet
from graphviz import Digraph

def make_dot_from_multiple_outputs(outputs, params=None):
    """ 创建一个图来可视化多个输出的模型 """
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map.get(id(u), '') if params is not None else ''
                node_name = '%s\n %s' % (name, str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    if isinstance(outputs, tuple):
        for o in outputs:
            add_nodes(o.grad_fn)
    else:
        add_nodes(outputs.grad_fn)

    return dot

# 确保你的 HandGestureNet 模型类已经定义，并且导入成功
model = HandGestureNet(5, 'cpu')

# 创建一个随机输入张量来代表你的输入图像
x = torch.randn(1, 3, 320, 320)

# 获取模型的两个输出
output1, output2 = model(x)

# 使用自定义的 make_dot_from_multiple_outputs 来生成可视化图
vis_graph = make_dot_from_multiple_outputs((output1, output2), params=dict(model.named_parameters()))

# 输出图像到文件
vis_graph.render('./handgesturenet_model_multi_output', format='png')
