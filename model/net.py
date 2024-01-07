import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU()
            )

        if use_dropout:
            self.block = nn.Sequential(self.block, nn.Dropout(0.5))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UNetBlock(3, 64, down=True, use_dropout=False)
        self.down2 = UNetBlock(64, 128, down=True, use_dropout=False)
        self.down3 = UNetBlock(128, 256, down=True, use_dropout=False)
        self.down4 = UNetBlock(256, 512, down=True, use_dropout=True)
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(512, 256, down=False, use_dropout=False)
        self.up3 = UNetBlock(256, 128, down=False, use_dropout=False)
        self.final = nn.Conv2d(128, 21, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        return self.final(torch.cat([u3, d1], 1))


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # 调整输入维度以匹配UNet的输出
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class HandGestureNet(nn.Module):
    def __init__(self, max_hand_num):
        super(HandGestureNet, self).__init__()
        self.max_hand_num = max_hand_num
        self.unet = UNet()
        self.unet_output_dim = 537600  # UNet输出的扁平化维度
        self.mlp = MLPHead(self.unet_output_dim, max_hand_num)
        # self.gesture_heads = nn.ModuleList([nn.Linear(1, 1) for _ in range(max_hand_num)])
        # 调整关键点检测头的输入维度为整个gesture_logits的输出
        self.keypoint_heads = nn.ModuleList([nn.Linear(max_hand_num, 3) for _ in range(1 * 21)])

    def forward(self, x):
        # size: 640
        # x shape: [batch_size, image_channel, size, size]
        # print(f"input shape: {x.shape}")
        tensor_19 = torch.tensor(19, device='cpu') 
        outputs = []
        for one_batch in x:
            
            one_batch = one_batch.unsqueeze(0)
            # print(f"one_batch: {one_batch.shape}")
            features = self.unet(one_batch)  # [batch_size, keypoints_number, size/2, size/2]
            # print(f"feature shape: {features.shape}")
            flat_features = features.view(features.size(0), -1)
            gesture_logits = self.mlp(flat_features)

            # 获取手势分类的概率分布
            gesture_probs = F.softmax(gesture_logits, dim=1)
            # 选择概率最高的类别作为预测结果
            gesture_values = torch.argmax(gesture_probs, dim=1)


            # 关键点检测头处理
            keypoints = [head(gesture_logits) for head in self.keypoint_heads]
            
            # print(f"Keypoints: \n{keypoints}")
            
            # print(f"Gesture Values Length: {gesture_values.shape}")

            one_batch_outputs = []
            for i in range(self.max_hand_num):
                hand_data = []
                for kp in keypoints:
                    # 确保关键点头输出的张量尺寸正确
                    if kp.shape[0] > i:  # 确保索引 i 在张量的第一个维度范围内
                        hand_data.append(
                            [(kp[i][0].item() + 1)/2,  # 限制在 [0, 1] 区间之内
                             (kp[i][1].item() + 1)/2]  # 限制在 [0, 1] 区间之内
                            )
                    else:
                        hand_data.append([0.0, 0.0])  # 如果索引超出范围，用None填充
                gesture_value = gesture_values[i] if i < len(gesture_values) else tensor_19
                # print("hand_data: ", len(hand_data))
                one_batch_outputs.append([gesture_value, hand_data])

            outputs.append(one_batch_outputs)
        
        # print(f"OutPut Length: {len(outputs)}")
        # 
        return outputs

    
if __name__ == "__main__":
    # 实例化网络
    max_hand_num = 5  # 可以根据实际需求调整
    net = HandGestureNet(max_hand_num)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # 打印网络摘要
    summary(net, input_size=(3, 256, 256), batch_size=8)
