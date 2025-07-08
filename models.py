import torch

# K -> Number of Pakcet
# B -> dimension of embeddings
K, B = 40, 100

class FeatureEmbedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureEmbedding, self).__init__()
        self.direction_embedding = torch.nn.Linear(input_dim, output_dim)
        self.pktlen_embedding = torch.nn.Linear(input_dim, output_dim)
        self.IAT_embedding = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        pass


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
def resnet50(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


class SmartDetector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SmartDetector, self).__init__()
        self.feature_embedding = FeatureEmbedding(input_dim, output_dim)
        
        self.encoder = resnet50(num_classes=output_dim)

        # TODO: fix the size of projection_head
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * 3, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.projection_dim)
        )

        self.loss_fn = ContrastiveLoss(temperature=0.1)

    def forward(self, x1, x2):
        x1 = self.feature_embedding(x1)
        x2 = self.feature_embedding(x2)

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        p1 = self.projection_head(x1)
        p2 = self.projection_head(x2)

        loss = self.loss_fn(p1, p2)
        
        return loss
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: 两个增强视图的表示 [batch_size, projection_dim]
        """
        batch_size = z_i.shape[0]

        # 计算相似度矩阵
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        representations = torch.functional.normalize(representations, dim=1)  # 归一化
        similarity_matrix = torch.matmul(representations, representations.T)  # [2*batch_size, 2*batch_size]
        
        # 应用温度参数
        similarity_matrix = similarity_matrix / self.temperature
        
        # 创建标签：对角线上的正样本对
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(similarity_matrix.device)
        
        # 移除对角线（自身相似度）
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(similarity_matrix.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # 计算正样本的相似度
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # 计算负样本的相似度
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # InfoNCE损失
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        loss = torch.functional.cross_entropy(logits, labels)
        return loss
    

