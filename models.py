import torch
import torchvision

# K -> Number of Pakcet
# B -> dimension of embeddings
K, B = 40, 100

# D -> dimension of deep representation
D = 64


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context)
        embeds = torch.mean(embeds, dim=1)  # 使用mean而不是sum
        out = self.linear(embeds)  # 使用embeds而不是out
        return out


class SAM(torch.nn.Module):
    def __init__(self, pkt_len_model = CBOW(1500, 100), iat_model = CBOW(10000, 100)):
        super(SAM, self).__init__()
        # 使用训练好的CBOW模型的嵌入层
        self.pkt_len_embeddings = pkt_len_model.embeddings
        self.iat_embeddings = iat_model.embeddings
        
        # # 冻结嵌入层权重（可选）
        # self.pkt_len_embeddings.weight.requires_grad = False
        # self.iat_embeddings.weight.requires_grad = False

    def _get_pkt_dir_embeds(self, pkt_dir_seq):
        """
        直接根据pkt_dir_seq的值生成嵌入向量
        输入: pkt_dir_seq [batch_size, seq_len] 值为1或-1
        输出: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = pkt_dir_seq.shape
        
        # 创建与pkt_dir_seq相同形状的嵌入矩阵
        pkt_dir_embeds = pkt_dir_seq.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        pkt_dir_embeds = pkt_dir_embeds.expand(batch_size, seq_len, 100)  # [batch_size, seq_len, embedding_dim]
        
        return pkt_dir_embeds
        
    def forward(self, x):
        """
        - 输入：
        x -> [batch_size, 3, seq_len] 包长度、包方向和间隔时间序列
        - 输出：
        x -> [batch_size, 3, seq_len, embedding_dim] 
            - 相当于 resnet的输入格式（channels, height, width）
        """
        # 分离输入序列
        pkt_len_seq = x[:, 0, :]  # [batch_size, seq_len]
        pkt_dir_seq = x[:, 1, :]  # [batch_size, seq_len]
        iat_seq = x[:, 2, :]      # [batch_size, seq_len]

        # 获取嵌入向量
        pkt_len_embeds = self.pkt_len_embeddings(pkt_len_seq)  # [batch_size, seq_len, embedding_dim]
        pkt_dir_embeds = self._get_pkt_dir_embeds(pkt_dir_seq)  # [batch_size, seq_len, embedding_dim]
        iat_embeds = self.iat_embeddings(iat_seq)              # [batch_size, seq_len, embedding_dim]

        # 拼接嵌入向量
        x = torch.stack([pkt_len_embeds, pkt_dir_embeds, iat_embeds], dim=1)
        
        return x


def resnet50(out_dim=1000):
    model = torchvision.models.resnet50(weights=None)

    # 修改全连接层为深层表示的向量维度
    model.fc = torch.nn.Linear(model.fc.in_features, out_dim)

    return model


class SmartDetector(torch.nn.Module):
    def __init__(self, sam = SAM()):
        super(SmartDetector, self).__init__()
        self.feature_embedding = sam
        
        self.encoder = resnet50(D)

        # 投影头
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
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
        representations = torch.nn.functional.normalize(representations, dim=1)  # 归一化
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
        
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss


class Classifier(torch.nn.Module):
    """
    Classifier model that combines SAM and ResNet50 for feature extraction and classification.
    - sam: SAM model for feature embedding
    - encoder: ResNet50 encoder for feature extraction
    """

    def __init__(self, samrt_detector = SmartDetector(), num_classes=2):
        super(Classifier, self).__init__()
        self.feature_embedding = samrt_detector.feature_embedding
        self.encoder = samrt_detector.encoder

        # feature_embedding 和 encoder 不需要训练
        for param in self.feature_embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.output = torch.nn.Linear(D, num_classes)


    def forward(self, x):
        x = self.feature_embedding(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x