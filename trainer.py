import torch
import json
import random
import models

class Word2VecDataset(torch.utils.data.Dataset):
    def __init__(self, pkt_len_seqs, pkt_dir_seqs, iat_seqs, window_size=5):
        self.pkt_len_seqs = pkt_len_seqs
        self.iat_seqs = iat_seqs
        self.window_size = window_size
        
        # 生成训练样本
        self.samples = []
        for i in range(len(pkt_len_seqs)):
            self._generate_samples(i)
    
    def _generate_samples(self, seq_idx):
        """为每个序列生成CBOW训练样本"""
        pkt_len_seq = self.pkt_len_seqs[seq_idx]
        iat_seq = self.iat_seqs[seq_idx]
        
        seq_len = len(pkt_len_seq)
        
        for center_idx in range(self.window_size, seq_len - self.window_size):
            # 为每个特征生成上下文窗口
            pkt_len_context = []
            iat_context = []
            
            for i in range(center_idx - self.window_size, center_idx + self.window_size + 1):
                if i != center_idx:
                    pkt_len_context.append(pkt_len_seq[i])
                    iat_context.append(iat_seq[i])
            
            # 中心词
            pkt_len_target = pkt_len_seq[center_idx]
            iat_target = iat_seq[center_idx]
            
            self.samples.append({
                'pkt_len_context': torch.tensor(pkt_len_context, dtype=torch.long),  # 修改为long
                'pkt_len_target': torch.tensor(pkt_len_target, dtype=torch.long),    # 修改为long
                'iat_context': torch.tensor(iat_context, dtype=torch.long),          # 修改为long
                'iat_target': torch.tensor(iat_target, dtype=torch.long)             # 修改为long
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def _load_data_from_jsonl(data_path):
    """从JSONL文件加载数据"""
    pkt_len_seqs = []
    pkt_dir_seqs = []
    iat_seqs = []
    
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if len(data['payload_len_seq']) <= 4:
                continue

            if len(data['payload_len_seq']) < 40:
                padding_len = 40 - len(data['payload_len_seq'])
                data['payload_len_seq'] += [0] * padding_len
                data['payload_ts_seq'] += [0] * padding_len

            # 处理packet len序列 - 确保是整数且在词汇表范围内
            pkt_len_seq = data['payload_len_seq'][:40]
            # 限制在0-1499范围内，超出范围的值映射到1499
            pkt_len_processed = []
            for d in pkt_len_seq:
                val = int(abs(d))
                if val >= 1500:  # 如果超出词汇表大小
                    val = 1499   # 映射到最大值
                pkt_len_processed.append(val)
            pkt_len_seqs.append(pkt_len_processed)
            
            pkt_dir_seqs.append([1 if d > 0 else -1 for d in pkt_len_seq])

            # 处理间隔时间序列 - 确保是整数且在词汇表范围内
            tss = data['payload_ts_seq'][:40]
            iat_seq = [0]
            for i in range(1, len(tss)):
                iat = tss[i] - tss[i - 1]
                if iat < 0:
                    iat = 0
                iat = int(iat * 1000)
                if iat >= 10000:  # 如果超出词汇表大小
                    iat = 9999    # 映射到最大值
                iat_seq.append(iat)
            iat_seqs.append(iat_seq)   
    # print(len(pkt_len_seqs), len(pkt_dir_seqs), len(iat_seqs))

    # 添加数据验证
    print(f"数据加载完成:")
    print(f"pkt_len 范围: {min(min(seq) for seq in pkt_len_seqs)} - {max(max(seq) for seq in pkt_len_seqs)}")
    print(f"iat 范围: {min(min(seq) for seq in iat_seqs)} - {max(max(seq) for seq in iat_seqs)}")
    print(f"总共 {len(pkt_len_seqs)} 个序列")
            
    return pkt_len_seqs, pkt_dir_seqs, iat_seqs


def word2vec_train(data_path):
    pkt_len_seqs, pkt_dir_seqs, iat_seqs = _load_data_from_jsonl(data_path)

    # 创建三个独立的CBOW模型，确保vocab_size足够大
    PKT_LEN_VOCAB_SIZE = 1500  # 包长度词汇表大小
    IAT_VOCAB_SIZE = 10000      # 间隔时间词汇表大小
    
    pkt_len_embeddings = models.CBOW(PKT_LEN_VOCAB_SIZE, 100)
    iat_embeddings = models.CBOW(IAT_VOCAB_SIZE, 100)

    # 将模型移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pkt_len_embeddings.to(device)
    iat_embeddings.to(device)
    
    # 创建三个独立的优化器
    pkt_len_optimizer = torch.optim.Adam(pkt_len_embeddings.parameters(), lr=0.001)
    iat_optimizer = torch.optim.Adam(iat_embeddings.parameters(), lr=0.001)
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建数据集和数据加载器
    dataset = Word2VecDataset(pkt_len_seqs, pkt_dir_seqs, iat_seqs)
    print(f"数据集大小: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练循环
    epochs = 10
    for epoch in range(epochs):
        total_pkt_len_loss = 0
        total_iat_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = {key: value.to(device) for key, value in batch.items()}
                # 训练pkt_len CBOW
                pkt_len_optimizer.zero_grad()
                pkt_len_output = pkt_len_embeddings(batch['pkt_len_context'])
                pkt_len_loss = criterion(pkt_len_output, batch['pkt_len_target'])
                pkt_len_loss.backward()
                pkt_len_optimizer.step()
                total_pkt_len_loss += pkt_len_loss.item()

                # 训练iat CBOW
                iat_optimizer.zero_grad()
                iat_output = iat_embeddings(batch['iat_context'])
                iat_loss = criterion(iat_output, batch['iat_target'])
                iat_loss.backward()
                iat_optimizer.step()
                total_iat_loss += iat_loss.item()
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"PKT_LEN context shape: {batch['pkt_len_context'].shape}")
                print(f"PKT_LEN context range: {batch['pkt_len_context'].min()} - {batch['pkt_len_context'].max()}")
                print(f"PKT_LEN target range: {batch['pkt_len_target'].min()} - {batch['pkt_len_target'].max()}")
                print(f"IAT context range: {batch['iat_context'].min()} - {batch['iat_context'].max()}")
                print(f"IAT target range: {batch['iat_target'].min()} - {batch['iat_target'].max()}")
                raise e
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"PKT_LEN Loss: {total_pkt_len_loss/len(dataloader):.4f}")
        print(f"IAT Loss: {total_iat_loss/len(dataloader):.4f}")
        print("-" * 50)
    
    sam = models.SAM(pkt_len_embeddings, iat_embeddings)  # 确保SAM模型可以使用这两个嵌入模型
    torch.save(sam.state_dict(), 'model_params/sam.pth')  # 保存SAM模型参数

    return sam


def _augmentation(pkt_len_seq, pkt_dir_seq, iat_seq):
    aug_len_seq = []
    aug_dir_seq = []
    aug_iat_seq = []
    
    for i in range(0,len(pkt_len_seq)):
        q = random.uniform(0, 1)
        if q <= 0.5:
            #生成包
            z = random.randint(0, 1499)
            a = random.uniform(0, 0.2)
            d = random.choice([-1, 1])

            # Insert the augmented packet
            aug_len_seq.append(z)
            aug_dir_seq.append(d)
            aug_iat_seq.append(a * 1000)

        r = random.uniform(0, 1)
        if r <= 0.5:
            theta = random.uniform(0, 0.2) * 1000
            iat_seq[i] += int(theta)
            iat_seq[i] = min(iat_seq[i], 9999)  # 限制在0-9999范围内
            
        # Insert the original packet i
        aug_len_seq.append(pkt_len_seq[i])
        aug_dir_seq.append(pkt_dir_seq[i])
        aug_iat_seq.append(iat_seq[i])

    aug_len_seq = torch.tensor(aug_len_seq[:40], dtype=torch.long)
    aug_dir_seq = torch.tensor(aug_dir_seq[:40], dtype=torch.long)
    aug_iat_seq = torch.tensor(aug_iat_seq[:40], dtype=torch.long)

    return aug_len_seq, aug_dir_seq, aug_iat_seq


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, pkt_len_seqs, pkt_dir_seqs, iat_seqs):
        self.pkt_len_seqs = pkt_len_seqs
        self.pkt_dir_seqs = pkt_dir_seqs
        self.iat_seqs = iat_seqs
    
    def __len__(self):
        return len(self.pkt_len_seqs)
    
    def __getitem__(self, idx):
        pkt_len_seq = self.pkt_len_seqs[idx]
        pkt_dir_seq = self.pkt_dir_seqs[idx]
        iat_seq = self.iat_seqs[idx]

        # 生成增强视图
        aug_pkt_len_seq, aug_pkt_dir_seq, aug_iat_seq = _augmentation(pkt_len_seq, pkt_dir_seq, iat_seq)
        view1 = torch.stack([aug_pkt_len_seq, aug_pkt_dir_seq, aug_iat_seq], dim=0)  # [3, seq_len]

        aug_pkt_len_seq, aug_pkt_dir_seq, aug_iat_seq = _augmentation(pkt_len_seq, pkt_dir_seq, iat_seq)
        view2 = torch.stack([aug_pkt_len_seq, aug_pkt_dir_seq, aug_iat_seq], dim=0)  # [3, seq_len]

        return view1, view2


def smart_detector_train(sam, data_path='data/test.jsonl'):

    pkt_len_seqs, pkt_dir_seqs, iat_seqs = _load_data_from_jsonl(data_path)
    dataset = ContrastiveDataset(pkt_len_seqs, pkt_dir_seqs, iat_seqs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"对比学习数据集大小: {len(dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smart_detector = models.SmartDetector(sam=sam)
    smart_detector.to(device)

    optimizer = torch.optim.Adam(smart_detector.parameters(), lr=1e-4)

    for epoch in range(10):
        for batch in dataloader:
            view1, view2 = batch
            view1 = view1.to(device)
            view2 = view2.to(device)
            optimizer.zero_grad()

            loss = smart_detector(view1, view2)

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{10}, Loss: {loss.item():.4f}")

    # # 保存模型参数
    # torch.save({
    #     'sam_state_dict': sam.state_dict(),
    #     'smart_detector_state_dict': smart_detector.state_dict()
    # }, 'model_params/smart_detector.pth')

    torch.save(smart_detector.state_dict(), 'model_params/smart_detector.pth')  # 保存SmartDetector模型参数

    return smart_detector


if __name__ == "__main__":
    sam = word2vec_train('data/minitest.jsonl')  # 替换为实际数据路径

    smart_detector_train(sam, 'data/minitest.jsonl')  # 替换为实际数据路径
