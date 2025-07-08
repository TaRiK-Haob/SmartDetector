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
            # TODO: 根据实际数据格式解析
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
                iat = round(float(iat), 4)
                if iat >= 10000:  # 如果超出词汇表大小
                    iat = 9999    # 映射到最大值
                iat_seq.append(iat)
            iat_seqs.append(iat_seq)
    
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
    
    # 创建三个独立的优化器
    pkt_len_optimizer = torch.optim.Adam(pkt_len_embeddings.parameters(), lr=0.001)
    iat_optimizer = torch.optim.Adam(iat_embeddings.parameters(), lr=0.001)
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建数据集和数据加载器
    dataset = Word2VecDataset(pkt_len_seqs, pkt_dir_seqs, iat_seqs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练循环
    epochs = 10
    for epoch in range(epochs):
        total_pkt_len_loss = 0
        total_iat_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
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
    
    # 保存训练好的嵌入模型
    torch.save({
        'pkt_len_embeddings': pkt_len_embeddings.state_dict(),
        'iat_embeddings': iat_embeddings.state_dict()
    }, 'word2vec_embeddings.pth')
    
    return pkt_len_embeddings, iat_embeddings


if __name__ == "__main__":
    pkt_len_embeddings, iat_embeddings = word2vec_train('data/test.jsonl')  # 替换为实际数据路径