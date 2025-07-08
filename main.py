import trainer
import models

def main():
    # 训练Word2Vec模型 构建SAM
    pkt_len_embeddings, iat_embeddings = trainer.word2vec_train('data/test.jsonl')  # 替换为实际数据路径
    sam = models.SAM(pkt_len_embeddings, iat_embeddings)

    # 创建SmartDetector模型
    trainer.pretrain(sam)

if __name__ == "__main__":
    main()