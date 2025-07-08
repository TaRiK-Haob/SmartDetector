import trainer
import models

def main():
    pkt_len_embeddings, iat_embeddings = trainer.word2vec_train('test.jsonl')  # 替换为实际数据路径
    sam = models.SAM(pkt_len_embeddings, iat_embeddings)

if __name__ == "__main__":
    main()