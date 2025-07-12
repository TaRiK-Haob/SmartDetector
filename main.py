import trainer, tester
import models
import torch
import os


def load_models(path='model_params/smart_detector.pth'):
    # stat_dict = torch.load(path)
    # print("Loaded state dict keys:", stat_dict.keys())

    model = models.SmartDetector()
    model.load_state_dict(torch.load(path))

    # 取消SAM模型的嵌入层参数的梯度计算
    model.feature_embedding.pkt_len_embeddings.weight.requires_grad = False
    model.feature_embedding.iat_embeddings.weight.requires_grad = False

    return model


def main():
    # Pretrain the a Embedding model(i.e. SAM)
    # and use SmartDetector to train a encoder for downstream classification tasks
    if os.path.exists('model_params/smart_detector.pth'):
        print("Loading SmartDetector model from file...")
        sd = load_models()  # 加载SmartDetector模型
    else:
        print("Training SmartDetector model...")
        sd = trainer.smart_detector_train(
            trainer.word2vec_train("data/minitest.jsonl"), # 训练并创建语义属性矩阵SAM (未来指定为sam_train.jsonl)
            "data/minitest.jsonl"     # 训练SmartDetector模型(指定为pre_train.jsonl)
        )
    
    # 预训练分类模型
    if os.path.exists('model_params/smart_detector.pth'):
        print("Loading Classifier model from file...")
        classifier = models.Classifier(sd, 2)
        classifier.load_state_dict(torch.load('model_params/finetune.pth'))
    else:
        print("Pre-Training Classifier model...")
        # 创建分类器模型
        classifier = models.Classifier(sd, 2)
        classifier = trainer.classifier_finetune(classifier, "data/classifier.jsonl")   # 训练下游任务Classifier模型(指定为classifier.jsonl)


    tester.test(classifier, "data/test.jsonl")  # 测试分类器模型




if __name__ == "__main__":
    main()