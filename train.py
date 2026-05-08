#!/usr/bin/env python3
"""
中文新闻文本分类训练脚本
支持TextCNN和BERT模型的训练、验证和测试
"""

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path

# 设置HuggingFace镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# 导入项目模块
from src.data.data_loader import NewsDataLoader, TextProcessor
from src.data.preprocessing import ChineseTextPreprocessor
from src.data.vectorization import Word2VecVectorizer, BERTVectorizer
from src.models.textcnn import TextCNN, TextCNNConfig, create_textcnn_model
from src.models.bert_classifier import BertClassifier, BertClassifierConfig, create_bert_classifier, create_bert_tokenizer
from src.training.trainer import ModelTrainer, create_data_loaders

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NewsClassificationTrainer:
    """新闻分类训练器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.data_loader = NewsDataLoader()
        self.text_processor = TextProcessor()
        self.preprocessor = ChineseTextPreprocessor()
        
        # 模型和训练器
        self.model = None
        self.trainer = None
        self.tokenizer = None
        
        # 数据
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        default_config = {
            'model_type': 'bert',  # 'textcnn' or 'bert'
            'data_path': 'data/raw/Train.txt',
            'output_dir': 'results',
            'model_dir': 'models',
            'batch_size': 16,
            'num_epochs': 5,
            'learning_rate': 2e-5,
            'max_length': 512,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'save_best': True,
            'early_stopping_patience': 3,
            'textcnn_config': {
                'vocab_size': 20000,
                'embedding_dim': 100,
                'filter_sizes': [3, 4, 5],
                'num_filters': 100,
                'dropout_rate': 0.5
            },
            'bert_config': {
                'model_name': 'bert-base-chinese',
                'dropout_rate': 0.3,
                'freeze_bert': False
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("=" * 60)
        print("开始加载和预处理数据")
        print("=" * 60)
        
        # 加载原始数据
        print(f"加载数据文件: {self.config['data_path']}")
        df = self.data_loader.load_txt(
            self.config['data_path'],
            separator='\t',
            text_column=2,
            label_column=1,
            id_column=0
        )
        
        if df is None:
            raise ValueError("数据加载失败")
        
        print(f"原始数据形状: {df.shape}")
        print(f"类别分布:")
        print(df['label'].value_counts())
        
        # 预处理数据
        print("\n开始预处理数据...")
        # 先进行基本清理，不去重
        df_processed = df.dropna(subset=['text', 'label']).copy()
        df_processed = df_processed[df_processed['text'].str.len() >= 2]
        df_processed = df_processed[df_processed['text'].str.len() <= 1000]
        df_processed = df_processed.reset_index(drop=True)
        
        print(f"预处理后数据形状: {df_processed.shape}")
        print(f"类别分布:")
        print(df_processed['label'].value_counts())
        
        # 划分数据集
        print("\n划分数据集...")
        # 检查每个类别的样本数
        label_counts = df_processed['label'].value_counts()
        min_samples = label_counts.min()
        print(f"最少类别样本数: {min_samples}")
        
        # 如果最少类别样本数小于2，不使用分层抽样
        stratify = df_processed['label'] if min_samples >= 2 else None
        if stratify is None:
            print("警告: 某些类别样本数太少，不使用分层抽样")
        
        self.train_data, self.val_data, self.test_data = self.data_loader.split_data(
            df_processed,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state'],
            stratify=stratify is not None
        )
        
        # 获取类别名称
        self.class_names = sorted(df_processed['label'].unique().tolist())
        print(f"类别数量: {len(self.class_names)}")
        print(f"类别名称: {self.class_names}")
        
        print("数据加载和预处理完成!")
        return df_processed
    
    def prepare_textcnn_data(self):
        """准备TextCNN数据"""
        print("\n准备TextCNN数据...")
        
        # 预处理文本
        train_texts, train_labels = self.preprocessor.preprocess_dataframe(
            self.train_data, 'text', 'label'
        )
        val_texts, val_labels = self.preprocessor.preprocess_dataframe(
            self.val_data, 'text', 'label'
        )
        test_texts, test_labels = self.preprocessor.preprocess_dataframe(
            self.test_data, 'text', 'label'
        )
        
        # 创建词汇表
        all_texts = train_texts + val_texts + test_texts
        vocab = set()
        for text in all_texts:
            vocab.update(text)
        
        vocab = ['<PAD>', '<UNK>'] + sorted(list(vocab))
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        print(f"词汇表大小: {len(vocab)}")
        
        # 文本转索引
        def texts_to_indices(texts, word_to_idx, max_length=100):
            indices = []
            for text in texts:
                text_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text[:max_length]]
                # 填充或截断
                if len(text_indices) < max_length:
                    text_indices.extend([word_to_idx['<PAD>']] * (max_length - len(text_indices)))
                else:
                    text_indices = text_indices[:max_length]
                indices.append(text_indices)
            return np.array(indices)
        
        max_length = 100
        X_train = texts_to_indices(train_texts, word_to_idx, max_length)
        X_val = texts_to_indices(val_texts, word_to_idx, max_length)
        X_test = texts_to_indices(test_texts, word_to_idx, max_length)
        
        # 更新配置
        self.config['textcnn_config']['vocab_size'] = len(vocab)
        self.config['textcnn_config']['num_classes'] = len(self.class_names)
        
        return (X_train, train_labels), (X_val, val_labels), (X_test, test_labels), vocab
    
    def prepare_bert_data(self):
        """准备BERT数据"""
        print("\n准备BERT数据...")
        
        # 创建BERT分词器
        bert_config = BertClassifierConfig(
            model_name=self.config['bert_config']['model_name'],
            num_classes=len(self.class_names),
            max_length=self.config['max_length']
        )
        self.tokenizer = create_bert_tokenizer(bert_config)
        
        # 编码文本
        def encode_texts(texts):
            encoded = self.tokenizer.encode_texts(texts)
            return encoded['input_ids'], encoded['attention_mask']
        
        train_texts = self.train_data['text'].tolist()
        val_texts = self.val_data['text'].tolist()
        test_texts = self.test_data['text'].tolist()
        
        # 编码训练集
        train_input_ids, train_attention_mask = encode_texts(train_texts)
        train_labels = self.preprocessor.label_encoder.fit_transform(self.train_data['label'])
        
        # 编码验证集
        val_input_ids, val_attention_mask = encode_texts(val_texts)
        val_labels = self.preprocessor.label_encoder.transform(self.val_data['label'])
        
        # 编码测试集
        test_input_ids, test_attention_mask = encode_texts(test_texts)
        test_labels = self.preprocessor.label_encoder.transform(self.test_data['label'])
        
        X_train = (train_input_ids, train_attention_mask)
        X_val = (val_input_ids, val_attention_mask)
        X_test = (test_input_ids, test_attention_mask)
        
        return (X_train, train_labels), (X_val, val_labels), (X_test, test_labels)
    
    def create_model(self):
        """创建模型"""
        print(f"\n创建{self.config['model_type'].upper()}模型...")
        
        if self.config['model_type'].lower() == 'textcnn':
            # 创建TextCNN模型
            config = TextCNNConfig(**self.config['textcnn_config'])
            self.model = create_textcnn_model(config)
            
        elif self.config['model_type'].lower() == 'bert':
            # 创建BERT模型
            config = BertClassifierConfig(
                model_name=self.config['bert_config']['model_name'],
                num_classes=len(self.class_names),
                dropout_rate=self.config['bert_config']['dropout_rate'],
                freeze_bert=self.config['bert_config']['freeze_bert'],
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                num_epochs=self.config['num_epochs'],
                max_length=self.config['max_length']
            )
            self.model = create_bert_classifier(config)
            
        else:
            raise ValueError(f"不支持的模型类型: {self.config['model_type']}")
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建训练器
        self.trainer = ModelTrainer(
            model=self.model,
            device=self.device,
            save_dir=self.config['model_dir']
        )
    
    def train_model(self):
        """训练模型"""
        print("\n" + "=" * 60)
        print("开始训练模型")
        print("=" * 60)
        
        # 准备数据
        if self.config['model_type'].lower() == 'textcnn':
            (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = self.prepare_textcnn_data()
        else:  # bert
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_bert_data()
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=self.config['batch_size'],
            model_type=self.config['model_type']
        )
        
        # 训练模型
        start_time = time.time()
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config['num_epochs'],
            learning_rate=self.config['learning_rate'],
            early_stopping_patience=self.config['early_stopping_patience'],
            save_best=self.config['save_best']
        )
        training_time = time.time() - start_time
        
        print(f"\n训练完成! 用时: {training_time:.2f}秒")
        
        # 评估模型
        print("\n评估模型...")
        results = self.trainer.evaluate(test_loader, self.class_names)
        
        # 打印结果
        print(f"\n测试结果:")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        
        # 保存结果
        self._save_results(results, history, training_time)
        
        # 绘制图表
        self._plot_results(history, results)
        
        return results, history
    
    def _save_results(self, results, history, training_time):
        """保存结果"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 保存评估结果
        results_dict = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'training_time': training_time,
            'model_type': self.config['model_type'],
            'class_names': self.class_names,
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        with open(os.path.join(self.config['output_dir'], 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        # 保存分类报告
        with open(os.path.join(self.config['output_dir'], 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("分类报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"模型类型: {self.config['model_type']}\n")
            f.write(f"训练时间: {training_time:.2f}秒\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"精确率: {results['precision']:.4f}\n")
            f.write(f"召回率: {results['recall']:.4f}\n")
            f.write(f"F1分数: {results['f1_score']:.4f}\n\n")
            f.write("详细分类报告:\n")
            f.write(str(classification_report(results['true_labels'], results['predictions'], 
                                            target_names=self.class_names)))
        
        print(f"\n结果已保存到: {self.config['output_dir']}")
    
    def _plot_results(self, history, results):
        """绘制结果图表"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 绘制训练历史
        self.trainer.plot_training_history(
            save_path=os.path.join(self.config['output_dir'], 'training_history.png')
        )
        
        # 绘制混淆矩阵
        self.trainer.plot_confusion_matrix(
            results['confusion_matrix'],
            self.class_names,
            save_path=os.path.join(self.config['output_dir'], 'confusion_matrix.png')
        )
        
        print("图表已保存到结果目录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中文新闻文本分类训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--model', type=str, choices=['textcnn', 'bert'], default='bert', help='模型类型')
    parser.add_argument('--data', type=str, default='data/raw/Train.txt', help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = NewsClassificationTrainer(args.config)
    
    # 更新配置
    trainer.config['model_type'] = args.model
    trainer.config['data_path'] = args.data
    trainer.config['num_epochs'] = args.epochs
    trainer.config['batch_size'] = args.batch_size
    trainer.config['learning_rate'] = args.lr
    trainer.config['output_dir'] = args.output
    
    try:
        # 加载和预处理数据
        trainer.load_and_preprocess_data()
        
        # 创建模型
        trainer.create_model()
        
        # 训练模型
        results, history = trainer.train_model()
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"最终测试准确率: {results['accuracy']:.4f}")
        print(f"最终F1分数: {results['f1_score']:.4f}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
