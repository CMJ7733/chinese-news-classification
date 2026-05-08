"""
文本向量化模块
支持Word2Vec和BERT两种向量化方法
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import pickle
import os
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置HuggingFace镜像源
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class Word2VecVectorizer:
    """Word2Vec向量化器"""
    
    def __init__(self, 
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 2,
                 workers: int = 4,
                 sg: int = 0):
        """
        初始化Word2Vec向量化器
        
        Args:
            vector_size: 词向量维度
            window: 窗口大小
            min_count: 最小词频
            workers: 并行线程数
            sg: 训练算法，0=CBOW, 1=Skip-gram
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
    
    def fit(self, texts: List[List[str]]):
        """
        训练Word2Vec模型
        
        Args:
            texts: 分词后的文本列表
        """
        print("开始训练Word2Vec模型...")
        
        # 训练Word2Vec模型
        self.model = Word2Vec(
            sentences=texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        
        # 构建词汇表
        self.vocab_size = len(self.model.wv)
        self.word2idx = {word: idx for idx, word in enumerate(self.model.wv.index_to_key)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Word2Vec模型训练完成，词汇表大小: {self.vocab_size}")
    
    def transform(self, texts: List[List[str]]) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            texts: 分词后的文本列表
            
        Returns:
            文本向量矩阵
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        vectors = []
        
        for text in texts:
            if not text:
                # 空文本用零向量填充
                vectors.append(np.zeros(self.vector_size))
                continue
            
            # 计算文本中所有词向量的平均值
            word_vectors = []
            for word in text:
                if word in self.word2idx:
                    word_vectors.append(self.model.wv[word])
            
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(self.vector_size)
            
            vectors.append(text_vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[List[str]]) -> np.ndarray:
        """训练模型并转换文本"""
        self.fit(texts)
        return self.transform(texts)
    
    def save_model(self, save_path: str):
        """保存模型"""
        if self.model is not None:
            self.model.save(save_path)
            
            # 保存其他属性
            model_info = {
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'workers': self.workers,
                'sg': self.sg,
                'vocab_size': self.vocab_size,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word
            }
            
            with open(save_path + '_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print(f"Word2Vec模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型"""
        self.model = Word2Vec.load(load_path)
        
        # 加载其他属性
        with open(load_path + '_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        self.vector_size = model_info['vector_size']
        self.window = model_info['window']
        self.min_count = model_info['min_count']
        self.workers = model_info['workers']
        self.sg = model_info['sg']
        self.vocab_size = model_info['vocab_size']
        self.word2idx = model_info['word2idx']
        self.idx2word = model_info['idx2word']
        
        print(f"Word2Vec模型已从 {load_path} 加载")


class BERTVectorizer:
    """BERT向量化器"""
    
    def __init__(self, 
                 model_name: str = 'bert-base-chinese',
                 max_length: int = 512,
                 batch_size: int = 32,
                 device: str = 'auto'):
        """
        初始化BERT向量化器
        
        Args:
            model_name: BERT模型名称
            max_length: 最大序列长度
            batch_size: 批处理大小
            device: 设备类型
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"BERT模型已加载: {model_name}")
        print(f"使用设备: {self.device}")
    
    def _tokenize_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对文本进行tokenization
        
        Args:
            texts: 文本列表
            
        Returns:
            (input_ids, attention_masks)
        """
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # 使用BERT tokenizer
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为BERT向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量矩阵
        """
        print("开始使用BERT提取文本向量...")
        
        vectors = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenization
            input_ids, attention_masks = self._tokenize_texts(batch_texts)
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            # 获取BERT输出
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)
                # 使用[CLS]标记的向量作为句子表示
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(batch_vectors)
        
        return np.vstack(vectors)
    
    def transform_from_words(self, texts: List[List[str]]) -> np.ndarray:
        """
        从分词结果转换为BERT向量
        
        Args:
            texts: 分词后的文本列表
            
        Returns:
            文本向量矩阵
        """
        # 将分词结果重新组合成文本
        combined_texts = [' '.join(words) for words in texts]
        return self.transform(combined_texts)


class TFIDFVectorizer:
    """TF-IDF向量化器"""
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95):
        """
        初始化TF-IDF向量化器
        
        Args:
            max_features: 最大特征数
            ngram_range: n-gram范围
            min_df: 最小文档频率
            max_df: 最大文档频率
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        self.scaler = StandardScaler()
    
    def fit(self, texts: List[List[str]]):
        """
        训练TF-IDF模型
        
        Args:
            texts: 分词后的文本列表
        """
        # 将分词结果转换为字符串
        text_strings = [' '.join(words) for words in texts]
        
        # 训练TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(text_strings)
        
        # 标准化
        self.scaler.fit(tfidf_matrix.toarray())
        
        print(f"TF-IDF模型训练完成，特征数: {tfidf_matrix.shape[1]}")
    
    def transform(self, texts: List[List[str]]) -> np.ndarray:
        """
        将文本转换为TF-IDF向量
        
        Args:
            texts: 分词后的文本列表
            
        Returns:
            文本向量矩阵
        """
        # 将分词结果转换为字符串
        text_strings = [' '.join(words) for words in texts]
        
        # 转换为TF-IDF向量
        tfidf_matrix = self.vectorizer.transform(text_strings)
        
        # 标准化
        return self.scaler.transform(tfidf_matrix.toarray())
    
    def fit_transform(self, texts: List[List[str]]) -> np.ndarray:
        """训练模型并转换文本"""
        self.fit(texts)
        return self.transform(texts)


class VectorizerFactory:
    """向量化器工厂类"""
    
    @staticmethod
    def create_vectorizer(vectorizer_type: str, **kwargs):
        """
        创建向量化器
        
        Args:
            vectorizer_type: 向量化器类型 ('word2vec', 'bert', 'tfidf')
            **kwargs: 向量化器参数
            
        Returns:
            向量化器实例
        """
        if vectorizer_type.lower() == 'word2vec':
            return Word2VecVectorizer(**kwargs)
        elif vectorizer_type.lower() == 'bert':
            return BERTVectorizer(**kwargs)
        elif vectorizer_type.lower() == 'tfidf':
            return TFIDFVectorizer(**kwargs)
        else:
            raise ValueError(f"不支持的向量化器类型: {vectorizer_type}")


if __name__ == "__main__":
    # 测试向量化器
    from preprocessing import create_sample_data, ChineseTextPreprocessor
    
    # 创建示例数据
    df = create_sample_data()
    preprocessor = ChineseTextPreprocessor()
    processed_texts, encoded_labels = preprocessor.preprocess_dataframe(df)
    
    print("测试Word2Vec向量化器:")
    word2vec_vectorizer = Word2VecVectorizer(vector_size=50)
    word2vec_vectors = word2vec_vectorizer.fit_transform(processed_texts)
    print(f"Word2Vec向量形状: {word2vec_vectors.shape}")
    
    print("\n测试TF-IDF向量化器:")
    tfidf_vectorizer = TFIDFVectorizer(max_features=1000)
    tfidf_vectors = tfidf_vectorizer.fit_transform(processed_texts)
    print(f"TF-IDF向量形状: {tfidf_vectors.shape}")
    
    print("\n测试BERT向量化器:")
    try:
        bert_vectorizer = BERTVectorizer()
        # 将分词结果重新组合
        combined_texts = [' '.join(words) for words in processed_texts]
        bert_vectors = bert_vectorizer.transform(combined_texts)
        print(f"BERT向量形状: {bert_vectors.shape}")
    except Exception as e:
        print(f"BERT向量化器测试失败: {e}")
