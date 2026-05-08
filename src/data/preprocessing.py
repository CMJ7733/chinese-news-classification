"""
中文文本预处理模块
包含分词、清洗、标准化等功能
"""

import re
import jieba
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class ChineseTextPreprocessor:
    """中文文本预处理器"""
    
    def __init__(self, 
                 stopwords_path: Optional[str] = None,
                 min_length: int = 1,
                 max_length: int = 50):
        """
        初始化预处理器
        
        Args:
            stopwords_path: 停用词文件路径
            min_length: 最小文本长度
            max_length: 最大文本长度
        """
        self.stopwords = set()
        self.min_length = min_length
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
        # 加载停用词
        if stopwords_path and os.path.exists(stopwords_path):
            self.load_stopwords(stopwords_path)
        else:
            # 使用默认停用词
            self._load_default_stopwords()
    
    def _load_default_stopwords(self):
        """加载默认停用词"""
        default_stopwords = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        ]
        self.stopwords = set(default_stopwords)
    
    def load_stopwords(self, stopwords_path: str):
        """从文件加载停用词"""
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f if line.strip()])
        except Exception as e:
            print(f"加载停用词失败: {e}")
            self._load_default_stopwords()
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除电话号码
        text = re.sub(r'\d{3,4}-\d{7,8}|\d{11}', '', text)
        
        # 移除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 清洗后的文本
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤停用词和短词
        words = [word for word in words 
                if word not in self.stopwords 
                and len(word) >= self.min_length
                and word.strip()]
        
        return words
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        完整的文本预处理流程
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的词列表
        """
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 检查长度
        if len(cleaned_text) < self.min_length or len(cleaned_text) > self.max_length:
            return []
        
        # 分词
        words = self.segment_text(cleaned_text)
        
        return words
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'text',
                           label_column: str = 'label') -> Tuple[List[List[str]], np.ndarray]:
        """
        预处理DataFrame数据
        
        Args:
            df: 包含文本和标签的DataFrame
            text_column: 文本列名
            label_column: 标签列名
            
        Returns:
            (预处理后的文本列表, 标签数组)
        """
        print("开始预处理数据...")
        
        # 预处理文本
        processed_texts = []
        valid_labels = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            words = self.preprocess_text(text)
            
            if words:  # 只保留非空的文本
                processed_texts.append(words)
                valid_labels.append(row[label_column])
        
        # 编码标签
        if valid_labels:
            encoded_labels = self.label_encoder.fit_transform(valid_labels)
        else:
            encoded_labels = np.array([])
        
        print(f"预处理完成，有效样本数: {len(processed_texts)}")
        print(f"类别数: {len(self.label_encoder.classes_)}")
        print(f"类别: {self.label_encoder.classes_}")
        
        return processed_texts, encoded_labels
    
    def save_preprocessor(self, save_path: str):
        """保存预处理器"""
        preprocessor_data = {
            'stopwords': self.stopwords,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'label_encoder': self.label_encoder
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"预处理器已保存到: {save_path}")
    
    def load_preprocessor(self, load_path: str):
        """加载预处理器"""
        with open(load_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.stopwords = preprocessor_data['stopwords']
        self.min_length = preprocessor_data['min_length']
        self.max_length = preprocessor_data['max_length']
        self.label_encoder = preprocessor_data['label_encoder']
        
        print(f"预处理器已从 {load_path} 加载")


def create_sample_data():
    """创建示例数据"""
    sample_data = {
        'text': [
            '今天股市大涨，投资者信心增强，科技股表现尤为突出。',
            '中国足球队在世界杯预选赛中表现出色，以3:1战胜对手。',
            '最新的人工智能技术突破，将推动自动驾驶汽车的发展。',
            '央行宣布降准，释放流动性支持实体经济发展。',
            '奥运会游泳比赛中，中国选手打破世界纪录获得金牌。',
            '5G网络建设加速，为智慧城市建设提供技术支撑。',
            '新能源汽车销量持续增长，环保理念深入人心。',
            '电影票房创新高，国产影片质量不断提升。'
        ],
        'label': [
            '财经', '体育', '科技', '财经', '体育', '科技', '科技', '娱乐'
        ]
    }
    
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # 测试预处理器
    preprocessor = ChineseTextPreprocessor()
    
    # 创建示例数据
    df = create_sample_data()
    print("原始数据:")
    print(df)
    
    # 预处理数据
    processed_texts, encoded_labels = preprocessor.preprocess_dataframe(df)
    
    print("\n预处理结果:")
    for i, (text, label) in enumerate(zip(processed_texts, encoded_labels)):
        print(f"样本 {i+1}: {text} -> 标签: {label}")
