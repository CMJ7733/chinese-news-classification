"""
数据加载和预处理脚本
支持多种数据格式的加载和预处理
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import jieba
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class NewsDataLoader:
    """新闻数据加载器"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.data = None
        self.label_encoder = LabelEncoder()
    
    def load_csv(self, file_path: str, 
                 text_column: str = 'text',
                 label_column: str = 'label',
                 encoding: str = 'utf-8') -> pd.DataFrame:
        """
        加载CSV文件
        
        Args:
            file_path: 文件路径
            text_column: 文本列名
            label_column: 标签列名
            encoding: 文件编码
            
        Returns:
            数据DataFrame
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功加载CSV文件: {file_path}")
            print(f"数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            return None
    
    def load_json(self, file_path: str,
                  text_field: str = 'text',
                  label_field: str = 'label',
                  encoding: str = 'utf-8') -> pd.DataFrame:
        """
        加载JSON文件
        
        Args:
            file_path: 文件路径
            text_field: 文本字段名
            label_field: 标签字段名
            encoding: 文件编码
            
        Returns:
            数据DataFrame
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            print(f"成功加载JSON文件: {file_path}")
            print(f"数据形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return None
    
    def load_txt(self, file_path: str,
                 separator: str = '\t',
                 text_column: int = 2,
                 label_column: int = 1,
                 id_column: int = 0,
                 encoding: str = 'utf-8') -> pd.DataFrame:
        """
        加载TXT文件 (格式: ID\t类别\t文本内容)
        
        Args:
            file_path: 文件路径
            separator: 分隔符
            text_column: 文本列索引 (默认为2，即第三列)
            label_column: 标签列索引 (默认为1，即第二列)
            id_column: ID列索引 (默认为0，即第一列)
            encoding: 文件编码
            
        Returns:
            数据DataFrame
        """
        try:
            data = []
            with open(file_path, 'r', encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        parts = line.split(separator)
                        if len(parts) >= 3:  # 确保有ID、类别、文本三列
                            data.append({
                                'id': parts[id_column],
                                'text': parts[text_column],
                                'label': parts[label_column]
                            })
                        elif len(parts) == 2:  # 兼容两列格式
                            data.append({
                                'id': line_num,
                                'text': parts[text_column if text_column < len(parts) else 0],
                                'label': parts[label_column if label_column < len(parts) else 1]
                            })
            
            df = pd.DataFrame(data)
            print(f"成功加载TXT文件: {file_path}")
            print(f"数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"加载TXT文件失败: {e}")
            return None
    
    def load_data(self, 
                  file_path: str,
                  file_type: str = 'auto',
                  **kwargs) -> pd.DataFrame:
        """
        自动加载数据文件
        
        Args:
            file_path: 文件路径
            file_type: 文件类型 ('csv', 'json', 'txt', 'auto')
            **kwargs: 其他参数
            
        Returns:
            数据DataFrame
        """
        if file_type == 'auto':
            file_type = file_path.split('.')[-1].lower()
        
        if file_type == 'csv':
            return self.load_csv(file_path, **kwargs)
        elif file_type == 'json':
            return self.load_json(file_path, **kwargs)
        elif file_type == 'txt':
            return self.load_txt(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    def preprocess_data(self, 
                       df: pd.DataFrame,
                       text_column: str = 'text',
                       label_column: str = 'label',
                       min_length: int = 10,
                       max_length: int = 1000) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df: 原始数据
            text_column: 文本列名
            label_column: 标签列名
            min_length: 最小文本长度
            max_length: 最大文本长度
            
        Returns:
            预处理后的数据
        """
        print("开始预处理数据...")
        
        # 移除空值
        df = df.dropna(subset=[text_column, label_column])
        
        # 移除重复项
        df = df.drop_duplicates(subset=[text_column])
        
        # 过滤文本长度
        df = df[df[text_column].str.len() >= min_length]
        df = df[df[text_column].str.len() <= max_length]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        print(f"预处理后数据形状: {df.shape}")
        print(f"类别分布:")
        print(df[label_column].value_counts())
        
        return df
    
    def split_data(self, 
                   df: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分数据集
        
        Args:
            df: 数据
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            stratify: 是否分层抽样
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        stratify_column = df['label'] if stratify else None
        
        # 先划分训练+验证集和测试集
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_column
        )
        
        # 再划分训练集和验证集
        if stratify:
            stratify_column = train_val_df['label']
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=stratify_column
        )
        
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print(f"测试集大小: {len(test_df)}")
        
        return train_df, val_df, test_df


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, 
                 stopwords_path: Optional[str] = None,
                 min_length: int = 2,
                 max_length: int = 512):
        """
        初始化文本处理器
        
        Args:
            stopwords_path: 停用词文件路径
            min_length: 最小词长度
            max_length: 最大词长度
        """
        self.stopwords = set()
        self.min_length = min_length
        self.max_length = max_length
        
        # 加载停用词
        if stopwords_path and os.path.exists(stopwords_path):
            self.load_stopwords(stopwords_path)
        else:
            self._load_default_stopwords()
    
    def _load_default_stopwords(self):
        """加载默认停用词"""
        default_stopwords = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
            '自己', '这', '那', '他', '她', '它', '们', '我们', '你们', '他们', '这个', 
            '那个', '这样', '那样', '什么', '怎么', '为什么', '因为', '所以', '但是', 
            '然后', '如果', '虽然', '因为', '所以', '但是', '然后', '如果', '虽然'
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
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        import re
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
        """中文分词"""
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
    
    def process_texts(self, texts: List[str]) -> List[List[str]]:
        """批量处理文本"""
        processed_texts = []
        
        for text in tqdm(texts, desc="处理文本"):
            # 清洗文本
            cleaned_text = self.clean_text(text)
            
            # 检查长度
            if len(cleaned_text) < self.min_length or len(cleaned_text) > self.max_length:
                processed_texts.append([])
                continue
            
            # 分词
            words = self.segment_text(cleaned_text)
            processed_texts.append(words)
        
        return processed_texts


def create_sample_dataset(save_path: str = 'data/raw/sample_news.csv'):
    """创建示例数据集"""
    sample_data = {
        'text': [
            '今天股市大涨，投资者信心增强，科技股表现尤为突出，市场成交量明显放大。',
            '中国足球队在世界杯预选赛中表现出色，以3:1战胜对手，晋级下一轮比赛。',
            '最新的人工智能技术突破，将推动自动驾驶汽车的发展，改变交通出行方式。',
            '央行宣布降准，释放流动性支持实体经济发展，银行股应声上涨。',
            '奥运会游泳比赛中，中国选手打破世界纪录获得金牌，为国争光。',
            '5G网络建设加速，为智慧城市建设提供技术支撑，推动数字化转型。',
            '新能源汽车销量持续增长，环保理念深入人心，传统车企加速转型。',
            '电影票房创新高，国产影片质量不断提升，文化产业发展迅速。',
            '教育部门发布新政策，促进教育公平，提高教育质量。',
            '医疗技术不断进步，新药研发取得重大突破，为患者带来希望。',
            '体育产业快速发展，全民健身意识增强，体育消费持续增长。',
            '科技创新推动经济发展，新兴产业成为增长新动能。',
            '金融科技发展迅速，移动支付普及，金融服务更加便捷。',
            '环保政策持续加码，绿色发展成为主旋律，清洁能源占比提升。',
            '文化创意产业蓬勃发展，传统文化与现代科技深度融合。'
        ],
        'label': [
            '财经', '体育', '科技', '财经', '体育', '科技', '科技', '娱乐', 
            '教育', '医疗', '体育', '科技', '财经', '环保', '文化'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存数据
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"示例数据集已保存到: {save_path}")
    
    return df


if __name__ == "__main__":
    # 测试数据加载器
    loader = NewsDataLoader()
    
    # 创建示例数据
    sample_df = create_sample_dataset()
    
    # 加载数据
    df = loader.load_data('data/raw/sample_news.csv')
    
    # 预处理数据
    df_processed = loader.preprocess_data(df)
    
    # 划分数据集
    train_df, val_df, test_df = loader.split_data(df_processed)
    
    # 测试文本处理器
    processor = TextProcessor()
    texts = df_processed['text'].tolist()
    processed_texts = processor.process_texts(texts)
    
    print(f"处理前文本示例: {texts[0]}")
    print(f"处理后文本示例: {processed_texts[0]}")
