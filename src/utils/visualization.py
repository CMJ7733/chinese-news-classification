"""
可视化分析工具
提供各种图表和可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import wordcloud
from wordcloud import WordCloud
import jieba
from collections import Counter
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TextVisualizer:
    """文本可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图表大小
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_class_distribution(self, 
                               labels: List[str],
                               title: str = "类别分布",
                               save_path: Optional[str] = None):
        """
        绘制类别分布图
        
        Args:
            labels: 标签列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 统计类别分布
        label_counts = Counter(labels)
        labels_list = list(label_counts.keys())
        counts_list = list(label_counts.values())
        
        # 创建柱状图
        bars = plt.bar(labels_list, counts_list, color=self.colors[:len(labels_list)])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_text_length_distribution(self, 
                                    texts: List[str],
                                    title: str = "文本长度分布",
                                    save_path: Optional[str] = None):
        """
        绘制文本长度分布图
        
        Args:
            texts: 文本列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 计算文本长度
        text_lengths = [len(text) for text in texts]
        
        # 创建直方图
        plt.hist(text_lengths, bins=50, color=self.colors[0], alpha=0.7, edgecolor='black')
        
        # 添加统计信息
        mean_length = np.mean(text_lengths)
        median_length = np.median(text_lengths)
        
        plt.axvline(mean_length, color='red', linestyle='--', 
                   label=f'平均长度: {mean_length:.1f}')
        plt.axvline(median_length, color='green', linestyle='--', 
                   label=f'中位数长度: {median_length:.1f}')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('文本长度（字符数）', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"文本长度分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_word_frequency(self, 
                           texts: List[str],
                           top_n: int = 20,
                           title: str = "词频统计",
                           save_path: Optional[str] = None):
        """
        绘制词频统计图
        
        Args:
            texts: 文本列表
            top_n: 显示前N个词
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 分词并统计词频
        all_words = []
        for text in texts:
            words = jieba.lcut(text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        
        words, counts = zip(*top_words)
        
        # 创建水平柱状图
        bars = plt.barh(range(len(words)), counts, color=self.colors[0])
        
        # 设置y轴标签
        plt.yticks(range(len(words)), words)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('词频', fontsize=12)
        plt.ylabel('词语', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"词频统计图已保存到: {save_path}")
        
        plt.show()
    
    def create_word_cloud(self, 
                         texts: List[str],
                         title: str = "词云图",
                         save_path: Optional[str] = None,
                         width: int = 800,
                         height: int = 400):
        """
        创建词云图
        
        Args:
            texts: 文本列表
            title: 图表标题
            save_path: 保存路径
            width: 图片宽度
            height: 图片高度
        """
        # 合并所有文本
        all_text = ' '.join(texts)
        
        # 创建词云
        wordcloud = WordCloud(
            font_path='simhei.ttf',  # 中文字体
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"词云图已保存到: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                             y_true: List[int],
                             y_pred: List[int],
                             class_names: List[str],
                             title: str = "混淆矩阵",
                             save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 创建热力图
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             title: str = "训练历史",
                             save_path: Optional[str] = None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            title: 图表标题
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['train_loss'], label='训练损失', color=self.colors[0])
        ax1.plot(history['val_loss'], label='验证损失', color=self.colors[1])
        ax1.set_title('模型损失', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(history['train_acc'], label='训练准确率', color=self.colors[0])
        ax2.plot(history['val_acc'], label='验证准确率', color=self.colors[1])
        ax2.set_title('模型准确率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, 
                               features: List[str],
                               importance_scores: List[float],
                               title: str = "特征重要性",
                               top_n: int = 20,
                               save_path: Optional[str] = None):
        """
        绘制特征重要性图
        
        Args:
            features: 特征名称列表
            importance_scores: 重要性分数列表
            title: 图表标题
            top_n: 显示前N个特征
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 创建DataFrame并排序
        df = pd.DataFrame({
            'feature': features,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # 创建水平柱状图
        bars = plt.barh(range(len(df)), df['importance'], color=self.colors[0])
        
        # 设置y轴标签
        plt.yticks(range(len(df)), df['feature'])
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('重要性分数', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        
        plt.show()
    
    def plot_embedding_visualization(self, 
                                   embeddings: np.ndarray,
                                   labels: List[int],
                                   class_names: List[str],
                                   method: str = 'tsne',
                                   title: str = "嵌入可视化",
                                   save_path: Optional[str] = None):
        """
        绘制嵌入可视化图
        
        Args:
            embeddings: 嵌入向量
            labels: 标签列表
            class_names: 类别名称
            method: 降维方法 ('tsne' 或 'pca')
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 降维
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 绘制散点图
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], 
                       embeddings_2d[mask, 1],
                       c=self.colors[i % len(self.colors)],
                       label=class_names[label],
                       alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(f'{method.upper()} 1', fontsize=12)
        plt.ylabel(f'{method.upper()} 2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"嵌入可视化图已保存到: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, 
                             model_names: List[str],
                             metrics: Dict[str, List[float]],
                             title: str = "模型性能对比",
                             save_path: Optional[str] = None):
        """
        绘制模型性能对比图
        
        Args:
            model_names: 模型名称列表
            metrics: 指标字典
            title: 图表标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            bars = axes[i].bar(model_names, values, color=self.colors[:len(model_names)])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
            
            axes[i].set_title(metric_name, fontsize=14, fontweight='bold')
            axes[i].set_ylabel('分数', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型对比图已保存到: {save_path}")
        
        plt.show()


def create_analysis_report(data: Dict[str, Any], 
                          save_path: str = 'results/analysis_report.html'):
    """
    创建分析报告
    
    Args:
        data: 分析数据
        save_path: 保存路径
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>中文新闻文本分类分析报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            .metric {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .highlight {{ background-color: #e8f4f8; padding: 15px; margin: 15px 0; border-left: 4px solid #3498db; }}
        </style>
    </head>
    <body>
        <h1>中文新闻文本分类分析报告</h1>
        
        <div class="highlight">
            <h2>数据集概览</h2>
            <p><strong>总样本数:</strong> {data.get('total_samples', 'N/A')}</p>
            <p><strong>类别数:</strong> {data.get('num_classes', 'N/A')}</p>
            <p><strong>平均文本长度:</strong> {data.get('avg_text_length', 'N/A')}</p>
        </div>
        
        <div class="metric">
            <h2>模型性能</h2>
            <p><strong>准确率:</strong> {data.get('accuracy', 'N/A'):.4f}</p>
            <p><strong>精确率:</strong> {data.get('precision', 'N/A'):.4f}</p>
            <p><strong>召回率:</strong> {data.get('recall', 'N/A'):.4f}</p>
            <p><strong>F1分数:</strong> {data.get('f1_score', 'N/A'):.4f}</p>
        </div>
        
        <div class="metric">
            <h2>训练信息</h2>
            <p><strong>训练轮数:</strong> {data.get('epochs', 'N/A')}</p>
            <p><strong>最佳验证准确率:</strong> {data.get('best_val_acc', 'N/A'):.4f}</p>
            <p><strong>训练时间:</strong> {data.get('training_time', 'N/A')}</p>
        </div>
        
        <div class="highlight">
            <h2>分析结论</h2>
            <p>{data.get('conclusion', '分析完成')}</p>
        </div>
    </body>
    </html>
    """
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存HTML报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"分析报告已保存到: {save_path}")


if __name__ == "__main__":
    # 测试可视化工具
    visualizer = TextVisualizer()
    
    # 创建示例数据
    sample_texts = [
        "今天股市大涨，投资者信心增强",
        "中国足球队在世界杯预选赛中表现出色",
        "最新的人工智能技术突破",
        "央行宣布降准，释放流动性",
        "奥运会游泳比赛中，中国选手打破世界纪录"
    ]
    
    sample_labels = ['财经', '体育', '科技', '财经', '体育']
    
    # 测试各种可视化功能
    visualizer.plot_class_distribution(sample_labels)
    visualizer.plot_text_length_distribution(sample_texts)
    visualizer.plot_word_frequency(sample_texts, top_n=10)
    visualizer.create_word_cloud(sample_texts)
    
    # 创建分析报告
    analysis_data = {
        'total_samples': len(sample_texts),
        'num_classes': len(set(sample_labels)),
        'avg_text_length': np.mean([len(text) for text in sample_texts]),
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1_score': 0.85,
        'epochs': 10,
        'best_val_acc': 0.88,
        'training_time': '2小时30分钟',
        'conclusion': '模型在中文新闻文本分类任务上表现良好，准确率达到85%。'
    }
    
    create_analysis_report(analysis_data)
