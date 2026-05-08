"""
模型训练和评估模块
支持TextCNN和BERT模型的训练、验证和测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        """
        初始化早停
        
        Args:
            patience: 耐心值
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            
        Returns:
            是否应该早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存检查点"""
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 save_dir: str = 'models'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 设备类型
            save_dir: 模型保存目录
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.model.to(self.device)
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # 获取数据
            if len(batch) == 2:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
            elif len(batch) == 3:
                # BERT模型
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, 
                      val_loader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            criterion: 损失函数
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if len(batch) == 2:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                elif len(batch) == 3:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 0.001,
              early_stopping_patience: int = 5,
              save_best: bool = True) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
            save_best: 是否保存最佳模型
            
        Returns:
            训练历史
        """
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 早停机制
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            
            # 早停检查
            if early_stopping(val_loss, self.model):
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        return self.train_history
    
    def evaluate(self, 
                test_loader: DataLoader,
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            class_names: 类别名称列表
            
        Returns:
            评估结果
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if len(batch) == 2:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    
                elif len(batch) == 3:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 分类报告
        if class_names:
            report = classification_report(
                all_labels, all_predictions, 
                target_names=class_names, 
                output_dict=True
            )
        else:
            report = classification_report(
                all_labels, all_predictions, 
                output_dict=True
            )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
        
        return results
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'train_history': self.train_history
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        print(f"模型已从 {filepath} 加载")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_history['train_loss'], label='训练损失')
        ax1.plot(self.train_history['val_loss'], label='验证损失')
        ax1.set_title('模型损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_history['train_acc'], label='训练准确率')
        ax2.plot(self.train_history['val_acc'], label='验证准确率')
        ax2.set_title('模型准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray, 
                             class_names: List[str],
                             save_path: Optional[str] = None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()


def create_data_loaders(X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_val: np.ndarray, 
                       y_val: np.ndarray,
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       batch_size: int = 32,
                       model_type: str = 'textcnn') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        X_test: 测试特征
        y_test: 测试标签
        batch_size: 批处理大小
        model_type: 模型类型
        
    Returns:
        (训练加载器, 验证加载器, 测试加载器)
    """
    if model_type.lower() == 'textcnn':
        # TextCNN使用2D数据
        train_dataset = TensorDataset(
            torch.LongTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.LongTensor(X_test),
            torch.LongTensor(y_test)
        )
    
    elif model_type.lower() == 'bert':
        # BERT使用3D数据 (input_ids, attention_mask, labels)
        train_dataset = TensorDataset(
            torch.LongTensor(X_train[0]),  # input_ids
            torch.LongTensor(X_train[1]),  # attention_mask
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X_val[0]),
            torch.LongTensor(X_val[1]),
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.LongTensor(X_test[0]),
            torch.LongTensor(X_test[1]),
            torch.LongTensor(y_test)
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试训练器
    from models.textcnn import TextCNN, TextCNNConfig
    
    # 创建示例数据
    X_train = np.random.randint(0, 1000, (100, 50))
    y_train = np.random.randint(0, 5, 100)
    X_val = np.random.randint(0, 1000, (20, 50))
    y_val = np.random.randint(0, 5, 20)
    X_test = np.random.randint(0, 1000, (20, 50))
    y_test = np.random.randint(0, 5, 20)
    
    # 创建模型
    config = TextCNNConfig(vocab_size=1000, num_classes=5)
    model = TextCNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes
    )
    
    # 创建训练器
    trainer = ModelTrainer(model)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=16, model_type='textcnn'
    )
    
    # 训练模型
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=3,
        learning_rate=0.001
    )
    
    # 评估模型
    results = trainer.evaluate(test_loader)
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"F1分数: {results['f1_score']:.4f}")
