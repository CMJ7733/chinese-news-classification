# 基于深度学习的中文新闻文本分类

本项目利用深度学习技术对中文新闻文本进行自动分类，将新闻文档精确划分至预定义的类别（如体育、财经、科技等）。项目涵盖了中文分词、文本向量化（Word2Vec和BERT）、TextCNN和BERT分类模型等关键组件。

## 项目特点

- 🚀 **多种模型支持**: 支持TextCNN和BERT两种主流深度学习模型
- 🔤 **中文优化**: 专门针对中文文本处理进行优化，支持jieba分词
- 📊 **完整流程**: 从数据预处理到模型训练、评估的完整机器学习流程
- 📈 **可视化分析**: 提供丰富的可视化工具和性能分析
- ⚙️ **灵活配置**: 支持通过配置文件自定义模型参数
- 📝 **详细文档**: 完整的代码注释和使用说明

## 项目结构

```
神经网络项目/
├── src/                          # 源代码目录
│   ├── data/                     # 数据处理模块
│   │   ├── preprocessing.py      # 文本预处理
│   │   ├── vectorization.py     # 文本向量化
│   │   └── data_loader.py       # 数据加载
│   ├── models/                   # 模型定义
│   │   ├── textcnn.py           # TextCNN模型
│   │   └── bert_classifier.py   # BERT分类器
│   ├── training/                 # 训练模块
│   │   └── trainer.py           # 模型训练器
│   └── utils/                    # 工具函数
│       └── visualization.py     # 可视化工具
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   └── processed/                # 预处理后数据
├── models/                       # 模型保存目录
├── results/                      # 结果输出目录
├── configs/                      # 配置文件
│   └── default_config.json      # 默认配置
├── train.py                      # 训练脚本
├── requirements.txt              # 依赖包
└── README.md                     # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

将您的新闻数据放在 `data/raw/` 目录下，支持以下格式：
- CSV文件：包含 `text` 和 `label` 列
- JSON文件：包含 `text` 和 `label` 字段
- TXT文件：每行格式为 "文本\t标签"

如果没有数据，程序会自动创建示例数据集。

### 2. 训练模型

#### 训练所有模型
```bash
python train.py --data_path data/raw/your_data.csv --model_type both
```

#### 只训练TextCNN模型
```bash
python train.py --data_path data/raw/your_data.csv --model_type textcnn
```

#### 只训练BERT模型
```bash
python train.py --data_path data/raw/your_data.csv --model_type bert
```

### 3. 查看结果

训练完成后，结果会保存在 `results/` 目录下：
- `textcnn_results.json`: TextCNN模型结果
- `bert_results.json`: BERT模型结果
- `analysis_report.html`: 分析报告
- `textcnn_model/`: TextCNN模型文件
- `bert_model/`: BERT模型文件

## 配置说明

可以通过修改 `configs/default_config.json` 来自定义模型参数：

### TextCNN配置
```json
{
  "textcnn": {
    "vectorizer": {
      "vector_size": 100,      # 词向量维度
      "window": 5,             # 窗口大小
      "min_count": 2           # 最小词频
    },
    "model": {
      "embedding_dim": 100,    # 嵌入维度
      "filter_sizes": [3, 4, 5], # 卷积核大小
      "num_filters": 100,      # 卷积核数量
      "dropout_rate": 0.5      # Dropout比率
    },
    "batch_size": 32,          # 批处理大小
    "num_epochs": 10,          # 训练轮数
    "learning_rate": 0.001     # 学习率
  }
}
```

### BERT配置
```json
{
  "bert": {
    "model": {
      "model_name": "bert-base-chinese", # BERT模型名称
      "dropout_rate": 0.3,               # Dropout比率
      "freeze_bert": false               # 是否冻结BERT参数
    },
    "batch_size": 16,                    # 批处理大小
    "num_epochs": 3,                     # 训练轮数
    "learning_rate": 2e-5                # 学习率
  }
}
```

## 模型说明

### TextCNN模型
- 基于卷积神经网络的文本分类模型
- 使用多个不同大小的卷积核捕获局部特征
- 适合处理中等长度的文本
- 训练速度快，资源消耗少

### BERT模型
- 基于预训练BERT的中文文本分类模型
- 利用BERT的强大表示能力
- 适合处理各种长度的文本
- 通常能获得更好的分类效果

## 功能特性

### 文本预处理
- 中文分词（jieba）
- 文本清洗（移除HTML、URL等）
- 停用词过滤
- 文本长度控制

### 文本向量化
- Word2Vec词向量
- BERT预训练向量
- TF-IDF向量

### 模型训练
- 自动早停机制
- 学习率调度
- 模型检查点保存
- 训练过程可视化

### 评估指标
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 混淆矩阵

### 可视化分析
- 类别分布图
- 文本长度分布
- 词频统计
- 词云图
- 训练历史曲线
- 混淆矩阵热力图
- 嵌入向量可视化

## 使用示例

### 基本使用
```python
from src.data.data_loader import NewsDataLoader
from src.models.textcnn import create_textcnn_model, TextCNNConfig
from src.training.trainer import ModelTrainer

# 加载数据
loader = NewsDataLoader()
df = loader.load_data('data/raw/news.csv')
df = loader.preprocess_data(df)

# 创建模型
config = TextCNNConfig(vocab_size=10000, num_classes=5)
model = create_textcnn_model(config)

# 训练模型
trainer = ModelTrainer(model)
# ... 训练过程
```

### 自定义预处理
```python
from src.data.preprocessing import ChineseTextPreprocessor

# 创建预处理器
preprocessor = ChineseTextPreprocessor(
    stopwords_path='data/stopwords.txt',
    min_length=5,
    max_length=500
)

# 预处理文本
processed_texts, labels = preprocessor.preprocess_dataframe(df)
```

### 可视化分析
```python
from src.utils.visualization import TextVisualizer

# 创建可视化器
visualizer = TextVisualizer()

# 绘制类别分布
visualizer.plot_class_distribution(labels)

# 创建词云
visualizer.create_word_cloud(texts)
```

## 性能优化建议

1. **数据质量**: 确保数据质量，移除噪声和重复样本
2. **文本长度**: 根据任务调整文本长度限制
3. **模型选择**: 
   - 短文本：TextCNN
   - 长文本：BERT
   - 资源受限：TextCNN
4. **超参数调优**: 根据验证集表现调整学习率、批大小等
5. **数据增强**: 可以尝试数据增强技术提高模型泛化能力

## 常见问题

### Q: 如何添加新的数据格式？
A: 在 `src/data/data_loader.py` 中添加新的加载方法。

### Q: 如何修改模型架构？
A: 在 `src/models/` 目录下修改对应的模型文件。

### Q: 如何添加新的评估指标？
A: 在 `src/training/trainer.py` 的 `evaluate` 方法中添加。

### Q: 训练过程中出现内存不足怎么办？
A: 减小批处理大小（batch_size）或使用梯度累积。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

MIT License

## 联系方式

如有问题，请通过Issue联系。

---

**注意**: 本项目仅供学习和研究使用，请遵守相关法律法规和学术道德规范。
