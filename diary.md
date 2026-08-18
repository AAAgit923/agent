# 日记
### 8.12
• 看李沐《Attention Is All You Need》论文精讲视频 
• 手画 Transformer 架构图（Encoder-Decoder、自注意力、残差连接、LayerNorm）
• 用箭头标注数据流向，理解为什么比 RNN 好

### 8.13
 看李宏毅 GenAI 课 — Attention 部分
• 手写 Q/K/V 计算过程（手算一个简单例子）
• 理解 Multi-Head Attention 和 Scaled Dot-Product Attention

### 8.14
读 OpenAI/Anthropic 官方文档关于 Tokenizer 的说明
• 理解温度 / Top-p / Top-K 采样的区别
• pip install tiktoken 统计任意文本的 Token 数
• 对比不同模型的 Context Window 长度

### 8.15
了解 GPT-4o / Qwen2.5 / DeepSeek-V3 / LLaMA4 的核心差异
• 读 DeepSeek-V3 技术报告（重点 MoE 稀疏激活）
• 整理一张"主流模型横向对比表"

### 8.16
结合 CV 底子，了解 GPT-4o 视觉 / Qwen-VL / LLaVA
• 理解"视觉编码器 + LLM"的经典架构
• 跑通一个多模态 LLM API 调用（识图 + 问答）

### 8.17
用 API Key 跑通第一次 LLM 调用
练习不同参数（temperature / max_tokens / top_p）
整理 W1 学习笔记，画 LLM 知识 MindMap