# Agent开发学习笔记：多模态大模型（视觉LLM）

**学习目标**：结合CV基础，掌握多模态LLM核心架构、主流模型差异，跑通识图问答API，建立Agent多模态感知基础能力

**前置基础**：CV基础（ViT、图像特征、Patch切片）、Python基础、大模型API调用基础

**学习时间**：单日专项学习（Agent开发入门）

## 一、核心认知：多模态LLM在Agent中的作用

传统文本Agent仅能处理文字输入，无法感知图像、屏幕截图、实景画面等视觉信息。**多模态大模型是Agent的“视觉眼睛”**，核心作用是将视觉图像信息转化为结构化文本信息，供给Agent的规划、推理、决策模块使用，是视觉智能Agent的核心基础。

所有视觉Agent的底层逻辑：**图像输入 → 多模态模型视觉理解 → 文本观察摘要 → Agent思考决策 → 执行动作**

## 二、主流视觉多模态模型对比（GPT\-4o / Qwen\-VL / LLaVA）

结合CV视觉基础，从架构、开源性、适用场景、Agent开发价值四个维度对比三大主流模型：

### 1\. LLaVA（开源经典标杆）

- **核心架构**：CLIP\-ViT视觉编码器 \+ 开源LLM（Vicuna等）\+ 线性Adapter投影层，是**视觉编码器\+LLM**范式的开山落地模型

- **核心特点**：架构极简、开源完全可本地部署、轻量化、推理速度快

- **短板**：复杂推理、细节理解能力弱，多轮对话能力一般

- **Agent适用场景**：本地轻量化Agent、端侧智能感知、低成本视觉推理任务

- **学习价值**：最适合理解多模态底层原理，所有开源视觉LLM的基础范式

### 2\. Qwen\-VL（通义千问多模态，国内首选）

- **核心架构**：自研高精度ViT视觉编码器 \+ 通义千问大语言模型，优化了大图、多图、文本细节识别

- **核心特点**：中文理解极强、支持多图输入、OCR识别、图表解析，同时支持API调用和本地部署

- **短板**：通用视觉推理能力略弱于GPT\-4o

- **Agent适用场景**：国内业务Agent、办公截图识别、文档图表解析、中文场景视觉问答

- **学习价值**：学生/开发者首选，免费额度充足，适配国内开发环境

### 3\. GPT\-4o Vision（闭源商用顶级模型）

- **核心架构**：打破传统双模块拆分，端到端统一多模态架构，图文音共享Token体系，无明显独立视觉编码器

- **核心特点**：通用视觉理解天花板、支持手写、公式、复杂图表、实景画面、视频帧理解

- **短板**：闭源、收费、无法本地部署、国内访问需特殊配置

- **Agent适用场景**：高精度智能Agent、复杂场景推理、通用视觉智能任务

- **学习价值**：行业落地标杆，用于体验顶级多模态感知能力

## 三、核心架构精讲：视觉编码器 \+ LLM 经典范式（必考核心）

LLaVA、Qwen\-VL等90%开源多模态模型均遵循该架构，是Agent视觉能力的底层原理，完美结合CV Transformer知识。

### 1\. 完整工作流水线

**原始图像 → Patch切片 → 视觉编码器（ViT） → 图像Embedding/Token → Adapter投影对齐 → 与文本Token拼接 → LLM推理生成文本**

### 2\. 模块拆解（CV知识点落地）

- **图像Patch切片**：将整张图片切分为固定大小的小块（如16×16），和ViT图像预处理逻辑一致

- **视觉编码器（ViT）**：CV核心模块，负责将像素图像转化为**视觉特征向量（图像Token）**，模型无法直接读取像素，只能读取向量特征

- **Adapter投影层（核心桥梁）**：解决视觉编码器输出向量维度与LLM文本Embedding维度不匹配的问题，是多模态融合的关键

- **模态拼接**：将处理好的图像Token序列 \+ 用户提问文本Token序列拼接，送入大语言模型

- **LLM推理**：大模型仅负责文本推理，全程看不到原始图片，只通过图像Token理解视觉信息并输出答案

### 3\. 关键核心结论（Agent开发关键）

1. LLM本身**不具备视觉能力**，视觉能力完全来自视觉编码器

2. 图像Token会占用LLM上下文窗口，图片分辨率越高、数量越多，上下文消耗越大，容易导致Agent长轮次推理溢出

3. 多模态模型的本质：**视觉翻译器**，把图像翻译成LLM可理解的文本特征

## 四、实战代码：多模态LLM识图\+问答 API 完整可运行Demo

提供两套主流方案，适配国内/海外环境，可直接复制运行，模拟Agent视觉感知能力

### 方案一：Qwen\-VL 阿里通义千问（国内推荐、免费可用）

**前置准备**：安装依赖 \+ 获取阿里Dashscope API Key（官网免费申领）

```Plain Text
# 安装依赖
pip install dashscope

# 完整可运行代码
import dashscope
from dashscope import MultiModalConversation

# 配置你的API Key
dashscope.api_key = "你的Dashscope_API_KEY"

def agent_vision_qa(image_url: str, question: str) -> str:
    """
    Agent视觉感知核心函数：识图+问答
    :param image_url: 公网可访问图片链接
    :param question:  Agent视角的提问（观察、总结、提取信息）
    :return: 模型视觉理解结果（供给Agent决策使用）
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": question}
            ]
        }
    ]
    # 调用多模态模型
    response = MultiModalConversation.call(model="qwen-vl-plus", messages=messages)
    # 结果解析
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        return f"视觉识别失败：{response.message}"

# 模拟Agent视觉观察任务
if __name__ == "__main__":
    # 测试图片公网URL，可自行替换
    test_img_url = "https://img0.baidu.com/it/u=1234567890,1234567890&fm=253&fmt=auto&app=138&f=JPEG?w=800&h=500"
    # Agent专用提问：生成结构化观察摘要
    res = agent_vision_qa(test_img_url, "请详细描述这张图片的内容，生成一段结构化观察摘要，用于Agent后续推理决策")
    print("【Agent视觉观察结果】")
    print(res)

```

### 方案二：GPT\-4o Vision 调用（海外顶级模型）

```Plain Text
# 安装依赖
pip install openai

# 完整可运行代码
from openai import OpenAI

# 初始化客户端
client = OpenAI(api_key="你的OpenAI_API_KEY")

def gpt4o_vision_qa(image_url: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        temperature=0.1  # 低温度，保证Agent观察结果稳定、客观
    )
    return response.choices[0].message.content

# 测试运行
if __name__ == "__main__":
    test_img = "图片公网URL"
    result = gpt4o_vision_qa(test_img, "提取图片中的关键信息，作为Agent的环境观察数据")
    print(result)

```

### 实战注意事项

- 所有API接口**不支持本地图片路径**，必须使用公网可访问URL

- Agent场景需设置低temperature（0\.1\~0\.3），保证视觉观察结果稳定、无随机发散

- 核心使用逻辑：识图摘要 → 灌入Agent提示词 → 完成视觉驱动决策

## 五、高频面试/学习思考题（巩固核心）

1. 为什么多模态模型不直接将原始像素输入LLM？

2. 图像Token过多会对Agent运行造成什么影响？

3. Adapter投影层在多模态架构中的核心作用是什么？

4. LLaVA、Qwen\-VL、GPT\-4o分别适合什么Agent开发场景？

## 六、学习总结

1\. 开源多模态模型通用架构为**视觉编码器（ViT）\+ Adapter \+ LLM**，是视觉Agent的底层核心；

2\. LLaVA用于学习原理，Qwen\-VL用于国内落地开发，GPT\-4o用于高精度通用场景；

3\. 多模态API的核心价值是为Agent提供**视觉转文本**的观察能力，是视觉智能Agent开发的第一步。

> （注：部分内容可能由 AI 生成）
