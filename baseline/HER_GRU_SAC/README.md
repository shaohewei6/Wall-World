# HER GRU-SAC：仅推理

EXP141的HER训练、独立GRU Actor、无碰撞屏蔽基线。保留采样和argmax两种解码。HER用于原训练阶段，推理不执行HER重标注。

本目录只有`agent.py`、`model.py`、包入口和本说明。复用发布仓库的`model/encoder.py`、`model/lora.py`、`model/vendor/`及文件哈希工具；无需原项目的pure_SAC、wallworld、finetine或experiments模块。不会构造Critic、优化器或回放池，也不依赖世界模型权重。依赖与发布仓库相同。

在发布仓库根目录运行：

```python
from baseline.HER_GRU_SAC import Agent

agent = Agent.from_checkpoint(
    "checkpoints/baseline/HER_GRU_SAC/exp141_latest.pth",
    encoder_checkpoint="checkpoints/encoder/dinov2_vits14_pretrain.pth",
    encoder_adapter="checkpoints/encoder/lora_epoch_019.pth",
    device="cuda", precision="bfloat16",
)
# image：224×224局部RGB；goal：目标相对起点的整数格坐标。
agent.reset(image, goal=[-36, -30], mode="sample", seed=18810100)
decision = agent.decide()
if decision["action"] is not None:
    action = decision["action"]
    next_image, _, info = world.step(action)
    result = agent.step(action, next_image, info["actual_displacement"],
                        collision=info["collision"])
```

包导入也支持`from Wall_World.baseline.HER_GRU_SAC import Agent`。环境由调用方提供，可复用发布目录的`world/`；Agent不读取地图或绝对坐标。`actual_displacement`按格计，动作编号与原环境一致：上、下、左、右、左上、右上、左下、右下。碰撞时位移必须为0，否则必须原子移动一格。

`mode="argmax"`使用确定性解码。每个真实步骤调用decide，再用step提交反馈；同一状态重复decide返回同一待执行动作，不重复更新GRU或采样。返回的probabilities为8个动作的概率，logits为Actor原始输出；这不是世界模型的动作可达性。达到目标或达到checkpoint中的回合上限后，decide返回action=None。reset开始新回合并清空GRU和历史。

19维辅助输入保留原有归一化：相对起点位置、相对起点目标、当前目标偏移分别除63，上一格位移、碰撞标记、累计碰撞数除32、剩余步数比例及全1动作掩码。上一步收到的奖励也输入GRU，因此推理仍需维护访问点与有向边次数，以按原奖励公式计算反馈。结构、奖励和归一化参数从checkpoint读取，不复制训练配置。

与EXP189一致的精度是CUDA/BF16编码→FP16特征量化→FP32 Actor。CPU/FP32可用于smoke，但精度变化可能改变概率或动作。三个路径均显式指定，相对路径以调用时工作目录为基准；可用checkpoint_sha256、encoder_sha256、adapter_sha256分别校验。只加载可信PyTorch checkpoint。所需权重已原样复制到发布目录checkpoints，运行时不下载权重；完整清单和原始来源见checkpoints/manifest.yaml。

| 原资产 | SHA256 |
|---|---|
| results/EXP-GRU-SAC-HER-NO-MASK-141/long/latest.pth | c9bf5a38fc2c8c1e9964958d478bab0fb373972d66dfdc65be000d944cac9ec5 |
| checkpoint/dinov2_vits14_pretrain/dinov2_vits14_pretrain.pth | b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 |
| checkpoint/finetine/dinov2-finetine/lora_epoch_019.pth | 6888fcae8a8203bd25e81f1b2c55cf1eeac37cc728b04c6110bb3393e90cfd3a |

本目录仅包含推理代码和说明，支持实际保留的EXP141结构与输入语义。权重集中在checkpoints目录。
