# 运行与接口说明

[返回研究说明](README.md)

以下为发布版本的配置、推理和离线训练接口。研究方法、结果及限制见 README。

## 目录结构

```text
.
├── agent.py                 世界模型推理与导航 Agent
├── agent_train.py           世界模型离线训练入口
├── config/
│   ├── world.yaml           世界生成、环境几何和任务
│   ├── world_model.yaml     模型结构、权重、设备和精度
│   ├── navigation.yaml      候选搜索、评分、安全守卫和脱困记忆
│   └── train.yaml           离线训练参数
├── model/                   编码器、状态更新和预测头
│   └── vendor/              本地 DINOv2 实现及其许可证
├── util/                    配置、权重、离线数据和损失工具
├── world/                   地图生成、局部渲染和环境接口
├── finetine/                视觉骨干、LoRA、DINO SSL/V-JEPA 辅助模块
├── baseline/HER_GRU_SAC/    HER GRU-SAC 基线，仅推理
└── checkpoints/             分类权重、来源和 SHA256 清单
```

`finetine` 沿用现有目录拼写。最终 Agent 使用 `model/` 中的冻结编码器；`finetine/` 中的其他训练模块不属于默认导航流程。

## 获取代码和权重

权重通过 Git LFS 管理。安装 Git LFS 后执行：

```bash
git lfs install
git clone https://github.com/shaohewei6/Wall-World.git
cd Wall-World
git lfs pull
```

如果已复制整个本地发布目录，并且其中包含完整权重文件，可直接使用该目录。代码运行时从本地加载权重，不调用 `torch.hub` 下载模型。

| 配置字段或权重角色 | 用途 |
| --- | --- |
| `world_model.yaml` → `weights.checkpoint` | 世界模型 |
| `world_model.yaml` → `weights.encoder_reference` | 标准初始化及紧凑 checkpoint 的冻结编码器引用 |
| 权重清单角色 `baseline` | GRU-SAC Actor 基线 |
| 权重清单角色 `encoder_pretrain` | 基线视觉骨干 |
| 权重清单角色 `encoder_adapter` | 基线冻结 LoRA 适配器 |

五份权重共 **439,100,753 字节（约439.1 MB）**。文件来源、大小及 SHA256 见 [权重清单](checkpoints/manifest.yaml)。其中来源路径用于记录研究资产的出处，运行时不需要这些原项目路径。加载器使用 PyTorch checkpoint 格式，应只加载可信文件。

## Python 环境

使用 Python 3.10 或更高版本。先安装适合本机 CPU/CUDA 环境的 PyTorch，再安装其余依赖：

```bash
python -m pip install numpy Pillow PyYAML tqdm
```

当前本地验证环境为 Python 3.10.12、PyTorch 2.14.0+cu130、NumPy 2.2.6、Pillow 12.3.0、PyYAML 6.0.3 和 tqdm 4.70.1；这是一组已验证版本，不是依赖锁定文件。未验证所有版本与硬件组合。

以下命令和示例均在本仓库根目录执行。默认配置使用 **CUDA/BF16**；CPU 运行应同时指定 `device="cpu"` 和 `precision="float32"`。`device="auto"` 只选择设备，不会自动调整精度。

## 世界模型推理

```python
import torch
from agent import Agent
from world import make_world

torch.set_num_threads(4)
agent = Agent.from_config("config", device="cpu", precision="float32")
world = make_world("config/world.yaml", model_config=agent.config)
image, origin = world.reset()

h = agent.reset(image)
prediction = agent.predict(h)
print(h.shape)                            # [1, 8, 256]
print(prediction["reachability"].shape)    # [1, 8]

# 两条6步动作序列，仅用内部状态进行 Dream 预测。
actions = torch.tensor([[0, 0, 3, 3, 1, 1], [4, 4, 5, 5, 6, 6]])
dream = agent.dream_rollout(actions)
print(dream["h"].shape)                   # [2, 6, 8, 256]

# 真实执行后提交新图像和真实位移，更新内部状态。
action = 3
next_image, position, info = world.step(action)
feedback = agent.step(action, next_image, info["actual_displacement"])
print(feedback["relative_position"])
```

`reset` 用初始观测建立修正状态 H+；`step` 先由 Dynamic 根据动作产生预测状态 H−，再由 Observation 根据新视觉和外部坐标累加器修正。外部坐标累加器使用环境反馈的真实位移，包括碰撞时的零位移。

`dream_rollout` 只接收动作序列和内部状态，不接收真实图像、真实坐标或地图；调用它不会推进真实环境，也不会替换 Agent 当前状态。预测输出包括视觉特征、坐标、碰撞及动作可达性。视觉输出是 **256×384 的特征 token**，不是解码后的 RGB 图像。

可达性由8个独立 sigmoid 给出，表示当前状态下下一步执行各动作能否成功位移，**不是总和为1的动作策略概率**。

## 导航示例

导航决策位于 `agent.py` 的 `Agent` 类中。`decide()` 返回待执行动作，调用方执行环境动作后，用 `step()` 提交反馈。

```python
from agent import Agent
from world import make_world

agent = Agent.from_config("config")  # 默认 CUDA/BF16
world = make_world("config/world.yaml", model_config=agent.config)

# 在任务生成阶段采样起点和目标，再将环境复位至起点。
_, origin = world.reset(seed=19001)
_, target = world.reset(seed=19002)
image, origin = world.reset(position=origin)
goal = (target - origin).tolist()
agent.reset(image, goal=goal)

while True:
    decision = agent.decide()
    if decision["action"] is None:
        print(decision["status"], decision["step"])
        break
    action = decision["action"]
    image, position, info = world.step(action)
    agent.step(action, image, info["actual_displacement"])
```

这是接口使用示例，不是固定评估任务或性能测试。若在 `world.yaml` 中设置了 `task.goal`，可在 `world.reset()` 后通过 `world.navigation_goal(origin)` 获取相对目标，替代上面的目标采样。

### 坐标和动作约定

- 坐标顺序为 `[x, y]`，x 向右增加，y 向下增加。
- Agent 的目标坐标相对于**回合起点**，不是相对于当前位置。默认有效范围为每轴 `[-63, 63]`。
- `world.reset()` 和 `world.step()` 返回绝对格坐标；`info["actual_displacement"]` 为整数格位移。默认一格为14像素。
- 一次离散动作成功时移动一格，碰撞时原地不动。接入其他环境时也需满足该原子运动语义。
- 输入图像支持 NumPy 数组或 PyTorch tensor，形状为 `[H,W,3]`、`[3,H,W]` 或对应批量形式；默认 H=W=224。建议使用 uint8 RGB 或 `[0,1]` 浮点 RGB，编码器内部完成归一化。

| 动作编号 | 方向 | 格位移 |
| --- | --- | --- |
| 0 | 上 | `(0, -1)` |
| 1 | 下 | `(0, 1)` |
| 2 | 左 | `(-1, 0)` |
| 3 | 右 | `(1, 0)` |
| 4 | 左上 | `(-1, -1)` |
| 5 | 右上 | `(1, -1)` |
| 6 | 左下 | `(-1, 1)` |
| 7 | 右下 | `(1, 1)` |

世界模型的低层推理接口支持批量输入；导航一次处理一个回合。重复调用 `decide()` 而不执行 `step()` 会返回同一待执行动作。`action=None` 表示成功、达到步数上限或没有安全候选，应检查返回的 `status`。

`navigation_state()` 和 `restore_navigation(snapshot)` 可保存和恢复 Agent 的内部状态、待执行队列、历史点表和随机数状态。环境状态需由调用方另行保存并同步恢复；恢复时模型权重、配置及精度必须匹配。

## 配置

YAML 内相对路径以**该配置文件所在目录**为基准；命令行显式传入的路径以当前工作目录为基准。配置使用严格字段校验，未知字段、重复字段或不兼容的模型结构会报错。

| 配置文件 | 主要参数 |
| --- | --- |
| [world.yaml](config/world.yaml) | 随机种子、地图生成或文件加载、房间/门/墙几何、起点与目标 |
| [world_model.yaml](config/world_model.yaml) | 权重路径与哈希、设备与精度、状态维度、网络深度、注意力头数及编码器参数 |
| [navigation.yaml](config/navigation.yaml) | 候选数量、预测长度、执行长度、温度、可达性混合、排斥场及登记条件、安全守卫 |
| [train.yaml](config/train.yaml) | 数据路径、批大小、序列分块、优化器、学习率调度、保存频率及恢复路径 |

使用已发布权重时，模型结构必须与 checkpoint 一致。可调整地图生成参数，但应保持观测尺寸、运动尺度与模型训练约定一致。加载已有地图时设置 `source: file` 和 `map_path`，文件应为包含二维 `wall_mask` 的 NPZ，True 表示墙体。

### 默认导航方法

每次规划生成 **128条、每条6步** 的候选：120条随机候选，以及8条覆盖所有首动作方向的候选。首动作基础分布均匀；后续动作围绕各轨迹的**首动作方向**采样，角度标准差依次为60°、45°、35°、30°和30°。基础分布与预测可达性相乘归一化，再混入10%的原基础分布。

任意一步预测碰撞即删除整条候选。安全候选按以下分数进行 softmax 采样，温度为0.02：

```text
路径分数 = 10 × 是否到达目标 + Φ(路径末端) − Φ(路径起点)
```

Φ 为目标吸引与局部排斥共同定义的势能，折扣因子为1。远离点使用半径8格、强度0.5的局部四次核，权重半衰期32步；目标位置的排斥关闭。默认不登记起点，无点数上限和登记冷却；同坐标重复登记保留最初出生时间。

安全候选最大评分不改善，或最近20步明显重复且后10步相对前10步的最佳目标距离改善不足1格时，可登记远离点。外部记忆维护最近20步坐标和远离点表。

选中路径最多执行4步，每个真实步骤都进行 Observation 更新和下一步碰撞守卫，必要时提前重新规划。默认真实执行上限为256步。准确数值与约束以配置和源码为准；修改配置后的结果不自动等同于原研究结果。

## HER GRU-SAC 基线推理

基线使用 GRU Actor，支持随机采样与 argmax，无碰撞屏蔽。HER（Hindsight Experience Replay，事后经验回放）用于训练阶段，推理阶段不执行 HER 重标注。当前目录不提供基线训练入口。

复用上面导航示例生成的 `world`、`origin` 和 `goal`，并先将环境复位：

```python
from pathlib import Path
import yaml
from baseline.HER_GRU_SAC import Agent as SACAgent

# 按公开权重清单中的角色定位文件。
manifest = yaml.safe_load(Path("checkpoints/manifest.yaml").read_text())
weights = {
    item["role"]: str(Path("checkpoints") / item["path"])
    for item in manifest["assets"]
}
image, _ = world.reset(position=origin)
baseline = SACAgent.from_checkpoint(
    weights["baseline"],
    encoder_checkpoint=weights["encoder_pretrain"],
    encoder_adapter=weights["encoder_adapter"],
    device="cuda", precision="bfloat16",
)
baseline.reset(image, goal=goal, mode="sample", seed=18810100)
while True:
    decision = baseline.decide()
    if decision["action"] is None:
        print(decision["end_reason"])
        break
    action = decision["action"]
    image, position, info = world.step(action)
    baseline.step(action, image, info["actual_displacement"],
                  collision=info["collision"])
```

设置 `mode="argmax"` 使用确定性解码。CPU 推理同样需要同时设置 `device="cpu"`、`precision="float32"`。基线返回的 `probabilities` 是互斥动作策略概率，与世界模型的独立动作可达性不同。更多输入和奖励反馈约定见 [基线说明](baseline/HER_GRU_SAC/README.md)。

## 世界模型离线训练

`agent_train.py` 从 `config/train.yaml` 读取参数，命令行显式参数覆盖 YAML。训练冻结 DINOv2 及其 LoRA，更新世界模型和预测头，使用最长6步的离线序列目标、跨分块内部状态传递及 AdamW 优化。

**训练数据不随本仓库发布。** 先提供录制轨迹，并设置 `data.root` 或传入 `--data`。推荐只读 NPY 分回合布局：

```text
data/train/
└── episode_000000/
    ├── observations.npy
    ├── actions.npy
    ├── states.npy
    └── collision_flags.npy
```

设 T 为回合动作数：

| 数组 | 形状与类型 | 语义 |
| --- | --- | --- |
| `observations` | `[T+1, 3, 224, 224]`，uint8 | 各时刻局部 RGB，包含初始帧 |
| `actions` | `[T]`，整数 | 0至7的动作编号 |
| `states` | `[T+1, 2]`，整数 | 智能体中心的绝对像素坐标 `[x,y]` |
| `collision_flags` | `[T]`，bool | 对应动作是否碰撞 |

默认像素中心与格坐标关系为 `states = patch_position × 14 + 7`，不能将格坐标直接写入 `states`。所有选中回合必须等长、T大于0且不超过4096；动作、碰撞和相邻位置必须一致。加载器也支持 `episode_*.npz`，其键名与上述四个数组相同，可放在数据根目录或其 `episodes/` 子目录。NPY 布局支持 mmap，优先使用它减少整回合解压开销。

查看参数，以及在已准备数据后运行训练：

```bash
python agent_train.py --help
python agent_train.py --train-config config/train.yaml
```

当前默认训练目标是30个优化器更新，仅用于短流程验证配置。`run.initialization: canonical` 从标准 step-zero 世界模型初始化开始；设置 `checkpoint` 则从 `world_model.yaml` 指定的发布权重初始化新的训练流。初始化权重与完整恢复训练是两种不同操作。

训练默认输出到 `results/offline_training/`，包括 `metrics.jsonl`、`status.json`、配置、数据清单、`summary.json` 以及 `latest.pth` / `previous.pth`。周期性 checkpoint 保存模型、优化器、调度器、随机数状态、内部状态和数据分块游标，并采用临时写入和原子替换。

例如，将已完成30步的同一次运行恢复至总计60步：

```bash
python agent_train.py --train-config config/train.yaml \
  --resume results/offline_training/latest.pth \
  --output results/offline_training --max-steps 60
```

`max_steps` 是包含恢复前步骤的**累计目标**。恢复要求数据、主要训练参数和源代码匹配，并使用原输出目录；新训练要求空输出目录。该恢复格式面向本发布入口产生的 checkpoint。

新训练产生的紧凑权重可由 `Agent.from_checkpoint` 加载，但仍需提供冻结 `encoder_reference`。若通过 YAML 切换新权重，需同时更新权重路径及相应 SHA256。
