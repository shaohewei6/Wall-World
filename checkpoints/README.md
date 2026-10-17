# 推理权重

本目录包含当前世界模型与HER GRU-SAC基线所需的5份资产。文件均从原项目逐字节复制，未转换、裁剪或重新保存；原资产保持不变。总大小439,100,753字节（约439.1 MB / 418.8 MiB）。机器可读的来源、用途、大小及SHA256见`manifest.yaml`。

```text
checkpoints/
  world_model/
    exp174_e6.pth                  最终世界模型，离线E6、step6576
    exp130_initial_fresh.pth       紧凑世界模型所需的冻结编码器引用
  baseline/HER_GRU_SAC/
    exp141_latest.pth              HER GRU-SAC基线
  encoder/
    dinov2_vits14_pretrain.pth      基线视觉骨干预训练权重
    lora_epoch_019.pth             基线冻结LoRA适配器
  manifest.yaml
```

世界模型通过`config/world_model.yaml`引用前两份资产，路径相对于YAML解析。基线的三个路径见`baseline/HER_GRU_SAC/README.md`，加载时显式指定。保留原checkpoint格式意味着部分文件也包含原训练状态，但推理加载器不会构造优化器或回放池；本次没有额外加入训练权重。

复制整个发布目录即可使用这些资产，不依赖原项目目录或符号链接。加载器支持SHA256校验，校验值见manifest.yaml。

权重二进制仍受根目录.gitignore的`*.pth`规则排除。本轮仅做本地整理，未提交或上传，公开分发方式尚未决定。
