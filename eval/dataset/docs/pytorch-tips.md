# PyTorch 技巧

- `model.eval()` + `torch.no_grad()` 做推理，省显存
- 学习率调度：CosineAnnealingLR 比 StepLR 平滑
- 混合精度：`torch.autocast` + `GradScaler`，训练提速明显
- 梯度裁剪 `clip_grad_norm_` 防梯度爆炸，RNN 场景必加
- `pin_memory=True` + `num_workers>0` 加速 DataLoader
