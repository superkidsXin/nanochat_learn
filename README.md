# nanochat_learn
nanochat_learn

## 目标
在单卡 3060 12GB 上跑通 `speedrun_1gpu_3060.sh` 全流程，并能从0复刻 nanochat 的四条主线：模型结构、训练工程、tokenizer、推理服务。
## 先跑通再深挖
运行：
```bash
bash speedrun_1gpu_3060.sh
```
产物（默认在 `~/.cache/nanochat`）：
- `tokenizer/`：`tokenizer.pkl`、`token_bytes.pt`
- `base_checkpoints/`、`mid_checkpoints/`、`chatsft_checkpoints/`
- `report/` 与生成的 `report.md`
## 阅读顺序（按依赖倒序）
1) tokenizer：`nanochat/tokenizer.py` + `rustbpe/src/lib.rs`
2) 模型结构：`nanochat/gpt.py`
3) 推理：`nanochat/engine.py`（KV cache、流式生成、工具 token 协议）
4) 训练入口：`scripts/base_train.py` → `scripts/mid_train.py` → `scripts/chat_sft.py`
5) 数据工程：`nanochat/dataloader.py`、`nanochat/dataset.py`
6) 评测与报告：`scripts/base_loss.py`、`scripts/base_eval.py`、`scripts/chat_eval.py`、`nanochat/report.py`、`tasks/`
7) Web 服务：`scripts/chat_web.py` + `nanochat/ui.html`
## 复刻路线（每一步都能运行）
v0（最小GPT）：实现 token→(B,T) batch→loss→采样；对照 `nanochat/gpt.py`
v1（KV cache）：实现 prefill+decode；对照 `nanochat/engine.py:KVCache`
v2（checkpoint）：save/load；对照 `nanochat/checkpoint_manager.py`
v3（tokenizer）：先用现成 tiktoken，再替换为 rustbpe 训练；对照 `nanochat/tokenizer.py`、`rustbpe/src/lib.rs`
v4（SFT）：实现对话渲染+mask 训练；对照 `tokenizer.render_conversation()` 与 `scripts/chat_sft.py`
v5（Web）：FastAPI SSE 流式；对照 `scripts/chat_web.py`
v6（DDP）：torchrun 环境变量→选卡→init pg；对照 `nanochat/common.py`

