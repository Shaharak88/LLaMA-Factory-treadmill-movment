**# LoRA Training for Treadmill Motion Detection**



**Improved Qwen2.5-VL-3B's treadmill belt motion detection from ~50% to >75% accuracy using LoRA fine-tuning.**



**## Training Command**

**```bash**

**llamafactory-cli train examples/train\_qlora/qwen25vl\_lora\_sft.yaml**

**```**



**## Configuration**



**- \*\*Config file\*\*: `examples/train\_qlora/qwen25vl\_lora\_sft.yaml`**

**- \*\*Dataset\*\*: `data/treadmill\_dataset.json` (72 videos: 36 moving, 36 stopped)**

**- \*\*Base Model\*\*: Qwen2.5-VL-3B-Instruct**

**- \*\*LoRA Rank\*\*: 8, Alpha: 16**

**- \*\*Epochs\*\*: 4**

**- \*\*Batch Size\*\*: 1 (with gradient accumulation)**

**- \*\*Quantization\*\*: 4-bit (for 8GB VRAM)**

**- \*\*Hardware\*\*: RTX 4070 Laptop**



**## Results**



**Training improved model accuracy on treadmill motion detection from baseline ~50% to ~75%.**



**## Files**



**- Training config: `examples/train\_qlora/qwen25vl\_lora\_sft.yaml`**

**- Dataset: `data/treadmill\_dataset.json`**

**- Output: `saves/qwen2vl-treadmill-lora/`**

