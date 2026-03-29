# HW1_314707044 - PathoQA LoRA Finetuning

本專案使用 `meta-llama/Llama-3.2-1B-Instruct` 搭配 LoRA 進行病理選擇題（A/B/C/D）微調，並產生 benchmark 預測檔上傳 Kaggle。

## 1. 專案結構

```text
HW1_314707044/
├── main.py
├── requirements.txt
├── dataset/
│   ├── dataset.csv
│   ├── benchmark.csv
│   └── splits/
│       ├── train.csv
│       └── val.csv
├── outputs/
│   ├── metrics.csv
│   ├── benchmark_predictions.csv
│   └── loss_graph/
└── saved_models/
    └── llama32_1b_lora/
```

## 2. 環境需求

- Python 3.10+
- CUDA GPU（建議）
- 主要套件：`torch`、`transformers`、`trl`、`peft`、`datasets`、`scikit-learn`、`wandb`

安裝：

```bash
pip install -r requirements.txt
```

## 3. 資料格式

### 訓練資料 `dataset/dataset.csv`
必備欄位：

- `question_id`
- `question`
- `opa`
- `opb`
- `opc`
- `opd`
- `ans`（0/1/2/3 分別對應 A/B/C/D）

### 測試資料 `dataset/benchmark.csv`
必備欄位：

- `question_id`
- `question`
- `opa`
- `opb`
- `opc`
- `opd`

## 4. 程式流程（`main.py`）

1. 讀取資料與檢查欄位完整性。
2. 依 `val_size` 分層切分 train/val，並輸出到 `dataset/splits/`。
3. 建立 chat prompt（system + user）與 completion（`Final Answer: X`）。
4. 使用 LoRA 微調模型，搭配 early stopping 與 epoch option shuffle。
5. 以 deterministic generation 進行 train/val 準確率評估。
6. 對 benchmark 產生預測並輸出 `outputs/benchmark_predictions.csv`。

## 5. 執行方式

### 完整流程（訓練 + 輸出）

```bash
python main.py --mode all
```

### 只訓練

```bash
python main.py --mode train
```

### 只輸出（需先有已訓練 adapter）

```bash
python main.py --mode output
```

## 6. 主要參數（預設值）

- `--model-name meta-llama/Llama-3.2-1B-Instruct`
- `--epochs 8`
- `--learning-rate 1e-4`
- `--batch-size 8`
- `--grad-accum 4`
- `--max-seq-length 512`
- `--val-size 0.1`
- `--lora-r 32`
- `--lora-alpha 64`
- `--max-new-tokens 16`

## 7. 輸出檔說明

- `outputs/metrics.csv`：紀錄 train/val accuracy 與資料筆數。
- `outputs/benchmark_predictions.csv`：Kaggle 上傳檔，格式為：
  - `question_id`
  - `pred`（0/1/2/3）

## 8. 實驗結果

- Local train accuracy：`0.8909`
- Local validation accuracy：`0.7256`
- **Kaggle accuracy：`0.7322`**

> 註：Kaggle 分數與本地驗證分數可能因資料分布差異而有些微落差。
