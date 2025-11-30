0. 建一個專案資料夾

假設專案叫：

mkdir ai-content-detector
cd ai-content-detector
mkdir data model

1. 安裝必要套件

在專案資料夾裡執行：

pip install transformers datasets torch scikit-learn evaluate streamlit


之後 requirements.txt 會寫給你。

2. 從 Hugging Face 下載 HC3 資料集並整理成 CSV

我們用 Hello-SimpleAI/HC3 的英文版本，這個資料集裡有：

question

human_answers（人類回答列表）

chatgpt_answers（ChatGPT 回答列表）
Hugging Face

新建 prepare_data.py：

# prepare_data.py
from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# 1. 下載 HC3 英文資料集（subset: all）
ds = load_dataset("Hello-SimpleAI/HC3", "all")  # train split by default

train_split = ds["train"]

rows = []

for item in train_split:
    question = item["question"]
    human_answers = item["human_answers"]
    chatgpt_answers = item["chatgpt_answers"]

    # 取每個問題的第一個人類回答 & 第一個 ChatGPT 回答
    if human_answers:
        rows.append({
            "text": human_answers[0],
            "label": "human"
        })
    if chatgpt_answers:
        rows.append({
            "text": chatgpt_answers[0],
            "label": "ai"
        })

df = pd.DataFrame(rows)
print(df["label"].value_counts())
print("Total samples:", len(df))

# 存成 CSV
out_path = "data/hc3_ai_human.csv"
df.to_csv(out_path, index=False)
print("✅ Saved:", out_path)


執行：

python prepare_data.py


完成後你會得到：data/hc3_ai_human.csv，內含兩欄：text, label。

3. 在本地訓練 Hugging Face 模型（DistilBERT）

新建 train.py：

# train.py
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/hc3_ai_human.csv"
OUTPUT_DIR = "model"

# 1. 讀 CSV
df = pd.read_csv(DATA_PATH)

label2id = {"human": 0, "ai": 1}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# 2. Train / Test Split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

train_ds = Dataset.from_pandas(train_df[["text", "label_id"]])
test_ds = Dataset.from_pandas(test_df[["text", "label_id"]])

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

# HF Trainer 格式
train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

train_ds = train_ds.rename_column("label_id", "labels")
test_ds = test_ds.rename_column("label_id", "labels")

train_ds.set_format("torch")
test_ds.set_format("torch")

# 4. 建立模型
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# 5. 評估指標
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    result = {}
    result.update(metric_acc.compute(predictions=preds, references=labels))
    result.update(metric_f1.compute(predictions=preds, references=labels))
    return result

# 6. TrainingArguments
args = TrainingArguments(
    output_dir="model_checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=2,  # 可以先跑 2 epoch 試試
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. 訓練
trainer.train()

# 8. 儲存模型到 ./model（之後 app 要讀這個）
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model saved to:", OUTPUT_DIR)


執行：

python train.py


成功後，model/ 資料夾會有：

config.json

pytorch_model.bin

tokenizer.json

tokenizer_config.json

special_tokens_map.json

...

這就是你之後在 app 裡載入的「本地訓練好的 AI Detector」。

4. 建 Streamlit App，使用你自己訓練的本地模型

新建 app.py：

# app.py
import re
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "model"  # 就是 train.py 存的目錄

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def clean_text(text: str) -> str:
    text = text.replace("\u200b", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict(text: str):
    tokenizer, model, device = load_model()
    text = clean_text(text)

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = np.exp(logits) / np.exp(logits).sum()

    # label2id = {"human": 0, "ai": 1}
    human_prob = float(probs[0])
    ai_prob = float(probs[1])
    return human_prob, ai_prob

# ================== Streamlit UI =========================
st.set_page_config(page_title="AI Content Detector", page_icon="🤖", layout="wide")

st.title("🤖 AI Content Detector")
st.write("Detect whether your text is more likely human-written or AI-generated.")
st.caption("Model: DistilBERT fine-tuned on HC3 (Human vs ChatGPT).")

text = st.text_area("Paste your text here", height=220)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_btn = st.button("Analyze", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if analyze_btn:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running local detector model..."):
            human_prob, ai_prob = predict(text)

        human_pct = human_prob * 100
        ai_pct = ai_prob * 100

        st.subheader("Result")
        st.metric("AI Probability", f"{ai_pct:.2f}%")
        st.progress(ai_prob)

        if ai_prob >= 0.7:
            st.error("This text is likely AI-generated.")
        elif ai_prob >= 0.4:
            st.warning("Mixed characteristics of AI and human writing.")
        else:
            st.success("This text is more likely human-written.")

        with st.expander("Details"):
            st.write(f"Human: {human_pct:.2f}%")
            st.write(f"AI: {ai_pct:.2f}%")


本地測試：

streamlit run app.py


確認在瀏覽器可以正常輸入文字、顯示機率。

5. 建立 requirements.txt

在專案根目錄新增 requirements.txt：

streamlit>=1.30.0
transformers>=4.37.0
datasets>=2.16.0
torch>=2.1.0
scikit-learn>=1.3.0
evaluate>=0.4.0
pandas>=2.0.0

6. 推到 GitHub
git init
git add .
git commit -m "AI content detector with local HF model"
git branch -M main
git remote add origin https://github.com/你的帳號/ai-content-detector.git
git push -u origin main

7. 部署到 Streamlit Cloud

到 Streamlit Community Cloud
 登入

點「New app」

選你的 GitHub repo：你的帳號/ai-content-detector

Branch：main

Main file path：app.py

點「Deploy」

Streamlit 會自動：

安裝 requirements.txt

使用 repo 裡的 model/ 目錄

跑 app.py