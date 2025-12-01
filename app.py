import streamlit as st
import time
import re
import torch
import plotly.graph_objects as go
from transformers import pipeline
import random
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# ==========================================
# 1. 頁面全域設定 (必須放在程式碼最上方)
# ==========================================
st.set_page_config(
    page_title="AI 文本鑑識系統",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS 樣式優化 (提升專業感)
# ==========================================
st.markdown(r'''
<style>
    /* 調整主容器頂部間距 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    /* 優化文字輸入框字體 */
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
        border-radius: 10px;
    }
    /* 讓按鈕更顯眼 */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
    }
    /* LIME 解釋的樣式 */
    .lime-explanation {
        border: 1px solid #444;
        padding: 15px;
        border-radius: 10px;
        background-color: #262730;
    }
</style>
''', unsafe_allow_html=True)

# ==========================================
# 3. 核心功能函數
# ==========================================

@st.cache_resource
def load_detectors():
    """
    載入多個 Hugging Face 模型。
    使用 @st.cache_resource 確保模型只會載入一次。
    """
    device = 0 if torch.cuda.is_available() else -1
    model_info = [
        ("ModernBERT Detector", "AICodexLab/answerdotai-ModernBERT-base-ai-detector"),
        ("RoBERTa Detector", "Hello-SimpleAI/chatgpt-detector-roberta")
    ]
    loaded_pipelines = []
    for display_name, model_id in model_info:
        try:
            pipe = pipeline("text-classification", model=model_id, device=device, return_all_scores=True)
            loaded_pipelines.append({"name": display_name, "pipe": pipe, "id": model_id})
        except Exception as e:
            print(f"⚠️ 模型 '{display_name}' 載入失敗: {e}")
    return loaded_pipelines

def clean_text(text: str) -> str:
    """清理輸入文本，移除不可見字元"""
    text = text.replace("\u200b", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def create_gauge_chart(score, title="綜合評分"):
    bar_color = "#10B981" if score > 50 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': "gray"}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "#E5E7EB",
            'steps': [{'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'}, {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.1)'}],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'family': "Arial"})
    return fig

def get_verdict(score):
    if score > 80: return "✅ 高機率為人類撰寫"
    elif score > 50: return "⚠️ 可能為混合內容 / 模稜兩可"
    else: return "🤖 高機率由 AI 生成"

# ==========================================
# 4. LIME 解釋器相關函數
# ==========================================

def get_lime_predictor(pipe, model_name):
    def predictor(texts):
        predictions = pipe(texts, truncation=True, max_length=512)
        probs = []
        for text_preds in predictions:
            prob_map = {p['label']: p['score'] for p in text_preds}
            human_prob, ai_prob = 0.0, 0.0
            if model_name == "ModernBERT Detector":
                human_prob, ai_prob = prob_map.get("LABEL_0", 0.0), prob_map.get("LABEL_1", 0.0)
            elif model_name == "RoBERTa Detector":
                human_prob = prob_map.get("Human", prob_map.get("Real", 0.0))
                ai_prob = prob_map.get("ChatGPT", prob_map.get("Fake", 0.0))
            probs.append([ai_prob, human_prob])
        return np.array(probs)
    return predictor

# ==========================================
# 5. 主程式邏輯
# ==========================================
def main():
    # --- 初始化 Session State ---
    if 'text_content' not in st.session_state: st.session_state.text_content = ""
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'lime_html' not in st.session_state: st.session_state.lime_html = None
    if 'available_indices' not in st.session_state: 
        sample_texts = { "AI": [...], "Human": [...] }
        st.session_state.all_samples = sum(sample_texts.values(), [])
        st.session_state.available_indices = list(range(len(st.session_state.all_samples)))

    # --- 側邊欄 ---
    with st.sidebar:
        st.header("🛡️ AI Sentinel"); st.caption("版本 v4.1 | LIME 解釋整合")
        st.info("**📊 判讀指南：**\n本工具使用雙模型分析來提升準確度。")
        st.markdown("### 使用模型\n- **ModernBERT**: 新一代高效能架構。\n- **RoBERTa**: 經典且穩定的偵測模型。")
        st.success("**新增功能：**\n報告底部新增 LIME 可視化解釋，標示影響判斷的關鍵詞彙。")
        st.markdown("**💡 注意：**\n結果僅供參考，不應作為絕對依據。"); st.caption("Designed for HW5")

    # --- 載入模型 ---
    active_pipelines = load_detectors()

    # --- 主介面 ---
    st.title("🕵️‍♂️ 專業級 AI 內容檢測儀")
    st.markdown("#### 透過雙模型交叉驗證與 LIME 解釋，深入了解 AI 的判斷依據")
    st.markdown("---")
    
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.subheader("📝 輸入待測文本")
        btn_cols = st.columns([1, 1])
        if btn_cols[0].button("隨機範例", key="random_sample"):
            if not st.session_state.available_indices:
                st.session_state.available_indices = list(range(len(st.session_state.all_samples)))
                st.toast("所有範例已顯示完畢，列表已重置。")
            random_index = random.choice(st.session_state.available_indices)
            st.session_state.text_content = st.session_state.all_samples[random_index]
            st.session_state.available_indices.remove(random_index)
            st.session_state.analysis_results, st.session_state.lime_html = None, None # 清除舊結果
            st.rerun()

        if btn_cols[1].button("🗑️ 清空", key="clear"):
            st.session_state.text_content, st.session_state.analysis_results, st.session_state.lime_html = "", None, None
            st.rerun()

        input_text = st.text_area("請在此貼上文章內容 (建議英文):", value=st.session_state.text_content, height=350, key="text_area_input")
        word_count = len(input_text.split())
        st.caption(f"目前字數: {word_count} words")
        analyze_btn = st.button("🔍 開始交叉分析", type="primary", use_container_width=True, disabled=(not active_pipelines))

    # --- 分析按鈕邏輯 ---
    if analyze_btn:
        st.session_state.text_content = st.session_state.text_area_input # 更新 state
        if not st.session_state.text_content.strip() or len(st.session_state.text_content.split()) < 3:
            st.session_state.analysis_results, st.session_state.lime_html = None, None
            with col2: st.warning("⚠️ 請輸入至少 3 個單字的有效文字內容！")
        else:
            with st.spinner("正在執行交叉驗證與 LIME 解釋..."):
                safe_text = clean_text(st.session_state.text_content)
                # 1. 分數計算
                results_detail = []
                for item in active_pipelines:
                    pipe, name = item['pipe'], item['name']
                    prediction_list = pipe(safe_text)[0]
                    prob_map = {p['label']: p['score'] for p in prediction_list}
                    human_prob = 0.0
                    if name == "ModernBERT Detector": human_prob = prob_map.get("LABEL_0", 0.0)
                    elif name == "RoBERTa Detector": human_prob = prob_map.get("Human", prob_map.get("Real", 0.0))
                    results_detail.append({ "name": name, "prob": human_prob })
                
                st.session_state.analysis_results = {
                    'avg_score': (sum(r['prob'] for r in results_detail) / len(results_detail)) * 100,
                    'details': results_detail
                }
                
                # 2. LIME 解釋
                model_to_explain = active_pipelines[0]
                explainer = LimeTextExplainer(class_names=["AI", "Human"])
                lime_predictor = get_lime_predictor(model_to_explain['pipe'], model_to_explain['name'])
                explanation = explainer.explain_instance(safe_text, lime_predictor, num_features=15, labels=(0, 1))
                st.session_state.lime_html = explanation.as_html(labels=(1,0))
                st.session_state.lime_model_name = model_to_explain['name']
            st.rerun()

    # --- 結果顯示區 ---
    with col2:
        st.subheader("📊 綜合分析報告")
        if not active_pipelines:
            st.error("❌ 無法載入任何 AI 模型。")
        elif st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.plotly_chart(create_gauge_chart(results['avg_score']), use_container_width=True)
            st.markdown(f"<h3 style='text-align: center; color: #FFF;'>{get_verdict(results['avg_score'])}</h3>", unsafe_allow_html=True)
            st.markdown("---")
            st.write("##### 🔬 雙模型交叉比對結果：")
            for res in results['details']:
                st.markdown(f"**{res['name']}**"); st.progress(res['prob'])
                st.caption(f"人類機率: {res['prob']:.2%} | {get_verdict(res['prob']*100)}")
            
            # 顯示 LIME 解釋
            if st.session_state.lime_html:
                st.markdown("---")
                st.subheader("💡 模型判斷依據 (LIME)")
                st.info(f"下方顯示 **{st.session_state.lime_model_name}** 的判斷依據。綠色為 **Human** 傾向，紅色為 **AI** 傾向。")
                components.html(st.session_state.lime_html, height=400, scrolling=True)
        else:
            st.info("👈 請在左側輸入文章，並點擊按鈕開始分析。")

if __name__ == "__main__":
    main()