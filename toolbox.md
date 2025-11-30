import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
import time

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI vs Human 文本偵測器 Pro",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定義 CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextArea textarea { font-size: 16px; line-height: 1.6; border-radius: 10px; }
    .highlight-ai { background-color: #ff4b4b4d; border-radius: 4px; padding: 2px 4px; border-bottom: 2px solid #ff4b4b; }
    .metric-card { background-color: #262730; padding: 20px; border-radius: 10px; text-align: center; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- 模型載入管理 (Hugging Face) ---
@st.cache_resource
def load_hf_model():
    """
    嘗試載入 Hugging Face 模型。
    如果使用者沒有安裝 transformers/torch，或是下載失敗，回傳 None。
    """
    try:
        from transformers import pipeline
        # 使用一個輕量且效果不錯的公開模型
        # Hello-SimpleAI/chatgpt-detector-roberta 是基於 RoBERTa 微調的模型
        detector = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta", top_k=None)
        return detector
    except ImportError:
        return "MISSING_LIB"
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- 分析邏輯 ---

def analyze_text(text, use_hf_model=True):
    if not text:
        return None

    # 基礎統計特徵 (無論是否用 AI 模型都需要)
    sentences = re.split(r'[.!?。！？]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
    words = re.findall(r'\w+', text)
    sent_lengths = [len(s) for s in sentences]
    
    if len(sentences) == 0 or len(words) == 0:
        return None

    std_sent_len = np.std(sent_lengths) if sent_lengths else 0
    
    result = {
        "sentences": sentences,
        "sent_lengths": sent_lengths,
        "std_dev": std_sent_len,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "mode": "Simulation" # 預設模式
    }

    # --- 分支 1: 使用 Hugging Face 模型 (真實 AI 偵測) ---
    if use_hf_model:
        hf_detector = load_hf_model()
        
        # 檢查是否成功載入
        if hf_detector == "MISSING_LIB":
            st.toast("⚠️ 未安裝 transformers，切換回模擬模式", icon="🔧")
        elif isinstance(hf_detector, str) and hf_detector.startswith("ERROR"):
            st.toast(f"⚠️ 模型載入失敗，切換回模擬模式", icon="⚠️")
        elif hf_detector:
            # 成功載入真模型
            result["mode"] = "Hugging Face (RoBERTa)"
            
            # 因為模型有輸入長度限制 (通常 512 tokens)，我們取前 512 字元做快速預測
            # 生產環境應該要做切塊 (chunking) 再平均，這裡做簡化處理
            truncated_text = text[:1000] 
            predictions = hf_detector(truncated_text)[0]
            
            # 解析預測結果
            # 模型輸出範例: [{'label': 'ChatGPT', 'score': 0.98}, {'label': 'Human', 'score': 0.02}]
            ai_score = 0.0
            for pred in predictions:
                if pred['label'] == 'ChatGPT' or pred['label'] == 'Fake':
                    ai_score = pred['score'] * 100
                elif pred['label'] == 'Human' or pred['label'] == 'Real':
                    # 如果是 Human 分數，AI 分數就是 100 - Human
                    pass 
            
            # 如果模型主要標籤就是 Human，我們需要轉換分數邏輯
            top_label = predictions[0]['label']
            top_score = predictions[0]['score']
            
            if top_label in ['Human', 'Real']:
                ai_score = (1 - top_score) * 100
            elif top_label in ['ChatGPT', 'Fake']:
                ai_score = top_score * 100
                
            result["ai_probability"] = ai_score
            return result

    # --- 分支 2: 統計模擬模式 (當沒有安裝 TF 時的備案) ---
    # 計算詞彙豐富度
    unique_words = len(set(words))
    ttr = unique_words / len(words) if len(words) > 0 else 0
    
    ai_score = 0
    regularity_score = max(0, 100 - (std_sent_len * 2)) 
    ai_score += regularity_score * 0.4
    
    common_connectors = ['the', 'and', 'is', 'of', 'to', '的', '是', '在', '有', '和']
    connector_count = sum(1 for w in words if w.lower() in common_connectors)
    connector_density = connector_count / len(words)
    
    if connector_density > 0.35: 
        ai_score += 20
        
    final_score = min(98, max(2, ai_score + 30))
    import random
    noise = random.uniform(-5, 5)
    final_score = min(100, max(0, final_score + noise))
    
    result["ai_probability"] = final_score
    return result

# --- UI 元件 ---

def draw_gauge_chart(score):
    color = "green" if score < 40 else "orange" if score < 70 else "red"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI 生成可能性 (%)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': score}
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def draw_complexity_chart(sent_lengths):
    df = pd.DataFrame({'句子序號': range(1, len(sent_lengths) + 1), '長度 (字數)': sent_lengths})
    fig = px.bar(df, x='句子序號', y='長度 (字數)', title="句子結構爆發度 (Burstiness)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), showlegend=False)
    fig.update_traces(marker_color='#00d4ff')
    return fig

# --- 主程式 ---

def main():
    # 預先檢查環境
    hf_status = load_hf_model()
    model_status_text = "🔴 統計模擬模式"
    model_status_color = "off"
    
    if hf_status and hf_status != "MISSING_LIB" and not isinstance(hf_status, str):
        model_status_text = "🟢 Hugging Face (RoBERTa)"
        model_status_color = "on"
    
    with st.sidebar:
        st.header("⚙️ 系統核心設定")
        
        # 顯示當前使用的引擎
        st.markdown(f"**目前引擎:**")
        if model_status_color == "on":
            st.success(model_status_text)
        else:
            st.warning(model_status_text)
            if hf_status == "MISSING_LIB":
                st.caption("💡 提示: 安裝 `torch transformers` 可啟用 AI 模式")

        use_hf = st.toggle("啟用 Hugging Face 模型", value=(model_status_color == "on"), disabled=(model_status_color == "off"))
        
        st.markdown("---")
        st.markdown("### 關於原理")
        st.info("Hugging Face 模式使用 `Hello-SimpleAI/chatgpt-detector-roberta` 模型進行深度語義分析。")

    st.title("🕵️‍♂️ AI Content Detector Pro")
    st.markdown("貼上文章，系統將分析其是否由 ChatGPT、Claude 或 Gemini 等 AI 生成。")

    text_input = st.text_area("在此輸入文章:", height=200, placeholder="請貼上內容...")

    col1, col2 = st.columns([1, 1])
    analyze_btn = col1.button("🔍 開始分析", type="primary", use_container_width=True)
    
    if analyze_btn and text_input:
        if len(text_input) < 10:
            st.warning("⚠️ 文本過短")
        else:
            with st.spinner(f"正在使用 {model_status_text} 分析中..."):
                # 分析
                result = analyze_text(text_input, use_hf_model=use_hf)
                
                if result:
                    st.divider()
                    
                    # 結果顯示區
                    st.caption(f"分析模式: {result.get('mode', 'Unknown')}")
                    
                    g_col1, g_col2 = st.columns([1, 2])
                    with g_col1:
                        st.plotly_chart(draw_gauge_chart(result['ai_probability']), use_container_width=True)
                    
                    with g_col2:
                        st.subheader("📊 分析指標")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("總字數", result['word_count'])
                        m2.metric("句子數量", result['sentence_count'])
                        m3.metric("結構變異數", f"{result['std_dev']:.2f}")
                        
                        score = result['ai_probability']
                        if score > 80:
                            st.error(f"**高度疑似 AI 生成** ({score:.1f}%)")
                        elif score > 50:
                            st.warning(f"**疑似混合內容** ({score:.1f}%)")
                        else:
                            st.success(f"**極可能是人類撰寫** ({score:.1f}%)")

                    st.plotly_chart(draw_complexity_chart(result['sent_lengths']), use_container_width=True)
                    
                    # 只有在分數高時才顯示高亮建議
                    if score > 50:
                        st.subheader("🔍 句子標記 (高風險區段)")
                        highlighted_text = ""
                        for sentence in result['sentences']:
                            # 如果是真 AI 模型，我們可以假設整段都被判定，這裡僅做視覺化模擬
                            # 若要精確到句子，需要將每個句子單獨丟進模型 (會很慢)
                            # 這裡採用混合邏輯：如果整篇是 AI，則標記結構最完美的句子
                            is_suspicious = abs(len(sentence) - np.mean(result['sent_lengths'])) < (result['std_dev'] * 0.8)
                            
                            if is_suspicious:
                                highlighted_text += f'<span class="highlight-ai">{sentence}</span> '
                            else:
                                highlighted_text += f'<span>{sentence}</span> '
                        
                        st.markdown(f'<div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; line-height: 2.0;">{highlighted_text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()