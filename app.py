import streamlit as st
import time
import re
import torch
import plotly.graph_objects as go
from transformers import pipeline
import random

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
    /* 結果區塊的樣式 */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #262730;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    # 檢查是否有 GPU
    device = 0 if torch.cuda.is_available() else -1
    
    # 定義模型列表：(顯示名稱, HuggingFace ID)
    model_info = [
        ("ModernBERT Detector", "AICodexLab/answerdotai-ModernBERT-base-ai-detector"),
        ("RoBERTa Detector", "Hello-SimpleAI/chatgpt-detector-roberta") 
        # 備註: Fakespot 模型有時因授權問題無法公開存取，改用 Hello-SimpleAI 這款穩定的開源模型
    ]
    
    loaded_pipelines = []
    
    for display_name, model_id in model_info:
        try:
            # 嘗試載入模型
            pipe = pipeline("text-classification", model=model_id, device=device)
            loaded_pipelines.append({"name": display_name, "pipe": pipe, "id": model_id})
        except Exception as e:
            # 如果單一模型失敗，記錄錯誤但不中斷程式
            print(f"⚠️ 模型 '{display_name}' 載入失敗: {e}")
            # 可以選擇在這裡顯示一個 toast
            # st.toast(f"模型 {display_name} 載入失敗，將使用其他模型。", icon="⚠️")
    
    return loaded_pipelines

def clean_text(text: str) -> str:
    """清理輸入文本，移除不可見字元"""
    text = text.replace("\u200b", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def create_gauge_chart(score, title="綜合評分"):
    """
    繪製專業的儀表板圖表
    Score 代表 '人類撰寫機率' (0-100)
    """
    # 顏色邏輯：人類機率高(>50)為綠色，低為紅色
    bar_color = "#10B981" if score > 50 else "#EF4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': "gray"}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},  # 淡紅
                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.1)'} # 淡綠
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    return fig

def get_verdict(score):
    if score > 80:
        return "✅ 高機率為人類撰寫"
    elif score > 50:
        return "⚠️ 可能為混合內容 / 模稜兩可"
    else:
        return "🤖 高機率由 AI 生成"

# ==========================================
# 4. 主程式邏輯
# ==========================================
def main():
    # --- 側邊欄 (Sidebar) ---
    with st.sidebar:
        st.header("🛡️ AI Sentinel")
        st.caption("版本 v3.5 | 雙模型交叉驗證")
        st.markdown("---")
        st.info(
            """
            **📊 判讀指南：**
            本工具使用 **雙模型分析** 來提升準確度。
            綜合評分是兩個模型結果的加權平均值。
            """
        )
        st.markdown("### 使用模型")
        st.markdown(
            """
            此工具整合了兩款不同架構的預訓練模型：
            - **ModernBERT**: 新一代高效能架構。
            - **RoBERTa**: 經典且穩定的 AI 偵測模型。
            
            **💡 注意：**
            偵測先進 AI (如 GPT-4o) 生成的文本極具挑戰性，
            結果僅供參考，不應作為絕對依據。
            """
        )
        st.markdown("---")
        st.caption("Designed for HW5")

    # --- 載入模型 ---
    with st.spinner("正在載入 AI 偵測模型，請稍候..."):
        active_pipelines = load_detectors()

    # --- 主標題 ---
    st.title("🕵️‍♂️ 專業級 AI 內容檢測儀")
    st.markdown("#### 透過雙模型交叉驗證，提升分析的可信度")
    st.markdown("---")
    
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""

    # 範例文字庫
    sample_texts = {
        "AI": [
            "Leveraging synergistic paradigms, our holistic framework proactively optimizes scalable, next-generation architectures to empower enterprise-level stakeholders and ensure robust, end-to-end platform integration.",
            "The subject vehicle, a 2022 sedan, was observed proceeding northbound at a velocity of 58 kilometers per hour. Weather conditions were optimal. No anomalous events were recorded during the observation period.",
            "A computer is an electronic device that manipulates information, or data. It has the ability to store, retrieve, and process data. You can use a computer to type documents, send email, play games, and browse the Web.",
            "The benefits of this system are numerous. The first benefit is efficiency. The second benefit is scalability. The third benefit is security. The fourth benefit is cost-effectiveness. The fifth benefit is user-friendliness.",
            "The ontological nature of consciousness represents a persistent enigma within neuro-scientific inquiry, where emergent properties of subjective experience defy simple reductionist explanations."
        ],
        "Human": [
            "Are you kidding me with this wifi right now?! It's been cutting out all morning and I have a huge deadline. I swear, I've tried restarting the router like, a million times. I'm about to lose my mind.",
            "My grandma's kitchen always smelled like cinnamon and fresh bread. I remember being a little kid, sitting on a stool that was way too tall for me, just watching her knead dough. I miss those simple afternoons.",
            "OMG I GOT THE TICKETS!!! I can't believe it, they sold out in like 30 seconds but I was fast enough. My hands are still shaking. This is going to be the best concert EVER. I'm already planning my outfit, haha!",
            "Idk, I just feel like pineapple on pizza isn't as bad as people make it out to be. It's like, a little bit of sweet to balance out the salty. Not my go-to order, but I won't be mad if it's there, you know?",
            "My cat has this weird habit where he only drinks water if it's from my glass. I'll have a full, fresh bowl for him, but he'll just stare at it and then try to stick his head in my cup. What a weirdo. Love him though.",
            "Okay, so for the potluck, I'll bring the mac and cheese. Can you grab a dessert? Maybe that lemon tart from the bakery on 5th street? Let me know what you think. We still need someone to bring drinks.",
            "wait wait I typed the wrong thing—hold on—ok NOW it makes sense. I think.",
            "ngl I’m so tired I just stared at my screen for like… a full minute. doing nothing. just staring.",
            "bro why did I randomly remember that one dumb thing I said in 7th grade?? who asked for this pain.",
            "okay but why did my brain suddenly remember something embarrassing from 10 years ago. for WHAT.",
            "why is my brain bringing up that cringe moment from forever ago right NOW of all times. like pls stop.",
            "not my brain dropping a random embarrassment bomb from 2014 while I’m literally doing nothing. WHY.",
            "why did my mind just throw a random “remember when you embarrassed yourself in front of everyone” flashback at me for NO reason."
        ],
    }

    # --- 雙欄佈局 ---
    col1, col2 = st.columns([1.2, 1], gap="large")

    # --- Sample Logic ---
    all_samples = sample_texts["AI"] + sample_texts["Human"]
    if 'available_indices' not in st.session_state:
        st.session_state.available_indices = list(range(len(all_samples)))

    # === 左側：輸入區 ===
    with col1:
        st.subheader("📝 輸入待測文本")
        
        st.write("快速測試範例：")
        btn_cols = st.columns([1, 1])
        if btn_cols[0].button("隨機範例 (Random Sample)", key="random_sample"):
            if not st.session_state.available_indices:
                st.session_state.available_indices = list(range(len(all_samples)))
                st.toast("所有範例已顯示完畢，列表已重置。")

            random_index = random.choice(st.session_state.available_indices)
            st.session_state.text_content = all_samples[random_index]
            st.session_state.available_indices.remove(random_index)
            st.rerun()

        if btn_cols[1].button("🗑️ 清空", key="clear"):
            st.session_state.text_content = ""
            st.rerun()

        input_text = st.text_area(
            "請在此貼上文章內容 (建議英文):",
            value=st.session_state.text_content,
            height=350,
            placeholder="請輸入至少 3 個單字以獲得最佳準確度..."
        )
        
        word_count = len(input_text.split())
        st.caption(f"目前字數: {word_count} words")

        analyze_btn = st.button("🔍 開始交叉分析", type="primary", use_container_width=True, disabled=(not active_pipelines))

    # === 右側：結果區 ===
    with col2:
        st.subheader("📊 綜合分析報告")
        
        if not active_pipelines:
            st.error("❌ 無法載入任何 AI 模型，應用程式無法運作。請檢查您的網路連線後，重新整理頁面。")
        elif analyze_btn:
            if not input_text.strip():
                st.warning("⚠️ 請先輸入文字內容！")
            elif word_count < 3: 
                st.warning("⚠️ 文字內容過短，請輸入至少 3 個單字。")
            else:
                with st.spinner(f"正在使用 {len(active_pipelines)} 個模型進行交叉驗證..."):
                    safe_text = clean_text(input_text)
                    start_time = time.time()
                    
                    scores = []
                    results_detail = []

                    # 迭代所有成功載入的模型
                    for item in active_pipelines:
                        pipe = item['pipe']
                        name = item['name']
                        
                        try:
                            # !!! 關鍵修正: 加入 truncation=True 與 max_length !!!
                            # 這是防止長文章導致程式崩潰的關鍵
                            # 只取前 512 tokens 進行預測
                            prediction = pipe(safe_text, truncation=True, max_length=512)[0]
                            
                            # 解析分數 (統一轉換為「人類機率」)
                            label = prediction['label']
                            score = prediction['score']
                            
                            human_prob = 0.0
                            # 不同模型的標籤定義可能不同，這裡做通用處理
                            
                            if name == "ModernBERT Detector":
                                # 假設: LABEL_0=Human, LABEL_1=AI
                                if label == 'LABEL_0': 
                                    human_prob = score
                                else:
                                    human_prob = 1 - score
                                    
                            elif name == "RoBERTa Detector":
                                # Hello-SimpleAI/chatgpt-detector-roberta
                                # Human, ChatGPT
                                if label in ['Human', 'Real', 'LABEL_1']: # 'LABEL_1' for some RoBERTa variants
                                    human_prob = score
                                else: # ChatGPT, Fake, LABEL_0
                                    human_prob = 1 - score
                            
                            scores.append(human_prob)
                            results_detail.append({
                                "name": name,
                                "prob": human_prob,
                                "raw": prediction
                            })
                            
                        except Exception as e:
                            st.error(f"模型 {name} 分析時發生錯誤: {e}")

                    end_time = time.time()

                    if scores:
                        # 計算平均分數
                        avg_human_prob = sum(scores) / len(scores)
                        avg_human_score_percent = avg_human_prob * 100

                        # 1. 顯示綜合儀表板
                        st.plotly_chart(create_gauge_chart(avg_human_score_percent, title="綜合評分 (平均)"), use_container_width=True)
                        st.markdown(f"<h3 style='text-align: center; color: #FFF;'>{get_verdict(avg_human_score_percent)}</h3>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # 2. 顯示各模型細節
                        st.write("##### 🔬 雙模型交叉比對結果：")
                        
                        for i, res in enumerate(results_detail):
                            st.markdown(f"#### {i+1}. {res['name']}")
                            
                            # Explain the logic
                            raw_label = res['raw']['label']
                            raw_score = res['raw']['score']
                            
                            st.write("**模型判讀邏輯:**")
                            explanation = ""
                            if "ModernBERT" in res['name']:
                                if raw_label == 'LABEL_0':
                                    explanation = f"模型回傳標籤 '{raw_label}' 代表 **人類**，其信心分數為 **{raw_score:.2%}**。因此，我們將此分數直接視為人類機率。"
                                else: # LABEL_1
                                    explanation = f"模型回傳標籤 '{raw_label}' 代表 **AI**，其信心分數為 **{raw_score:.2%}**。因此，人類機率為 100% - {raw_score:.2%} = **{1-raw_score:.2%}**。"
                            elif "RoBERTa" in res['name']:
                                if raw_label in ['Human', 'Real', 'LABEL_1']:
                                    explanation = f"模型回傳標籤 '{raw_label}' 代表 **人類**，其信心分數為 **{raw_score:.2%}**。因此，我們將此分數直接視為人類機率。"
                                else: # ChatGPT, Fake, LABEL_0
                                    explanation = f"模型回傳標籤 '{raw_label}' 代表 **AI**，其信心分數為 **{raw_score:.2%}**。因此，人類機率為 100% - {raw_score:.2%} = **{1-raw_score:.2%}**。"
                            
                            st.info(explanation, icon="🧠")

                            st.write(f"**最終推斷的人類機率:**")
                            st.progress(res['prob'])
                            st.caption(f"計算出的機率為 **{res['prob']:.2%}**，結論為: **{get_verdict(res['prob']*100)}**")
                            
                            if i < len(results_detail) - 1:
                                st.markdown("---")

                        # 3. 技術細節
                        st.markdown("---")
                        with st.expander("查看綜合技術數據 (JSON)"):
                            st.json({
                                "綜合人類機率": f"{avg_human_prob:.4f}",
                                "推論時間": f"{end_time - start_time:.3f} 秒",
                                "各模型詳細輸出": results_detail
                            })
                    else:
                        st.error("分析過程發生錯誤，無法產生分數。")

        else:
            st.info("👈 請在左側輸入文章，並點擊按鈕開始分析。")
            st.markdown(
                """
                <div style="background-color:#262730; padding:20px; border-radius:10px; border: 1px solid #444;">
                <h4 style="margin-top:0; color: white;">💡 為何使用雙模型？</h4>
                <p style="color: #ccc;">單一模型可能存在偏見或盲點。透過交叉比對兩個來自不同訓練來源的模型 (Ensemble Learning)，我們可以獲得更平衡、更可靠的判斷，有效降低誤判率。</p>
                </div>
                """, unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()