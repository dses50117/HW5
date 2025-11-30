# Project Plan: AI Text Detection Application (CRISP-DM Iteration)

This document outlines the development of the AI Text Detection application, adhering to the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework. It details the journey from initial concept to the final, refined application, including challenges and iterative improvements.

---

### Phase 1: Business Understanding

**Objective:** To create an interactive web application, leveraging Streamlit, capable of estimating whether a given English text is human-written or AI-generated. The application should provide a user-friendly interface and transparent analysis.

**Initial Goal:** Develop a functional prototype demonstrating AI text detection.

**Success Criteria:** A running Streamlit application that accepts text input, provides analysis, includes sample texts, and manages user interaction effectively. The project should be well-documented and reflect best practices.

---

### Phase 2: Data Understanding (Sample Text Curation)

This phase focused on understanding the "data" in the context of demonstration – the sample texts themselves.

**Process:**
*   Initially, a small, hardcoded list of basic AI and human sample texts was included to demonstrate the "Random Sample" functionality.
*   **Iteration based on User Feedback:** The user consistently requested "more correct," "easier to distinguish," and "extreme" samples to improve the clarity of the demo.
*   **Curated Samples:** The `sample_texts` dictionary in `app.py` was iteratively updated to include a diverse set of examples for both AI and human writing styles:
    *   **AI Samples:** Crafted to be overtly robotic, formal, repetitive, buzzword-heavy, or abstract/complex, intentionally mimicking common AI traits to make detection easier. (Currently 5 samples).
    *   **Human Samples:** Crafted to be informal, emotional, personal, conversational, include slang, or exhibit stream-of-consciousness patterns, making them clearly distinct from AI. (Currently 13 samples).
*   The final set of samples aims to clearly differentiate between AI and human-generated content for demonstration purposes.

---

### Phase 3: Data Preparation (Text Cleaning)

This phase ensures the input text is suitable for model processing.

**Process:**
*   A `clean_text` utility function was implemented to preprocess user input.
*   **Functionality:** This function removes invisible characters (`\u200b`), replaces newlines with spaces, and consolidates multiple spaces into single spaces (`re.sub(r"\s+", " ", text)`).
*   **Purpose:** Ensures consistent formatting and prevents potential issues with model tokenization or analysis due to extraneous characters.

---

### Phase 4: Modeling (Dual-Model Ensemble)

This phase involved selecting, loading, and integrating the AI detection models.

**Initial Approach:**
*   The first version used a single `fakespot-ai/roberta-base-ai-text-detection-v1` model via the Hugging Face `pipeline` API.

**Challenges & Iterations:**
*   **Accuracy Concerns:** User feedback indicated initial models lacked desired accuracy.
*   **Model Instability:** An attempt to integrate a highly-rated but potentially unstable model (`desklib/ai-text-detector-v1.01`) resulted in persistent "corrupted state dictionary" errors.
*   **Pivot to Ensemble:** To enhance robustness and accuracy, the strategy shifted to a **dual-model (ensemble) approach**, leveraging the strengths of multiple models.

**Final Model Selection:**
*   The application now loads two distinct AI text detection models:
    1.  **`AICodexLab/answerdotai-ModernBERT-base-ai-detector`**: A modern BERT-based model specifically trained on texts from various contemporary AI models (e.g., ChatGPT, Claude), intended for better detection of recent AI outputs.
    2.  **`Hello-SimpleAI/chatgpt-detector-roberta`**: A robust RoBERTa-based model, a stable and widely used detector. (Replaced an earlier `fakespot-ai` model due to better public accessibility/stability).
*   **Model Loading:** Models are loaded using `@st.cache_resource` for efficient caching, preventing re-download/re-load on subsequent runs. A loading spinner (`st.spinner`) is displayed during the initial load to improve user experience.
*   **Device Handling:** Models are automatically loaded to GPU (`cuda`) if available, otherwise to CPU, optimizing performance.

---

### Phase 5: Evaluation (Performance & Limitations)

This phase focused on evaluating the overall application performance and understanding the models' capabilities and limitations.

**User Feedback & Iteration:**
*   Consistent user feedback highlighted the inherent difficulty of accurate AI detection, especially for nuanced human texts or sophisticated AI outputs.
*   Specific instances of misclassification (e.g., formal human text flagged as AI, complex AI text flagged as human) were reported.

**Key Learning & Explanation:**
*   It was determined that achieving 100% accuracy with open-source AI detectors is not feasible due to the rapid evolution of AI writing models (the "AI arms race").
*   Open-source models often lag behind the capabilities of the latest proprietary AI generators.
*   The tool's purpose was refined to be a **probabilistic indicator** rather than a definitive truth-teller, emphasizing that results are estimates.
*   **UI Clarifications:** Extensive disclaimers and explanations were added to the sidebar and main interface to manage user expectations regarding model accuracy and the challenging nature of detecting advanced AI (e.g., GPT-4o, Gemini).

**Ensemble Advantage:** The dual-model ensemble improves the robustness of the average prediction by cross-referencing two distinct detection methodologies, offering a more balanced assessment than a single model could.

---

### Phase 6: Deployment (Application Features & UI/UX)

This phase encompasses all aspects of making the application functional and user-friendly.

**Core Application Features:**
*   **Streamlit Framework:** The entire application is built using Streamlit, providing a clean and interactive web interface.
*   **Dual-Model Analysis:** Text input is processed by two distinct AI detection models, with results averaged for a comprehensive score.
*   **Loading Spinner:** A `st.spinner` is implemented during initial model loading for better user feedback.
*   **Input Validation:** Minimum word count (3 words) is enforced for analysis, with appropriate warnings.
*   **Error Handling:** Graceful handling for model loading failures, and disabling the analyze button if models aren't loaded.

**User Interface & Experience (UI/UX):**
*   **Clean Layout:** Uses `st.set_page_config` for wide layout and custom CSS (`st.markdown`) for a professional aesthetic (e.g., optimized fonts, button styles, result cards).
*   **Gauge Chart Visualization:** `plotly.graph_objects` is used to create an intuitive gauge chart visualizing the "Human Probability" score.
*   **Verdict & Details:** Clear text verdicts (Human, Mixed, AI) are displayed, along with a detailed breakdown of each model's individual score and an expandable section for raw technical data.
*   **Sample Text Management:**
    *   A single "隨機範例 (Random Sample)" button is implemented.
    *   This button provides **non-repeating random sampling** from a combined pool of AI and human examples until all have been shown, after which the list resets. (`st.toast` is used to notify the user of a reset).
    *   A "🗑️ 清空 (Clear)" button resets the text input.
*   **Sidebar Information:** Provides clear "判讀指南 (Interpretation Guide)" and detailed information about the models used, along with disclaimers about limitations.

**Execution:**
*   The application is designed for local execution via `streamlit run app.py`.
*   Session state (`st.session_state`) is extensively used to maintain interaction and data persistence across reruns.
