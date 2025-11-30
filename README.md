# AI Content Detector Prototype

A Streamlit web application that demonstrates the user interface and functionality of an AI-generated text detector.

---

## About This Project

This application serves as a functional prototype for an AI content detector. It allows users to input text and receive a *simulated* analysis, providing a score indicating the likelihood of the text being human-written. The primary purpose of this project was to design and build a user-friendly interface and to structure the project according to data science best practices.

### Key Features
*   **Interactive Text Area:** Paste your own text for analysis.
*   **Sample Text Generator:** Instantly load a sample text with the "Use Sample" button.
*   **Simulated Analysis:** Get a visual "Human Content" score. **Note:** This score is randomly generated and does not represent a real analysis.

---

## Project Status

**Status:** 🟢 **Functional Prototype**

This application is a demonstration and does **not** contain a real AI detection model. The analysis feature is simulated.

---

## Development Process

This project was built following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology. This structured approach ensures that the project is well-documented and follows a logical progression from business understanding to deployment.

For a detailed, step-by-step breakdown of how this app was built, please see the [**project_plan.md**](project_plan.md) file.

---

## Getting Started

### Prerequisites
*   Python 3.7+
*   Git

### Installation & Running
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dses50117/HW5.git
    cd HW5
    ```

2.  **Install dependencies:**
    This project requires `streamlit`.
    ```bash
    pip install streamlit
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

---

## Future Work

The next logical step for this project is to replace the simulated analysis with a real AI detection model. This would involve:
1.  **Model Selection:** Choosing a pre-trained text classification model from a source like the [Hugging Face Hub](https://huggingface.co/models).
2.  **Integration:** Loading the model into the Streamlit application.
3.  **Inference:** Using the model to perform real analysis on the user's input text.