# AI Content Detector

This is a simple web application built with Streamlit that demonstrates an AI content detector. Users can paste text into a text area or use a sample text to get a simulated analysis of whether the content is AI-generated or human-written.

## Features

*   **Text Analysis:** Paste any text to analyze it.
*   **Sample Text:** Use the "Use Sample" button to load a random sample text for a quick demonstration.
*   **Simulated Results:** The application provides a "Human Content" score as a percentage, along with a visual progress bar and a qualitative assessment.

## Project Structure

This project was developed using the CRISP-DM methodology as a framework. For more details on the development process, please see the `project_plan.md` file.

## How to Run

### Prerequisites

*   Python 3.7+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dses50117/HW5.git
    cd HW5
    ```

2.  **Install the required dependencies:**
    The only dependency is Streamlit.
    ```bash
    pip install streamlit
    ```

### Running the Application

To start the Streamlit server, run the following command in your terminal:

```bash
streamlit run app.py
```

The application will open in a new tab in your default web browser.
