import streamlit as st
import random
import time

def main():
    """
    Main function to run the AI Content Detector Streamlit app.
    """
    st.set_page_config(page_title="AI Content Detector", page_icon="🤖")

    st.title("AI Content Detector")
    st.write("This tool helps you detect AI-generated content. Paste your text below to get started.")

    # Initialize session state
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""

    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram often used for testing typefaces and keyboards. It contains every letter of the English alphabet.",
        "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        "The sun always shines brightest after the rain. Life is full of ups and downs, but it's important to remember that challenges are temporary and opportunities for growth are always present. Embrace change and keep moving forward.",
        "In a world where information is constantly flowing, critical thinking skills are more important than ever. Learning to evaluate sources, identify biases, and construct logical arguments are essential for navigating complex issues and making informed decisions.",
        "The history of the internet is a fascinating journey from a military research project to a global communication network. It has revolutionized how we work, learn, and interact with each other, constantly evolving with new technologies and applications."
    ]

    # The text area value is bound to st.session_state.text_content.
    st.session_state.text_content = st.text_area(
        "Enter text to analyze",
        value=st.session_state.text_content,
        height=200
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Use Sample"):
            st.session_state.text_content = random.choice(sample_texts)
            st.rerun()

    with col2:
        if st.button("Check content"):
            if st.session_state.text_content:
                with st.spinner("Analyzing..."):
                    time.sleep(2)  # Simulate analysis time
                    human_score = random.randint(50, 100)
                    
                st.subheader("Analysis Result")
                
                # Custom progress bar
                st.write(f"**{human_score}% Human Content**")
                st.progress(human_score / 100)

                if human_score > 80:
                    st.success("This text is likely written by a human.")
                elif human_score > 60:
                    st.warning("This text may contain some AI-generated content.")
                else:
                    st.error("This text is likely AI-generated.")
            else:
                st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()

