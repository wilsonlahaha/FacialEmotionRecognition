import streamlit as st
from transformers import pipeline
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("😊 Emotion Detection & Story Generator")
st.markdown("""
This app detects emotions from facial expressions and generates a story based on the detected emotion.
Upload an image of a person's face to get started!
""")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Models Used:**
    1. **Emotion Detection:** LaiMein/Facial-Emotion-Recognition
    2. **Story Generation:** openai-community/gpt2-xl
    
    **How it works:**
    1. Upload a facial image
    2. The app detects the primary emotion
    3. A story is generated based on that emotion
    """)
    
    st.divider()
    st.caption("Note: Processing may take a moment for the first run as models download.")

# Initialize session state for models
@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model"""
    return pipeline("image-classification", model="LaiMein/Facial-Emotion-Recognition")

@st.cache_resource
def load_story_model():
    """Load the story generation model"""
    return pipeline("text-generation", model="openai-community/gpt2-xl")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image containing a face for emotion detection"
)

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Add a process button
        process_btn = st.button("🎭 Detect Emotion & Generate Story", type="primary")

with col2:
    if uploaded_file is not None and 'process_btn' in locals() and process_btn:
        with st.spinner("Loading models and processing..."):
            try:
                # Load models
                emotion_pipe = load_emotion_model()
                story_pipe = load_story_model()
                
                # Detect emotion
                with st.status("Detecting emotion...", expanded=True) as status:
                    emotion_pred = emotion_pipe(image)[0]['label'].lower()
                    status.update(label=f"✅ Emotion detected: **{emotion_pred.upper()}**", state="complete")
                
                # Display emotion result
                st.subheader(f"Detected Emotion: **{emotion_pred.upper()}**")
                
                # Generate story
                with st.status("Generating story...", expanded=True) as status:
                    story = story_pipe(
                        f"Tell a story about this {emotion_pred} person",
                        max_length=500,
                        min_length=200,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9
                    )
                    status.update(label="✅ Story generated!", state="complete")
                
                # Display story
                st.subheader("📖 Generated Story")
                story_text = story[0]['generated_text']
                
                # Format the story with better readability
                formatted_story = story_text.replace(f"Tell a story about this {emotion_pred} person", "").strip()
                if formatted_story:
                    st.write(formatted_story)
                else:
                    st.write(story_text)
                
                # Add download button for the story
                st.download_button(
                    label="📥 Download Story",
                    data=story_text,
                    file_name=f"emotion_story_{emotion_pred}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again or upload a different image.")
    
    elif uploaded_file is not None:
        st.info("Click the button to process the image and generate a story.")
    
    else:
        st.info("👈 Please upload an image to get started")

# Add some styling
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .stStatus {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Built with Streamlit and Hugging Face Transformers | Emotion Detection & Story Generation")
