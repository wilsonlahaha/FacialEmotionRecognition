import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import torch

# Set page configuration
st.set_page_config(
    page_title="Emotion Detection & Story Generator",
    page_icon="📖",
    layout="wide"
)

# Title and description
st.title("📖 Emotion Detection & Story Generator")
st.markdown("""
Detect emotions from facial expressions and generate creative stories using FLAN-T5!
Upload an image of a person's face to get started.
""")

# Sidebar for information and settings
with st.sidebar:
    st.header("⚙️ Settings & Information")
    
    # Model selection
    model_size = st.selectbox(
        "FLAN-T5 Model Size",
        ["Large", "Base", "Small", "XL"],
        index=0,
        help="Larger models generate better stories but require more memory"
    )
    
    # Story length settings
    st.subheader("Story Settings")
    max_length = st.slider("Maximum Story Length", 200, 1000, 600, 50)
    min_length = st.slider("Minimum Story Length", 50, 500, 200, 50)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 3.0, 2.5, 0.1,
                                 help="Higher values reduce repetition in the story")
    
    st.divider()
    
    # Information
    st.subheader("ℹ️ About")
    st.markdown("""
    **Models Used:**
    1. **Emotion Detection:** LaiMein/Facial-Emotion-Recognition
    2. **Story Generation:** Google FLAN-T5
    
    **Model Sizes:**
    - Small: 80M parameters
    - Base: 250M parameters  
    - Large: 780M parameters
    - XL: 3B parameters
    
    ⚠️ **Note:** XL model requires ~12GB RAM
    """)
    
    st.divider()
    st.caption("Models will download on first use (~1-11GB depending on selection)")

# Map model size to model names
MODEL_MAP = {
    "Small": "google/flan-t5-small",
    "Base": "google/flan-t5-base",
    "Large": "google/flan-t5-large",
    "XL": "google/flan-t5-xl"
}

# Initialize session state for models
@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model"""
    st.info("Loading emotion detection model...")
    return pipeline("image-classification", model="LaiMein/Facial-Emotion-Recognition")

@st.cache_resource
def load_story_model(model_name):
    """Load the story generation model with caching"""
    st.info(f"Loading FLAN-T5 model ({model_name.split('/')[-1]})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# File uploader section
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    help="Upload a clear image of a face for best results"
)

# Create layout columns
col1, col2 = st.columns([1, 1])

with col1:
    if uploaded_file is not None:
        # Display uploaded image
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image details
            with st.expander("📊 Image Details"):
                st.write(f"**Format:** {image.format}")
                st.write(f"**Size:** {image.size} pixels")
                st.write(f"**Mode:** {image.mode}")
        except Exception as e:
            st.error(f"Error loading image: {e}")

with col2:
    if uploaded_file is not None:
        st.subheader("🎭 Emotion Analysis")
        
        # Process button
        if st.button("✨ Analyze Emotion & Generate Story", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    # Load emotion model
                    emotion_pipe = load_emotion_model()
                    
                    # Create progress containers
                    progress_bar = st.progress(0, text="Starting analysis...")
                    
                    # Step 1: Detect emotion
                    progress_bar.progress(25, text="Detecting emotion...")
                    emotion_results = emotion_pipe(image)
                    
                    # Display emotion results
                    emotion_pred = emotion_results[0]['label']
                    emotion_score = emotion_results[0]['score']
                    
                    # Show emotion with confidence
                    st.success(f"**Detected Emotion:** {emotion_pred.upper()}")
                    st.metric("Confidence", f"{emotion_score:.1%}")
                    
                    # Show all detected emotions
                    with st.expander("View all emotion predictions"):
                        for result in emotion_results:
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.write(f"**{result['label']}**")
                            with col_b:
                                st.write(f"{result['score']:.1%}")
                    
                    progress_bar.progress(50, text="Loading story generator...")
                    
                    # Step 2: Load story model
                    selected_model = MODEL_MAP[model_size]
                    tokenizer, model = load_story_model(selected_model)
                    
                    progress_bar.progress(75, text="Generating story...")
                    
                    # Step 3: Generate story
                    input_text = f"Write a creative and engaging story about a person who is feeling {emotion_pred.lower()}. The story should be between 200 and 600 words."
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
                    
                    # Generate with parameters
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_length=max_length,
                            min_length=min_length,
                            repetition_penalty=repetition_penalty,
                            num_beams=4,
                            early_stopping=True
                        )
                    
                    # Decode the story
                    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    progress_bar.progress(100, text="Complete!")
                    progress_bar.empty()
                    
                    # Display the story
                    st.subheader(f"📚 Story: The {emotion_pred.title()} Journey")
                    
                    # Create a nice container for the story
                    story_container = st.container()
                    with story_container:
                        st.markdown(f"""
                        <div style='
                            background-color: #f8f9fa;
                            padding: 20px;
                            border-radius: 10px;
                            border-left: 5px solid #4CAF50;
                            margin: 10px 0;
                        '>
                            {story}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Story statistics
                    word_count = len(story.split())
                    st.caption(f"Story length: {word_count} words | Model: {model_size} | Generated with repetition penalty: {repetition_penalty}")
                    
                    # Download options
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.download_button(
                            label="📥 Download Story",
                            data=story,
                            file_name=f"{emotion_pred}_story.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col_d2:
                        # Copy to clipboard button
                        if st.button("📋 Copy to Clipboard", use_container_width=True):
                            st.code(story, language=None)
                            st.success("Story copied to clipboard!")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("""
                    **Troubleshooting tips:**
                    1. Try a smaller FLAN-T5 model size
                    2. Ensure you have enough memory (RAM)
                    3. Try a different image
                    4. Check your internet connection (models download on first use)
                    """)
        
        else:
            # Show instructions when no processing yet
            st.info("""
            **Ready to analyze!**
            
            Click the button above to:
            1. Detect the primary emotion in the image
            2. Generate a creative story based on that emotion
            
            Adjust story settings in the sidebar for different outputs.
            """)
            
            # Show current settings
            with st.expander("Current Settings"):
                st.write(f"**Model Size:** {model_size}")
                st.write(f"**Story Length:** {min_length}-{max_length} tokens")
                st.write(f"**Repetition Penalty:** {repetition_penalty}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit, Transformers, and FLAN-T5</p>
    <p>Emotion Detection + Creative Story Generation</p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .story-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)
