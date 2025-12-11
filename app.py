import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Emotion Detection & Story Generator",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("😊 Emotion Detection using Facial Emotion Recognition")
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses:
    1. **LaiMein/Facial-Emotion-Recognition** - For detecting emotions from facial expressions
    2. **GPT-2** - For generating creative stories based on detected emotions
    
    Upload an image of a face to detect the emotion and generate a story!
    """)
    st.markdown("---")
    st.markdown("**Note:** The models will be downloaded on first run (this may take a few minutes)")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # Image uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing a face for emotion detection"
    )
    
    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Display image info
        st.info(f"**Image details:** {image.size[0]}×{image.size[1]} pixels | Mode: {image.mode}")
        
        # Process button
        if st.button("🔍 Detect Emotion & Generate Story", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and generating story..."):
                try:
                    # Initialize emotion detection pipeline
                    with st.status("Loading emotion detection model...", expanded=True) as status:
                        emotion_pipe = pipeline("image-classification", model="LaiMein/Facial-Emotion-Recognition")
                        status.update(label="✅ Emotion detection model loaded!", state="complete")
                    
                    # Detect emotion
                    with st.status("Detecting emotion from image...", expanded=True) as status:
                        emotion_pred = emotion_pipe(image)[0]['label'].lower()
                        status.update(label=f"✅ Emotion detected: **{emotion_pred.upper()}**", state="complete")
                    
                    # Display emotion result
                    st.success(f"**Detected Emotion:** {emotion_pred.upper()}")
                    
                    # Generate story
                    with st.status("Generating story based on emotion...", expanded=True) as status:
                        story_pipe = pipeline("text-generation", model="openai-community/gpt2")
                        status.update(label="✅ Story generator model loaded!", state="complete")
                    
                    with st.status("Creating creative story...", expanded=True) as status:
                        story = story_pipe(
                            f"Tell a short creative story about this {emotion_pred} person",
                            max_length=500,
                            max_new_tokens=500,
                            do_sample=True,
                            temperature=0.8
                        )
                        status.update(label="✅ Story generated successfully!", state="complete")
                    
                    # Display story in col2
                    with col2:
                        st.header("📖 Generated Story")
                        st.markdown("---")
                        st.markdown(f"**Based on the emotion:** _{emotion_pred}_")
                        st.markdown("---")
                        
                        # Create a nice container for the story
                        story_container = st.container()
                        with story_container:
                            st.markdown("### 📝 The Story:")
                            st.write(story[0]['generated_text'])
                        
                        # Add download button for the story
                        story_text = story[0]['generated_text']
                        st.download_button(
                            label="📥 Download Story as Text",
                            data=story_text,
                            file_name=f"story_{emotion_pred}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again or use a different image.")
    
    else:
        # Show placeholder when no image is uploaded
        st.info("👆 Please upload an image to get started")
        
        # Display sample image or instructions
        with col2:
            st.header("📖 How it Works")
            st.markdown("""
            1. **Upload** an image containing a face
            2. **Click** the 'Detect Emotion & Generate Story' button
            3. **View** the detected emotion and generated story
            
            The app will:
            - Analyze facial expressions using AI
            - Detect the primary emotion
            - Generate a creative story based on that emotion
            
            **Example emotions detected:**
            - 😊 Happy
            - 😢 Sad  
            - 😠 Angry
            - 😲 Surprised
            - 😐 Neutral
            - 😨 Fearful
            """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by 🤗 Transformers | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
