import streamlit as st
from transformers import pipeline
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("😊 Emotion Detection & Storytelling")
st.markdown("""
This app detects emotions from facial images and generates a short story based on the detected emotion.
Upload a facial image to get started!
""")

# Initialize session state for caching models
@st.cache_resource
def load_emotion_model():
    """Load and cache the emotion detection model"""
    try:
        pipe = pipeline("image-classification", model="LaiMein/Facial-Emotion-Recognition")
        return pipe
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None

@st.cache_resource
def load_story_model():
    """Load and cache the story generation model"""
    try:
        pipe = pipeline("text-generation", model="openai-community/gpt2-xl")
        return pipe
    except Exception as e:
        st.error(f"Error loading story model: {e}")
        return None

# Sidebar for additional options
with st.sidebar:
    st.header("⚙️ Settings")
    story_length = st.slider("Story Length (tokens)", min_value=50, max_value=200, value=100, step=10)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Models used:**
    - Emotion Detection: LaiMein/Facial-Emotion-Recognition
    - Story Generation: GPT-2 XL
    
    Upload an image with a clear face for best results!
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image with a clear facial expression"
    )
    
    # Display uploaded image
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Open and display image
            image = Image.open(tmp_path).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add a process button
            process_button = st.button("🔍 Detect Emotion & Generate Story", type="primary")
            
            if process_button:
                with st.spinner("Loading models and processing..."):
                    # Load models
                    emotion_pipe = load_emotion_model()
                    story_pipe = load_story_model()
                    
                    if emotion_pipe and story_pipe:
                        # Detect emotion
                        with st.status("Detecting emotion...", expanded=True) as status:
                            emotion_pred = emotion_pipe(image)[0]['label']
                            status.update(label=f"Emotion detected: {emotion_pred}", state="running")
                            
                            # Display emotion with emoji
                            emotion_emojis = {
                                'happy': '😊',
                                'sad': '😢',
                                'angry': '😠',
                                'surprised': '😲',
                                'fearful': '😨',
                                'disgusted': '😖',
                                'neutral': '😐'
                            }
                            
                            emoji = emotion_emojis.get(emotion_pred.lower(), '😊')
                            st.subheader(f"{emoji} Detected Emotion: {emotion_pred}")
                            
                            # Generate story
                            status.update(label="Generating story...", state="running")
                            story = story_pipe(
                                f"Tell a short story about a {emotion_pred.lower()} person",
                                max_length=story_length,
                                max_new_tokens=story_length,
                                do_sample=True,
                                temperature=0.7
                            )
                            
                            generated_text = story[0]['generated_text']
                            status.update(label="Processing complete!", state="complete")
                        
                        # Display results in second column
                        with col2:
                            st.header("📖 Generated Story")
                            
                            # Emotion badge
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                                <h4 style="margin: 0;">Based on: <span style="color: #ff4b4b;">{emotion_pred}</span></h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Story in a nice container
                            st.markdown("""
                            <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-top: 10px;">
                            """, unsafe_allow_html=True)
                            st.write(generated_text)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Add download button for story
                            st.download_button(
                                label="📥 Download Story",
                                data=generated_text,
                                file_name=f"story_{emotion_pred.lower()}.txt",
                                mime="text/plain"
                            )
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        # Display sample image and instructions
        st.info("👆 Please upload an image to begin")
        st.markdown("""
        **Example images to try:**
        - Clear facial photo
        - Good lighting
        - Face clearly visible
        """)

# Display sample output when no image is uploaded
if uploaded_file is None:
    with col2:
        st.header("📖 Sample Output")
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h4 style="color: #ff4b4b;">Example Emotion: Happy 😊</h4>
            <p><i>Once upon a time, there was a happy person named Alex. Alex had just received wonderful news about a long-awaited promotion. The smile on their face could light up an entire room. They decided to share their joy by treating friends to dinner, spreading happiness wherever they went. The world seemed brighter through Alex's cheerful eyes...</i></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using Hugging Face Transformers & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
