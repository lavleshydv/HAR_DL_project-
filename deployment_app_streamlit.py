"""
Streamlit Web App for Human Activity Recognition
Provides easy web interface for video upload and activity prediction

Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path
import tempfile
import os

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Activity Recognition",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================

st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================

st.sidebar.title("🎬 HAR System")
st.sidebar.divider()

# Model selection
mode = st.sidebar.radio(
    "Select Mode:",
    ["📹 Upload Video", "🎥 Webcam", "📊 About"]
)

# ==================== LOAD MODEL ====================

@st.cache_resource
def load_trained_model():
    """Load trained model from disk"""
    try:
        model_path = "models/best_model.h5"
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

@st.cache_resource
def load_classes():
    """Load activity class names"""
    try:
        classes_path = "models/classes.pkl"
        with open(classes_path, 'rb') as f:
            classes = pickle.load(f)
        return classes
    except Exception as e:
        st.error(f"Could not load classes: {e}")
        return None

# ==================== VIDEO PROCESSING ====================

def preprocess_video(video_path, num_frames=20, frame_size=112):
    \"\"\"
    Extract and preprocess frames from video

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        frame_size: Target frame size (frame_size x frame_size)

    Returns:
        Array of shape (num_frames, frame_size, frame_size, 3)
    \"\"\"
    frames = []

    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            st.error("Could not read video file")
            return None

        # Calculate which frames to extract
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frame_idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx in indices:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (frame_size, frame_size))
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            frame_idx += 1

            if len(frames) == num_frames:
                break

        cap.release()

        if len(frames) != num_frames:
            st.error(f"Could only extract {len(frames)}/{num_frames} frames")
            return None

        return np.array(frames)

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def get_prediction(frames, model, classes):
    \"\"\"
    Get activity prediction from frames

    Args:
        frames: Preprocessed frame sequence
        model: Trained model
        classes: Activity class names

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    \"\"\"
    # Add batch dimension
    input_data = np.expand_dims(frames, axis=0)

    # Make prediction
    predictions = model.predict(input_data, verbose=0)

    # Get top prediction
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = classes[predicted_idx]

    return predicted_class, confidence, predictions[0]

# ==================== MODE 1: VIDEO UPLOAD ====================

if mode == "📹 Upload Video":
    st.title("🎬 Activity Recognition from Video")

    st.markdown("""
    Upload a video file and the AI will detect the activity!

    **Supported activities:**
    - 🚶 WalkingWithDog
    - 🏃 JumpingJack
    - 👊 Punch
    - 🏀 Basketball
    - 🐴 HorseRiding
    - 💪 PushUps
    - 🧘 TaiChi
    - ⚽ SoccerJuggling
    """)

    st.divider()

    # Load model and classes
    model = load_trained_model()
    classes = load_classes()

    if model is None or classes is None:
        st.error("❌ Could not load model or classes. Check that model files exist.")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        try:
            # Create columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📹 Video Preview")
                st.video(uploaded_file)

            with col2:
                st.subheader("🤖 Prediction Results")

                # Show processing status
                with st.spinner("Processing video... This may take a minute..."):
                    # Preprocess video
                    frames = preprocess_video(temp_video_path, num_frames=20, frame_size=112)

                    if frames is not None:
                        # Get prediction
                        predicted_class, confidence, probabilities = get_prediction(
                            frames, model, classes
                        )

                        # Display main prediction
                        st.metric(
                            label="Predicted Activity",
                            value=predicted_class,
                            delta=f"{confidence*100:.1f}% confident"
                        )

                        # Display confidence
                        st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")

                        # Show all probabilities
                        st.subheader("All Probabilities:")
                        prob_data = {
                            classes[i]: float(probabilities[i])
                            for i in range(len(classes))
                        }

                        # Sort by probability
                        sorted_probs = dict(sorted(
                            prob_data.items(),
                            key=lambda x: x[1],
                            reverse=True
                        ))

                        # Display as bar chart
                        st.bar_chart(sorted_probs)

                        # Display as table
                        st.dataframe(
                            {
                                "Activity": list(sorted_probs.keys()),
                                "Probability": [f"{v*100:.2f}%" for v in sorted_probs.values()]
                            },
                            use_container_width=True
                        )

        finally:
            # Clean up temporary file
            os.unlink(temp_video_path)

# ==================== MODE 2: WEBCAM (Placeholder) ====================

elif mode == "🎥 Webcam":
    st.title("🎥 Real-Time Activity Recognition")

    st.info("""
    📌 **Note:** Webcam access via browser is limited in Streamlit.

    For better webcam performance, run the desktop application:
    ```bash
    python webcam_inference.py
    ```

    Or use the Gradio app which has better webcam support:
    ```bash
    python app_gradio.py
    ```
    """)

    # Placeholder for webcam features
    st.markdown("""
    ### How it would work:
    1. Start webcam capture
    2. Buffer frames in real-time
    3. Run prediction every 0.5s
    4. Display activity on screen
    5. Show confidence meter

    The desktop script (`webcam_inference.py`) provides this functionality!
    """)

# ==================== MODE 3: ABOUT ====================

elif mode == "📊 About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 🎯 Human Activity Recognition (HAR)

    This is a **deep learning system** that recognizes human activities from videos.

    ### 🏗️ Architecture

    **MobileNetV2 + LSTM:**
    - **MobileNetV2**: Lightweight CNN for feature extraction
    - **LSTM**: Learns temporal patterns from sequences
    - **Total size**: ~50 MB (small and deployable)

    ### 📊 Model Details

    | Component | Details |
    |-----------|---------|
    | **CNN** | MobileNetV2 (pre-trained on ImageNet) |
    | **Temporal** | LSTM with 128 units |
    | **Architecture** | TimeDistributed → LSTM → Dense → Softmax |
    | **Input** | 20 frames, 112×112 pixels, RGB |
    | **Output** | 8 activity classes |
    | **Training** | Transfer learning + fine-tuning |

    ### 📈 Performance

    | Metric | Value |
    |--------|-------|
    | **Accuracy** | ~85% |
    | **Model size** | ~50 MB |
    | **Inference time** | ~0.3-0.5s |
    | **GPU memory** | ~2 GB |

    ### 🎓 Technologies Used

    - **Deep Learning**: TensorFlow/Keras
    - **Computer Vision**: OpenCV
    - **Data**: NumPy, Pandas
    - **Deployment**: Streamlit
    - **Visualization**: Matplotlib, Plotly

    ### 📝 Dataset

    **UCF101** - University of Central Florida dataset
    - 13,320 videos total
    - 101 action categories
    - We use 8 lightweight classes for this project

    ### 🚀 Features

    ✅ Video upload and prediction
    ✅ Real-time webcam (desktop app)
    ✅ Web-based interface
    ✅ Confidence scores
    ✅ Batch processing
    ✅ Model saving/loading

    ### 💡 Use Cases

    - Security & surveillance
    - Healthcare (activity monitoring)
    - Sports analytics
    - Human-computer interaction
    - Content recommendation
    - Fitness tracking

    ### 📚 Learn More

    - **Concepts**: See CONCEPTS.md for detailed explanations
    - **Code**: Check src/ folder for implementation
    - **Resume**: See RESUME_NOTES.md for presentation tips

    ### 🔬 Technologies

    **Frameworks:**
    - TensorFlow 2.10+
    - Keras
    - OpenCV

    **Techniques:**
    - Transfer learning
    - Fine-tuning
    - Temporal learning with LSTM
    - Data augmentation

    ### 👨‍💻 Author Notes

    This project was built for:
    - **Learning**: Understand HAR and deep learning
    - **Portfolio**: Showcase computer vision skills
    - **Production**: Deploy real applications
    - **Interviews**: Discuss interesting ML projects

    ### 🎯 Next Steps

    1. Upload a video to test the model
    2. Use desktop app for webcam
    3. Modify code for your own activities
    4. Deploy to the cloud
    5. Extend to more classes

    ---

    **Built with ❤️ using TensorFlow and Streamlit**
    """)

# ==================== FOOTER ====================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Architecture", "MobileNetV2 + LSTM")

with col2:
    st.metric("Activities", "8 Classes")

with col3:
    st.metric("Accuracy", "~85%")

st.markdown("""
---
*For best results, use videos showing clear, complete actions (3-10 seconds long)*
""")
