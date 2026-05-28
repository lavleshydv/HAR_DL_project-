# HAR Project - Professional Folder Structure

```
HAR_Video_Classification/
│
├── README.md                           # Project overview and guide
├── SETUP.md                            # Colab setup instructions
├── CONCEPTS.md                         # Detailed explanations of concepts
├── RESUME_NOTES.md                     # How to present this project
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
│
├── notebooks/
│   ├── 01_Colab_Complete_Setup.ipynb           # Main Colab notebook (RUN THIS)
│   ├── 02_Dataset_Preparation.ipynb            # Dataset download & preprocessing
│   ├── 03_Model_Training.ipynb                 # Training pipeline
│   ├── 04_Model_Evaluation.ipynb               # Evaluation & visualization
│   └── 05_Inference_Demo.ipynb                 # Inference & deployment demo
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration (paths, hyperparams)
│   ├── utils.py                        # Utility functions
│   ├── dataset_handler.py              # Dataset loading & preprocessing
│   ├── model_builder.py                # Model architecture
│   ├── trainer.py                      # Training loop
│   ├── evaluator.py                    # Evaluation metrics
│   ├── inference.py                    # Inference pipeline
│   └── video_processor.py              # Video processing utilities
│
├── models/
│   ├── best_model.h5                   # Saved model (after training)
│   ├── label_encoder.pkl               # Activity labels encoder
│   └── training_history.json           # Training metrics
│
├── data/
│   ├── raw/                            # Raw videos from UCF101
│   ├── processed/                      # Preprocessed frames
│   ├── labels.txt                      # List of activity classes
│   └── train_test_split.json           # Data split info
│
├── outputs/
│   ├── predictions/                    # Prediction outputs
│   ├── visualizations/                 # Graphs and plots
│   └── sample_videos/                  # Processed videos with predictions
│
├── deployment/
│   ├── app_streamlit.py                # Streamlit deployment app
│   ├── app_gradio.py                   # Gradio deployment app
│   ├── requirements_deploy.txt         # Deployment dependencies
│   └── inference_api.py                # REST API endpoint
│
├── research/
│   ├── paper_references.md             # Research papers
│   ├── interview_questions.md          # Interview prep
│   └── optimization_tips.md            # Performance optimization
│
└── tests/
    ├── test_preprocessing.py           # Preprocessing tests
    ├── test_model.py                   # Model tests
    └── test_inference.py               # Inference tests
```

## Why This Structure?

1. **notebooks/** → Colab-friendly, sequential workflow
2. **src/** → Reusable, modular, production-quality code
3. **models/** → Trained artifacts (easy to reload)
4. **deployment/** → Ready for Streamlit/Gradio deployment
5. **research/** → Interview + learning resources
