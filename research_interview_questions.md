# 🎤 Interview Preparation - HAR Project

This guide prepares you to discuss your Human Activity Recognition project in technical interviews.

## 📋 Table of Contents

1. [General Questions](#general-questions)
2. [Technical Deep Dives](#technical-deep-dives)
3. [Architecture Questions](#architecture-questions)
4. [ML/DL Questions](#mldi-questions)
5. [Code & Implementation](#code--implementation)
6. [Optimization & Scaling](#optimization--scaling)
7. [Deployment Questions](#deployment-questions)
8. [Problem Solving](#problem-solving)

---

## General Questions

### Q1: Tell me about your HAR project

**Good Answer:**

> I built a Human Activity Recognition system that uses deep learning to classify activities from videos. The model uses MobileNetV2 + LSTM architecture achieving 85% accuracy. I trained it on UCF101 dataset using 8 activity classes. The system can predict activities from uploaded videos or webcam feeds and is deployed via Streamlit. I specifically optimized it for Google Colab's free GPU tier, which was a key constraint.

**What they like:**

- ✅ Clear problem statement
- ✅ Specific metrics (85% accuracy)
- ✅ Architecture choice (MobileNetV2 + LSTM)
- ✅ Constraints acknowledged (free GPU)
- ✅ Multiple interfaces (upload, webcam, deployment)

**Follow-up points to cover:**

- Dataset: UCF101, 8 classes, lightweight
- Training time: 30-60 minutes on Colab
- Model size: ~50 MB
- Inference: 0.3-0.5s per video

---

### Q2: Why did you choose this particular project?

**Good Answer:**

> I chose HAR because it combines multiple skills: computer vision, deep learning, and deployment. It's also practically useful for security, healthcare, and sports analytics. The project scope was manageable for learning but complex enough to showcase multiple techniques. Plus, getting it to work on free Colab GPU was a good optimization challenge.

**What they like:**

- ✅ Practical applications
- ✅ Multiple technical skills
- ✅ Good learning opportunity
- ✅ Optimization mindset

---

### Q3: What was the most challenging part?

**Good Answer (Pick one):**

Option A - Data Handling:

> Getting the dataset preprocessing right. Initially I was loading entire videos into memory which crashed the system. I solved it by implementing streaming frame extraction with dynamic batching.

Option B - GPU Optimization:

> Making it work on free Colab GPU with limited memory. I experimented with frame size (112×112), sequence length (20 frames), and batch size (32) to find the sweet spot between accuracy and memory usage.

Option C - Architecture Decision:

> Choosing between 3D CNN and 2D CNN + RNN. 3D CNNs were out of memory but 2D CNN + LSTM worked great. The key insight was that MobileNetV2 is small enough to run TimeDistributed while LSTM handles temporal learning efficiently.

**What they like:**

- ✅ Concrete problem
- ✅ Solution explanation
- ✅ Trade-off analysis
- ✅ Technical depth

---

## Technical Deep Dives

### Q4: Explain your model architecture in detail

**Good Answer:**

```
INPUT: Video (20 frames, 112×112×3 RGB pixels)
       └─ Why 20 frames? Captures ~0.6-1 second of activity
       └─ Why 112×112? Balances speed (6.25× faster than 224×224)
                       and accuracy (large enough for features)

STAGE 1: CNN Feature Extraction
├─ TimeDistributed(MobileNetV2) layer
├─ Each frame → MobileNetV2 → 1280-dim feature vector
├─ Result: (batch, 20, 1280)
└─ Why MobileNetV2?
   ├─ Only 9 MB (vs ResNet 100+ MB)
   ├─ Pre-trained on ImageNet (1.2M images)
   ├─ Designed for mobile/edge devices
   └─ Still powerful enough for feature extraction

STAGE 2: Temporal Learning
├─ LSTM (128 units)
├─ Processes 20 feature vectors sequentially
├─ Maintains hidden state across frames
├─ Result: Single 128-dim vector summarizing motion
└─ Why LSTM?
   ├─ Learns temporal patterns
   ├─ Remembers previous frames
   ├─ Effective for sequence data
   └─ Only 128 units keeps it lightweight

STAGE 3: Classification
├─ Dense(64, ReLU) + Dropout(0.5)
├─ Dense(32, ReLU) + Dropout(0.5)
├─ Dense(8, Softmax)
└─ Result: Probability for each of 8 activities

OUTPUT: Activity class with confidence score
```

**Why this architecture?**

1. MobileNetV2 extracts spatial features efficiently
2. LSTM learns temporal dependencies
3. Dense layers provide expressive classification
4. Dropout prevents overfitting
5. Lightweight enough for free GPU

---

### Q5: Why LSTM and not 3D CNN?

**Good Answer:**

| Aspect               | LSTM + CNN2D | 3D CNN                  |
| -------------------- | ------------ | ----------------------- |
| **Memory**           | ~2 GB        | ~8-10 GB ❌             |
| **Speed**            | Fast         | Slow                    |
| **Accuracy**         | ~85% ✅      | Slightly better but OOM |
| **Interpretability** | Good         | OK                      |
| **Production Use**   | Proven ✅    | Overkill                |

**Key Difference:**

- **3D CNN**: Learns spatial-temporal features jointly
  - Captures motion direction/speed
  - Huge number of parameters
  - Needs much memory

- **2D CNN + LSTM**: Learns features then temporal
  - Separates concerns
  - Reuses pre-trained weights
  - Memory efficient

**Why I chose LSTM:**

> Given the constraint of free Colab GPU (12 GB), 3D CNN would crash after 2-3 videos. LSTM + MobileNetV2 stays well within memory while still achieving 85% accuracy. The separation of spatial and temporal learning is cleaner architecturally.

---

### Q6: How did you handle the temporal aspect?

**Good Answer:**

**Problem:** A single frame doesn't tell you if someone is walking or standing. You need to see the motion over time.

**Solution:**

1. Extract 20 frames from video (equally spaced)
2. Pass each through MobileNetV2 → 1280-dim features
3. Feed all 20 feature vectors to LSTM sequentially
4. LSTM learns: "Frame 1→2: leg moves → walking", "Frame 1→10: small changes → standing"

**Example:** Jumping Jack

```
Frame 1: Person standing (normalized body position)
Frame 5: Arms up, legs spread (different position)
Frame 10: Back to standing

LSTM sees this sequence and learns: "This is a jumping jack"
```

**Key insight:** LSTM has memory gates that:

- Remember important past info (which frames showed motion)
- Forget irrelevant info (exact pixel values)
- Connect temporal patterns (arms up → movement)

---

## Architecture Questions

### Q7: Why freeze early MobileNetV2 layers?

**Answer:**

**Early layers** (Layers 1-100):

- Learn generic features: edges, colors, corners
- Useful for ANY vision task
- Pre-trained on 1.2M ImageNet images
- No need to retrain

**Late layers** (Layers 101-125):

- Learn high-level patterns
- Could be specific to ImageNet (dog breeds, car models)
- Need adaptation for activity recognition
- Keep trainable for fine-tuning

**Analogy:** Learning sports

```
Don't need to relearn: "How to see humans" (early layers)
Do need to learn: "How to recognize basketball vs soccer" (late layers)
```

**Impact:**

- Faster training (only update 20 layers vs 125)
- Better regularization (prevent overfitting)
- Use learned knowledge effectively

---

### Q8: How do you handle variable-length videos?

**Answer:**

**Problem:** Videos have different lengths (10 sec, 20 sec, 1 min)

**Solution:** Always extract exactly 20 frames

```
Video length | Method
10 seconds   | Extract frames 0, 2.5, 5, 7.5, ... (equally spaced)
20 seconds   | Extract frames 0, 5, 10, 15, ... (equally spaced)
60 seconds   | Extract frames 0, 15, 30, 45, ... (equally spaced)

Result: Always 20 frames regardless of input length
```

**Why uniform:**

- Model expects fixed input shape: (20, 112, 112, 3)
- Captures full action arc (beginning, middle, end)
- Balances temporal resolution vs memory

---

## ML/DL Questions

### Q9: How do you prevent overfitting?

**Answer:** I use multiple techniques:

1. **Dropout (0.5):** Randomly disable 50% of neurons
   - Forces network to learn redundant representations
   - Applied after dense layers
   - Effective for small datasets

2. **L2 Regularization:** Penalty on large weights
   - Prevents weights from growing too large
   - Value: 1e-6 (very weak, just to help)

3. **Early Stopping:** Stop training when validation loss plateaus
   - Monitor: validation_loss
   - Patience: 5 epochs
   - Restore best weights

4. **Transfer Learning:** Freeze pre-trained layers
   - Reduces parameters to train
   - Less overfitting on small data

5. **Balanced Dataset:** Equal videos per class
   - Prevents bias toward common activities

**Result:** 85% validation accuracy maintained on test set (not overfitting)

---

### Q10: How do you handle class imbalance?

**Answer:**

**In our case:** Dataset is balanced (30 videos per activity)

**But if imbalanced:** Would use:

1. **Class weights:**

```python
class_weight = {
    0: 1.0,      # Common activity
    1: 5.0,      # Rare activity
    ...
}
model.fit(X, y, class_weight=class_weight)
```

2. **Weighted accuracy:**

- F1-score (balances precision/recall)
- Macro-average (equal weight to each class)

3. **Data augmentation:**

- Oversample rare classes
- Undersample common classes

4. **Focal loss:**

- Penalizes easy examples less
- Focuses on hard examples

---

### Q11: What metrics do you use beyond accuracy?

**Answer:**

Accuracy alone is misleading! Use:

**1. Precision (per class):**

```
Of all samples predicted as "Basketball",
how many were actually "Basketball"?
Good for: Avoiding false positives
```

**2. Recall (per class):**

```
Of all true "Basketball" samples,
how many did we correctly identify?
Good for: Avoiding false negatives
```

**3. F1-Score:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Balanced metric, 0-1 scale
```

**4. Confusion Matrix:**

```
Shows which classes get confused with each other
Example: "Basketball" often confused with "Soccer"
```

**My results:**

```
              Precision  Recall  F1-Score
WalkingWithDog  0.88     0.90    0.89
JumpingJack     0.85     0.82    0.83
Basketball     0.92     0.88    0.90
...
Macro Avg       0.88     0.85    0.86
```

---

## Code & Implementation

### Q12: Walk me through your data preprocessing pipeline

**Answer:**

```python
def preprocess_video(video_path):
    # Step 1: Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Step 2: Determine which frames to extract
    # If video has 150 frames, get frames [0, 7, 15, 22, ..., 142]
    indices = np.linspace(0, total_frames-1, num_frames=20).astype(int)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in indices:
            # Step 3: Convert BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 4: Resize to 112×112
            # Why? Matches MobileNetV2 input, reduces memory
            frame = cv2.resize(frame, (112, 112))

            # Step 5: Normalize to [0, 1]
            # Why? Neural networks prefer small numbers
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

        frame_idx += 1

    cap.release()

    return np.array(frames)  # Shape: (20, 112, 112, 3)
```

**Key decisions explained:**

1. **Frame extraction:** Equally-spaced to capture action arc
2. **RGB conversion:** Standard for ML (not BGR)
3. **Resize to 112×112:** Balance speed/accuracy
4. **Normalize:** Numerical stability

---

### Q13: How do you create training batches?

**Answer:**

```python
# Create batch from 32 videos
# Each video: (20, 112, 112, 3)
# Batch: (32, 20, 112, 112, 3)

BATCH_SIZE = 32  # Why 32? Good for Colab GPU memory

X_batch = np.random.randn(32, 20, 112, 112, 3).astype(np.float32)
y_batch = np.random.randint(0, 8, 32)
y_batch_encoded = keras.utils.to_categorical(y_batch, 8)

model.fit(X_batch, y_batch_encoded)
```

**Why batch size 32?**

- Smaller: 8-16 → Slower training, noisier gradients
- Larger: 64-128 → Out of memory on Colab
- 32: Sweet spot (2-4 min per epoch, stable)

**Data generator (for large datasets):**

```python
def data_generator(video_paths, batch_size=32):
    while True:
        # Shuffle
        indices = np.random.permutation(len(video_paths))

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[indices[i:i+batch_size]]

            # Load and preprocess
            X_batch = []
            y_batch = []
            for path, label in batch_paths:
                frames = preprocess_video(path)
                X_batch.append(frames)
                y_batch.append(label)

            yield np.array(X_batch), keras.utils.to_categorical(y_batch, 8)
```

---

## Optimization & Scaling

### Q14: How would you scale to 101 classes?

**Answer:**

**Current:** 8 activities, 30 videos each = 240 videos

**To 101 activities:**

1. Download full UCF101 (13,320 videos)
2. Some challenges:

| Challenge      | Solution                                    |
| -------------- | ------------------------------------------- |
| Memory         | Use data generator (don't load all at once) |
| Compute        | Paid Colab GPU or use TPU                   |
| Training time  | Might take 5-10 hours on T4 GPU             |
| Accuracy drops | More classes = harder to distinguish        |

**Implementation:**

```python
# Use data generator for streaming
train_gen = data_generator(
    video_paths_labels,
    batch_size=32
)

model.fit(
    train_gen,
    steps_per_epoch=len(videos) // 32,
    epochs=50,
    validation_data=val_gen
)
```

**Accuracy expectations:**

- Current: 85% on 8 classes
- Full 101: Probably ~40-50% (harder problem)
- Could improve with: larger model, more augmentation, ensemble

---

### Q15: How would you improve accuracy?

**Answer:**

**Quick wins (1-2% improvement):**

1. More data per class (50 instead of 30)
2. Data augmentation (rotation, zoom, flip, brightness)
3. Train longer (50+ epochs with early stopping)

**Medium effort (3-5% improvement):**

1. Fine-tune more CNN layers (unfreeze last 50 layers)
2. Use larger frames (224×224 instead of 112×112) - requires more GPU memory
3. Longer sequences (30 frames instead of 20)
4. Two LSTM layers instead of one

**High effort (5-10% improvement):**

1. Ensemble multiple models
2. Use 3D CNN (if paid GPU available)
3. Implement attention mechanism
4. Self-supervised pre-training
5. Multi-scale temporal pyramids

**What I'd do first:** Data augmentation + longer training (20% effort for 80% of gains)

---

### Q16: How would you optimize for inference speed?

**Answer:**

**Current:** 0.3-0.5s per video

**To make faster:**

1. **Model compression:**

```python
# TensorFlow Lite (mobile deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

2. **Quantization:**

```python
# Use int8 instead of float32 (4× smaller, slightly faster)
# Loss: 1-2% accuracy
# Gain: 4× smaller model, faster inference
```

3. **Pre-computation:**

```python
# Cache CNN features (don't recompute)
# If predicting same video multiple times:
features = mobilenet(frames)  # Compute once
predictions = [lstm_classifier(features) for _ in range(10)]
```

4. **Batching inference:**

```python
# Predict multiple videos at once
videos = [video1, video2, video3]
batch_features = mobilenet(np.stack(videos))  # Faster than 3 separate calls
```

5. **Streaming inference:**

```python
# For webcam: keep last 10 frames, add 10 new
# Don't reprocess same frames
```

**Speed-accuracy trade-off:**

```
            | Accuracy | Speed | Size
Original    | 85%      | 0.5s  | 50MB
Quantized   | 83%      | 0.2s  | 12MB  ← Good choice
TFLite      | 82%      | 0.1s  | 8MB   ← Mobile
```

---

## Deployment Questions

### Q17: How would you deploy this in production?

**Answer:**

**Architecture:**

```
┌─────────────────────────┐
│   Frontend (Web/Mobile) │
├─────────────────────────┤
│   API Gateway           │
│   (Load balancing)      │
├─────────────────────────┤
│   Inference Servers     │
│   (Docker containers)   │
├─────────────────────────┤
│   GPU Machine           │
│   (TensorFlow Serving)  │
└─────────────────────────┘
```

**Steps:**

1. **Containerize:**

```dockerfile
FROM tensorflow/tensorflow:latest-gpu

COPY models/ /app/models/
COPY src/ /app/src/
COPY inference_api.py /app/

CMD ["python", "inference_api.py"]
```

2. **Create REST API:**

```python
@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.files['video']
    frames = preprocess_video(video_file)
    prediction = model.predict(frames)
    return jsonify({'activity': prediction})
```

3. **Deploy options:**

- **Heroku:** Easy, small scale
- **AWS EC2:** More control
- **Google Cloud Run:** Serverless, pay-per-use
- **Docker Swarm:** Self-hosted

4. **Optimization for production:**

- Use TensorFlow Serving (not raw model)
- Load balance across GPUs
- Cache results
- Monitor latency/accuracy

---

### Q18: How would you monitor model performance in production?

**Answer:**

**Metrics to track:**

1. **Inference latency:** Should stay ~0.5s
2. **Accuracy:** Might drift over time
3. **Error rates:** False positives/negatives
4. **GPU utilization:** Cost optimization
5. **Requests per second:** Capacity

**Implementation:**

```python
@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()

    # Prediction
    result = model.predict(video)
    latency = time.time() - start

    # Log metrics
    metrics.record({
        'latency_ms': latency * 1000,
        'predicted_class': result['activity'],
        'confidence': result['confidence'],
        'timestamp': datetime.now()
    })

    # Alert if latency > 1s
    if latency > 1.0:
        send_alert(f"Slow inference: {latency}s")

    return result
```

---

## Problem Solving

### Q19: What if your model starts failing in production?

**Answer:** Systematic debugging:

1. **Check data:**

```python
# Did data distribution change?
# Are videos in different format/resolution?
# Are new activity types not in training?
```

2. **Check model:**

```python
# Did it get corrupted during deployment?
# Is quantized version less accurate than expected?
# Run on old test set - if it fails, data issue
```

3. **Check environment:**

```python
# Different GPU causes different predictions? (No, shouldn't)
# Different TensorFlow version? (Yes, can affect)
# Different preprocessing in production? (Common bug!)
```

4. **Common causes:**

- Input preprocessing changed (wrong resize)
- Label mapping changed
- Model version mismatch
- GPU numerical precision

5. **Solution:**

```python
# Add assertions
assert frames.shape == (20, 112, 112, 3)
assert frames.min() >= 0 and frames.max() <= 1.0
assert predictions.sum() == pytest.approx(1.0)
```

---

### Q20: Describe a time you debugged a hard problem

**Good scenarios to discuss:**

**Scenario 1: OOM Error**

> Initially got "CUDA out of memory" after 5 videos. Realized I was loading entire dataset into memory. Solution: Implemented streaming preprocessing with generators.

**Scenario 2: Low Accuracy**

> Model achieved 45% accuracy initially (should be 85%). Found bug: was using test set for training! Fixed data split, accuracy jumped to 85%.

**Scenario 3: Slow Training**

> Epoch taking 5 minutes. Profiled and found preprocessing was bottleneck. Solution: Pre-compute and cache all frames to disk, reduced to 1 min/epoch.

**Structure for answer:**

1. Problem observed
2. Hypothesis
3. How you debugged (print statements, profiler, etc.)
4. Root cause identified
5. Solution implemented
6. Verification

---

## Quick Reference

### Common Interview Answers

**"Why this architecture?"**

> MobileNetV2 is lightweight (9MB), pre-trained on ImageNet, and perfect for free Colab. LSTM learns temporal patterns that a single-frame CNN can't. Together they're fast, accurate, and deployable.

**"How do you prevent overfitting?"**

> Dropout, L2 regularization, early stopping, transfer learning, and balanced dataset. Achieved 85% validation accuracy (same as training), so minimal overfitting.

**"What's your biggest learnings?"**

> Transfer learning is powerful - using pre-trained ImageNet weights saved me weeks of training. GPU memory optimization was critical for Colab. Proper data preprocessing is 50% of the work.

**"What would you change?"**

> Add data augmentation for 2-3% accuracy gain, fine-tune more CNN layers for better features, and use ensemble methods for production deployment.

---

## Practice Questions to Explore

1. How do you choose hyperparameters? (learning rate, batch size, etc.)
2. What's the difference between validation and test set?
3. How do you handle class imbalance?
4. What's the difference between dropout and batch normalization?
5. How does attention mechanism work?
6. What's the difference between CNN and RNN?
7. How do you handle very long videos?
8. How would you detect multiple people doing activities?
9. What's the computational cost of your model?
10. How would you make this work on a mobile phone?

---

## Final Tips

✅ **Do:**

- Be specific with numbers (85%, 50MB, 0.5s)
- Explain trade-offs (accuracy vs speed)
- Show problem-solving approach
- Discuss why you chose design decisions
- Mention production considerations

❌ **Don't:**

- Say "I used X because it's popular"
- Forget about deployment
- Assume all architectural decisions are obvious
- Overlook constraints (GPU memory, latency)
- Stop learning at 85% accuracy

---

## Final Quote

> The best projects aren't just about achieving high accuracy - they're about understanding the trade-offs, optimizing for constraints, and building something deployable.

Good luck in your interviews! 🚀
