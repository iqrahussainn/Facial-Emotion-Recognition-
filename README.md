# Facial Emotion Recognition — Model Comparison

This project compares multiple deep-learning architectures for **Facial Emotion Recognition (FER)** using the FER-2013 dataset. FER is challenging due to subtle facial differences and dataset class imbalance .

---

## Dataset

* **Dataset:** FER-2013
* **Image size:** 48×48 (grayscale; converted to 3-channel for VGG16)
* **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
* **Train:** ~28,000 | **Test:** ~7,000
* **Challenge:** strong class imbalance (Disgust is under-represented) 

---

## Models

### Simple CNN

* 3 conv blocks → max-pool + dropout
* Dense(128) → Dropout → Dense(7, softmax)
* **Adam**, categorical cross-entropy, **20 epochs**

**Results:** Train ~27% | Test ~36%
→ Limited generalization; weak on minority classes .

---

### Hybrid CNN

* 4 conv blocks with **batch-norm + dropout**
* Dense(256) → Dropout → Dense(7, softmax)

**Results:** Train ~42% | Test ~49%
→ Better accuracy, still affected by imbalance .

---

### VGG16 (Transfer Learning)

* Pretrained VGG16 (ImageNet) as base
* GAP → Dense(512) → Dropout → Dense(256) → Dropout → Dense(7)
* Adam (1e-5), **25 epochs**, early stopping + checkpoints

**Results:** Test ~49%
→ Best features; minority emotions still weak .

---

## Model Comparison

| Model      | Test Accuracy | Strengths                    | Weaknesses                  |
| ---------- | ------------- | ---------------------------- | --------------------------- |
| Simple CNN | ~36%          | Simple, lightweight          | Poor generalization         |
| Hybrid CNN | ~49%          | Stronger accuracy            | Class imbalance limits      |
| VGG16      | ~49%          | Powerful pretrained features | Small gain, minorities weak |

**Conclusion:** Transfer learning and deeper CNNs improve results, but **class imbalance remains the key bottleneck** .

---

## Run in Google Colab

1. Open the notebook in **Google Colab**.
2. (Optional) Mount Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure dataset structure:

```
data/
 ├── train/
 └── test/
```

4. Run notebook cells in order: load data → train models → evaluate → compare.

---

## Future Work

* Oversampling / class-weighted training
* Stronger augmentation
* ResNet / EfficientNet experiments
* Real-time webcam FER in notebook 

---

## Acknowledgement

Completed for **AI335L — Deep Learning Lab (Fall 2025)**, Air University, under **Ms. Bismah Malik** .
