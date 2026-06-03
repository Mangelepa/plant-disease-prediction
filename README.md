# Plant Disease Detection using Deep Learning

A deep learning project system for automatically detecting and classifying plant diseases from leaf images using a Custom CNN and MobileNetV2 transfer learning.

## Project Overview

Plant diseases are one of the most significant threats to global food security. Early and accurate identification is critical, but traditional diagnosis methods relying on manual inspection or expert consultation are slow, expensive, and inaccessible at scale.

This project builds an image classification system that can automatically detect plant diseases from photographs of plant leaves, enabling fast, scalable, and cost-effective diagnosis for farmers and agricultural professionals.

Two deep learning models were trained and compared:
- A **Custom Convolutional Neural Network (CNN)** built from scratch
- A **MobileNetV2 Transfer Learning model** pre-trained on ImageNet

## Business Problem

 **Challenge**

 Delayed disease detection | Significant crop yield loss |
 Reliance on scarce expert knowledge | High consultation costs |
 Manual inspection at scale | Not feasible for large farms |
 Lack of accessible diagnostic tools | Farmers in rural areas go unsupported |

**Goal:** 
Deliver a trained model accurate enough for deployment in a mobile or web application where users upload a leaf photo and receive an instant disease diagnosis.

## Dataset

The dataset consists of labeled plant leaf images organized into **38 disease/healthy classes** across multiple plant species.

**Key dataset properties:**
- Images resized to **128×128 pixels**
- **RGB color mode**
- Labels inferred from folder names
- 38 classes (disease categories + healthy variants)
- Dataset is **well-balanced** — no severe class skew, no need for oversampling or class weighting
- **0 corrupted images** found across all splits

## Models
### Model 1 — Custom CNN
A sequential CNN built from scratch with three convolutional blocks:

| Layer | Details |
|---|---|
| Data Augmentation | Flip, Rotate, Zoom |
| Rescaling | Normalize pixel values (÷ 225) |
| Conv Block 1 | Conv2D(32) → MaxPool → BatchNorm |
| Conv Block 2 | Conv2D(64) → MaxPool → BatchNorm |
| Conv Block 3 | Conv2D(128) → MaxPool → BatchNorm |
| Head | GlobalAvgPool → Dense(256) → Dropout(0.5) → Dense(38, softmax) |

- **Optimizer:** Adam (lr=1e-4)
- **Loss:** Categorical Crossentropy

### Model 2 — MobileNetV2 (Transfer Learning) ⭐ Final Model

Leverages MobileNetV2 pre-trained on ImageNet as a frozen feature extractor, with a custom classification head added on top.

| Component | Details |
|---|---|
| Base Model | MobileNetV2 (ImageNet weights, frozen) |
| Input Preprocessing | MobileNetV2-specific preprocessing |
| Pooling | GlobalAveragePooling2D |
| Regularization | Dropout(0.3) |
| Output | Dense(38, softmax) |

- **Optimizer:** Adam (lr=1e-4)
- **Loss:** Categorical Crossentropy

---

## 📊 Results

| Metric | Custom CNN | MobileNetV2 |
|---|---|---|
| Training Accuracy | 89.8% | 94.4% |
| Validation Accuracy | 69.4% | 95.4% |
| Validation Loss | Higher | 0.152 |
| Generalization |Overfitting signs | Strong |

**MobileNetV2 was selected as the final model** due to:
- Higher validation accuracy (95.4% vs 69.4%)
- Lower validation loss
- Better generalization — the gap between training and validation accuracy is minimal, indicating no overfitting
- Lightweight architecture suitable for deployment on mobile/edge devices

### `requirements.txt`

```
tensorflow>=2.10.0
numpy
pandas
matplotlib
scikit-learn
Pillow
```

## Stakeholders

**Stakeholder Benefit** 
- Farmers- Quick, low-cost field diagnosis via mobile app 
- Agricultural researchers - Scalable crop health monitoring 
- Agronomists - AI-powered diagnostic support tool 
- App developers - Ready-to-deploy `.keras` model for integration 


## Future Work

**Fine-tune MobileNetV2** — unfreeze top layers for domain-specific fine tuning to push accuracy further
**Expand dataset** — include more plant species and rare disease classes
**Deploy as a web app** — Deploy the model for real time usage
**Multi-label classification** — detect multiple diseases present in a single leaf image
