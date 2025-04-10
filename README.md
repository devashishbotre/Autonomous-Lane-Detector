# Autonomous-Lane-Detector
This project implements an end-to-end Autonomous Lane Detection pipeline using deep learning techniques on the TuSimple dataset. The model architecture is based on a RESA-based (Recurrent Feature Shift Aggregator) design, incorporating a ResNet encoder, RESA module, and custom decoder to accurately detect and predict lane lines
## Overview
The goal of this project is to enable self-driving cars to recognize lane markings on the road using computer vision and machine learning. The model uses a ResNet encoder, a RESA module for feature aggregation, and a custom decoder to predict lane line positions from road images.
## Features
- **ResNet-34 Backbone**: Utilizes a pre-trained ResNet-34 encoder of 4 layers for robust feature extraction.
- **RESA Module**: Incorporates Recurrent Feature-Shift Aggregator (RESA) with 4 iterations for enhanced spatial context in lane segmentation.
- **Custom Decoder**: Employs a two-stage upsampling decoder with thinning and smoothing layers for precise lane detection.
- **Custom Loss Functions**: Combines cross-entropy loss with total variation, narrowness penalty, and edge enhancement losses for robust training.
- **Evaluation Metrics**: Computes F1 score, accuracy, and inference time on the training dataset.
- **Pre-trained Weights**: Supports loading pre-trained weights for immediate evaluation.
## Dataset Information
- **Dataset**: [TuSimple Lane Detection Dataset](https://github.com/TuSimple/tusimple-benchmark)
- **Description**: Contains 3,626 training images with lane annotations from highway driving scenarios, split across three JSON files: `label_data_0313.json`, `label_data_0531.json`, and `label_data_0601.json`.
- **Format**: Each annotation includes image paths (`raw_file`), lane coordinates (`lanes`), and vertical sample points (`h_samples`).
- **Usage**: The full training set is used for both training and evaluation in this implementation.
## Architecture
The model architecture leverages ResNet-34 with additional modules for lane detection:
- **Encoder**: ResNet-34
  - Pre-trained on ImageNet, includes layers from `conv1` to `layer4`.
  - Final output resolution: 1/32 of input, i.e., (45×80) with 512 channels.
- **RESA**:
  - 4 Stacked RESA modules
  - Each takes 512 channels and aggregates spatial features via directional shifts and depthwise refinement.
- **Decoder**: Two-stage upsampling
  - **Up1**: Conv2d (512 → 256), BatchNorm, ReLU, Upsample (scale factor 2) -> (90x160).
  - **Up2**: Conv2d (256 → 256), BatchNorm, ReLU, Upsample (scale factor 2) -> (180x320).
  - **Up3**: Conv2d (256 → 256), BatchNorm, ReLU, Upsample (scale factor 2) -> (360x640).
  - **Up4**: Conv2d (256 → 256), BatchNorm, ReLU, Upsample (scale factor 2) -> (720x1280).
  - **Thinning Layers**: Depthwise convolutions for refinement (256 channels).
  - **Smoothing**: Depthwise 5x5 convolution (256 channels).
- **Segmentation Head**: 1x1 convolution to output 2 classes (background and lane, 256 → 2 channels).
- **Post-processing**: Average pooling (3x3 kernel) for smoother outputs.
- **Output**: Binary segmentation mask at 720x1280 resolution.
## Results
1. **Evaluation Score**
     ```markdown
     | Metric                | Value    |
     |-----------------------|----------|
     | Average F1 Score      | 0.56     |
     | Average Accuracy      | 0.99     |
     | Inference Time (s/img)| 0.005432 | 
## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/lane-detection-resnet.git
   cd lane-detection-resnet
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
## Future Work
- Improving F1 score and smoother lane plotting
- Add support for CULane and BDD100K datasets
- Integrating obstacle detection with speed and time evaluation
## References
- [ResNet](https://arxiv.org/pdf/1512.03385v1)
- [RESA](https://arxiv.org/pdf/2008.13719) 
