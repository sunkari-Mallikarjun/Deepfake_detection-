# DeepFake-Detection


https://github.com/user-attachments/assets/868e3129-275f-4394-97f3-2476f6965e97



This project is a Flask-based social media application that allows users to manage their profiles, reset passwords, and upload profile pictures. A key highlight of the project is its advanced deepfake detection system integrated with both image and video content.

Deepfake Detection System Overview:
The application incorporates a multi-model deepfake detection pipeline that uses three state-of-the-art models:

MesoNet
XceptionNet
Vision Transformer (ViT)
Image Processing & Classification:
For detecting deepfakes in images:

MesoNet and XceptionNet are convolutional neural networks (CNNs) specifically designed for deepfake detection. MesoNet focuses on analyzing facial textures, while XceptionNet is more robust for detecting subtle facial manipulations.
ViT (Vision Transformer) brings in a modern approach by using transformer architecture to model long-range dependencies in the image, offering enhanced accuracy for image analysis.
The deepfake detection pipeline works by:

Preprocessing the image for each of the models.
Predicting the likelihood of the image being fake or real using each of the models.
Combining the outputs of all three models using a dynamic weighted average approach, which intelligently adjusts the influence of each model based on the confidence of their predictions, ensuring more accurate and reliable results.
Video Processing & Frame-by-Frame Classification:
For video-based deepfake detection:

The system extracts frames from the video at specified intervals.
Each frame is processed through the same multi-model classification pipeline used for images.
Results from each frame are aggregated using either:
Majority Voting: The label (real or fake) that appears most frequently across frames is chosen as the final prediction.
Dynamic Confidence: A weighted confidence approach is applied to calculate the probability that the video is real or fake based on the confidence scores of the models across all frames.
Model Outputs & Predictions:
MesoNet and XceptionNet: Outputs are numeric probabilities that indicate whether the content is real or fake. These models excel in spotting subtle pixel-level anomalies often found in deepfakes.
Vision Transformer: Provides a global understanding of the image by analyzing its entire structure, complementing the pixel-level scrutiny of the CNN models.
Why This Matters:
Deepfake detection is becoming increasingly important in the context of social media, where fake news, misinformation, and manipulated media can spread rapidly. This application’s ability to detect deepfakes in both images and videos provides a robust solution for content moderation and user security in a digital world where media authenticity is crucial.

By combining three different types of models and using a sophisticated voting and confidence aggregation mechanism, this project offers a cutting-edge approach to identifying deepfakes with high accuracy, making it a valuable tool for the detection of media manipulation.
