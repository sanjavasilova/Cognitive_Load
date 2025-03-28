# Cognitive Effort - Research Process and Implementation

## Overview
This document provides a detailed overview of my research on cognitive effort, including the methodology, tools, data collection, findings, and challenges encountered during the study.

## Contents
1. Introduction
2. Research Motivation and Goals
3. Methodology
4. Tools and Technologies Used
5. Data Collection and Processing
6. Key Findings and Observations
7. Challenges and Lessons Learned
8. Future Research Directions

## 1. Introduction
Cognitive effort is a crucial aspect of human performance, affecting decision-making, learning, and productivity. This research aimed to explore how physiological signals can be used to measure cognitive effort under different cognitive load conditions.

## 2. Research Motivation and Goals
The study was conducted to:
- **Understand the relationship** between physiological signals and cognitive load.
- **Develop a framework** for cognitive load classification.
- **Evaluate different machine learning models** for predicting cognitive effort.

## 3. Methodology
The study involved measuring physiological signals while participants performed tasks with varying cognitive loads (LOW_CL and HIGH_CL). The methodology included:
- **Data Collection:** Using wearable devices (Empatica E4 and Samsung) to record physiological signals.
- **Signal Processing:** Extracting features from blood volume pulse (BVP), electrodermal activity (EDA), and body temperature.
- **Preprocessing:** Removing noise and artifacts, normalizing data using Min-Max and Z-Score methods.
- **Machine Learning Models:** Training and evaluating models such as Support Vector Machines (SVM), Random Forest, Logistic Regression, and Convolutional Neural Networks (CNN).

## 4. Tools and Technologies Used
The study utilized the following tools:
- **Wearable Devices:** Empatica E4 and Samsung devices for physiological signal acquisition.
- **Data Processing Software:** Used for signal filtering, feature extraction, and normalization.
- **Machine Learning Frameworks:** Implemented models using Python-based libraries.

## 5. Data Collection and Processing
Data was collected from participants exposed to different cognitive load levels, with the following steps:
- **Feature Extraction:** Analyzing heart rate variability (HRV) from BVP, sympathetic nervous system activity from EDA, and body thermoregulation.
- **Preprocessing Techniques:** Removing noise, normalizing data distributions, and applying balancing techniques.

## 6. Key Findings and Observations
- **Samsung devices** provided more informative data for cognitive load classification than Empatica.
- **The choice of normalization technique** and model complexity significantly impacted classification accuracy.
- **CNN models** showed promising results, particularly those with 64 or 128 layers.

## 7. Challenges and Lessons Learned
- **Noise and Artifacts:** Required extensive preprocessing to ensure data quality.
- **Data Imbalance:** Techniques were needed to balance datasets and improve model generalization.
- **Feature Selection:** Determining the most relevant physiological features was critical for improving performance.

## 8. Future Research Directions
Future work can focus on:
- **Optimizing Model Architectures:** Exploring deeper neural networks and hybrid models.
- **Real-Time Cognitive Load Monitoring:** Implementing a real-time framework for cognitive effort assessment.
- **Expanding Data Collection:** Testing with a broader participant base to improve model generalization.

For a detailed breakdown of the research, you can read the full documentation in **Cognitive Effort.docx**.
