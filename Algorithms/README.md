Anomaly Detection Algorithm Component

Project: IoT Sensor Data Anomaly Detection in Algorithm Implementation and Evaluation

Prepared by: Mongkonkrit Aumpaisiriwong

Role: Algorithm Specialist Role

This directory contains the code artifact for the Algorithm Implementation and Evaluation component (Task Requirement 2). This script performs the end-to-end process of data loading, feature scaling, model training, benchmarking, and anomaly visualisation, ultimately selecting the most effective algorithm for the real-time IoT streaming environment.

1. Key Contribution

The main output of this component is the comparative analysis of three anomaly detection models:

Isolation Forest (IF)

Local Outlier Factor (LOF)

PyTorch Autoencoder (AE)

The Autoencoder was empirically selected for the final system architecture due to its superior Inference Throughput (over 5 million records/second), which is critical for handling high-velocity IoT sensor data streams.

2. Prerequisites

To execute the main script, you must have the following file available in the same directory:

SGSC_Weather_Sensor_Data.csv: The publicly available IoT dataset used for training and benchmarking. (Shape: ~742,141 records, 8 features after preprocessing).

A supporting file for the Autoencoder model class: The notebook imports ModelBuilder.py. Please ensure this file containing the Autoencoder and other necessary classes is also provided in the repository.

3. Setup and Execution

Install Dependencies: Ensure all required Python packages are installed using the provided requirements.txt file:

pip install requirement.txt



Run the Analysis: Open and run all cells in the Jupyter Notebook:

jupyter notebook Algorithm.ipynb


4. Outputs

Upon successful execution, the notebook will generate the following:

Console Output: Prints the comprehensive benchmarking results (latency and throughput) used in the Report's Table I.

Visualisations: Displays comparative 2D anomaly maps (PCA view) for all three models.

Result File: A new file is saved: SGSC_Weather_Anomaly_Results.csv containing the original data appended with the anomaly scores and binary classification columns from all three models.



5. References

scikit-learn. "Outlier detection with Local Outlier Factor (LOF)." https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html (accessed 14 Nov, 2025).

scikit-learn. "IsolationForest." https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html (accessed 14 Nov, 2025).

geeksforgeeks. "Autoencoders in Machine Learning." https://www.geeksforgeeks.org/machine-learning/auto-encoders/ (accessed 14 Nov, 2025).