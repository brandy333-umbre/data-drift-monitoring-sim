# data-drift-monitoring-sim
A lightweight simulator for exploring how machine-learning models degrade under data drift, with time-series monitoring of accuracy, F1, and log-loss.
This project is designed to simulate a real world problem in how an ML models performance changes over time when the data distribution shifts

What this project does:

The simulator:
- Trains a baseline classification model on stable data
- Gradually introduces data drift over multiple time steps
- Evaluates the model at each step
- Logs performance metrics to track degradation over time

The three types of drift supported by this project:
Covariate Drift
Feature distributions shift over time (e.g. sensors recalibrated, population behaviour changes), while labels remain defined the same way.

Label Shift
Class prevalence changes over time (e.g. disease incidence increases), even if feature distributions stay similar.

Combined Drift
Both feature distributions and label frequencies change simultaneously.

Model setup:
The model is trained once on the original (non-drifted) dataset and then kept fixed.

Supported models:
- Logistic Regression (default)
- Random Forest

This mirrors production systems where retraining is not continuous and performance must be monitored between updates.

How drift is evaluated:
At each simulated time step, the system computes:
- Accuracy
- Macro F1 score
- Log-loss

Metrics are:
- Printed to the console
- Saved as a CSV file for analysis or plotting

A simple example alert is included to demonstrate how monitoring thresholds could be triggered when performance drops.
