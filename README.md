# Predictive Maintenance of Industrial Machines Using Machine Learning

This beginner-friendly internship defence project connects Mechatronics Engineering and Data Science. It uses machine sensor and operating data to classify each observation as **normal operation** or **machine failure**.

## Why this is a Mechatronics project

Mechatronic systems combine mechanical components, electronics, sensors, control and software. Sensors measure quantities such as temperature, speed, torque and tool wear. Machine learning then finds patterns in those measurements that may indicate a developing fault. The prediction can help engineers inspect equipment before an unexpected shutdown.

## Dataset

The project uses the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset). It contains 10,000 synthetic observations designed to resemble industrial predictive-maintenance data.

| Column | Meaning | Used by model? |
|---|---|---:|
| `UDI` | Row identifier | No |
| `Product ID` | Product identifier | No |
| `Type` | Low, medium or high product-quality type | Yes |
| `Air temperature [K]` | Surrounding air temperature | Yes |
| `Process temperature [K]` | Machine process temperature | Yes |
| `Rotational speed [rpm]` | Shaft/tool speed | Yes |
| `Torque [Nm]` | Turning force | Yes |
| `Tool wear [min]` | Accumulated tool-use time | Yes |
| `Machine failure` | Target: 0 normal, 1 failure | Target |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Individual failure-mode labels | No—target leakage |

The failure-mode columns are excluded because they directly determine `Machine failure`. Giving them to the model would be like giving it the answer during an examination.

## Method

1. Inspect shape, columns, types, missing values, duplicates and class balance.
2. Keep six genuine operating inputs and the binary target.
3. One-hot encode the categorical product type.
4. Make an 80/20 stratified train/test split.
5. Train a shallow Decision Tree with balanced class weights.
6. Evaluate unseen test observations and run stratified five-fold cross-validation.

A Decision Tree was selected because its rules are visual and beginner-friendly. `max_depth=5` limits complexity, `min_samples_leaf=10` avoids rules based on very few observations, and `class_weight="balanced"` makes rare failures matter more during training.

## Evaluation terms

- **Accuracy:** fraction of all predictions that were correct.
- **Sensitivity/recall:** `TP / (TP + FN)`—fraction of real failures detected.
- **Specificity:** `TN / (TN + FP)`—fraction of healthy cases correctly recognised.
- **Precision:** `TP / (TP + FP)`—fraction of failure warnings that were correct.
- **F1-score:** balance between precision and recall.
- **Cross-validation:** repeats training/testing across five folds to check whether performance is consistent.

Sensitivity is safety-focused: a false negative can leave a failing machine in operation. Specificity is cost-focused: a false positive can cause an unnecessary inspection or shutdown. Accuracy alone can be misleading because 9,661 of the 10,000 observations are normal.

## Verified results

The fixed 80/20 test split produced:

| Metric | Result |
|---|---:|
| Accuracy | 92.90% |
| Sensitivity/recall | 88.24% |
| Specificity | 93.06% |
| Precision | 30.93% |
| F1-score | 45.80% |

The confusion matrix contained 1,798 true negatives, 134 false positives, 8 false negatives and 60 true positives. Five-fold cross-validation gave mean sensitivity of approximately 90.0%, showing similar failure detection across different folds.

The accuracy is lower than an “always normal” guess, but that guess would have 0% sensitivity and miss every failure. Balanced class weights deliberately trade some false alarms for much better failure detection. Low precision means that a warning should trigger inspection, not be treated as proof of failure.

## Run the project

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python src/predictive_maintenance.py
jupyter notebook notebooks/predictive_maintenance_analysis.ipynb
```

The script saves the trained pipeline and metrics under `models/`, and all charts under `images/`.

## Project structure

```text
predictive-maintenance-ml/
├── data/ai4i2020.csv
├── images/
├── models/
├── notebooks/predictive_maintenance_analysis.ipynb
├── src/predictive_maintenance.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Important limitation

The dataset is synthetic, so this is an educational proof of concept—not a production safety system. A real deployment would require physical sensor data collected from the actual machine, expert validation, monitoring and scheduled retraining.

## Defence questions and simple answers

**1. What problem are you solving?**  
I am using sensor and operating data to predict whether a machine is likely to be in a failure condition.

**2. Why is this classification?**  
The output belongs to one of two classes: normal operation or machine failure.

**3. Why did you choose a Decision Tree?**  
It can learn nonlinear rules and is easy to visualise and explain. I restricted its depth to reduce overfitting.

**4. Why remove UDI and Product ID?**  
They identify records or products but do not represent the machine's physical condition.

**5. Why remove the five failure-mode columns?**  
They directly reveal why the target is one. Using them would cause target leakage and unrealistic performance.

**6. Why use a stratified split?**  
Failures are rare. Stratification preserves approximately the same failure proportion in training and testing data.

**7. Why is accuracy insufficient?**  
A model predicting every observation as normal would still have about 96.6% accuracy while detecting no failures.

**8. What is sensitivity?**  
It is the proportion of actual failures detected. High sensitivity reduces missed failures.

**9. What is specificity?**  
It is the proportion of normal operations correctly recognised. High specificity reduces false alarms.

**10. What is a false negative here?**  
The machine is failing, but the model says it is normal. This is potentially dangerous and costly.

**11. What is a false positive here?**  
The machine is healthy, but the model raises a warning. This can cause unnecessary maintenance or downtime.

**12. What does cross-validation do?**  
It tests the approach on several different portions of the data, showing whether the result is stable rather than dependent on one split.

**13. Does correlation prove that a variable causes failure?**  
No. Correlation shows association, not causation.

**14. How could this be deployed on a real machine?**  
Sensors could send current readings to a computer or controller. The saved model would process the readings and send a warning to a maintenance dashboard.

**15. What would you improve with more time?**  
I would collect real machine data, compare a few suitable models, tune the warning threshold and test the system with maintenance engineers.
