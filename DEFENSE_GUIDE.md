# Internship Defence Guide

## Two-minute explanation

My project is titled **Predictive Maintenance of Industrial Machines Using Machine Learning**. In conventional maintenance, a machine may be serviced on a fixed schedule or only after it breaks down. Predictive maintenance uses its present condition to support an earlier maintenance decision.

I used the AI4I 2020 dataset. Its inputs represent measurements that can come from sensors and machine controllers: air and process temperature, rotational speed, torque, tool wear and product type. The target tells us whether a failure occurred.

After inspecting the data, I found no missing values or duplicate rows, but failures were much rarer than normal observations. I removed identifiers because they do not describe physical condition. I also removed the individual failure-mode labels because they directly reveal the final target and would cause data leakage.

I divided the data into 80% training and 20% testing using stratification. I trained a small Decision Tree because it is easy to interpret and can capture relationships between several machine variables. I used balanced class weights so that the rare failure cases received more attention.

I evaluated the model with a confusion matrix, accuracy, sensitivity, specificity, precision, F1-score and five-fold cross-validation. Sensitivity is especially important because it shows how many actual failures were detected. Specificity matters because too many false alarms lead to unnecessary maintenance and downtime.

This connects to Mechatronics because sensors and controllers produce information about a physical machine, while Data Science converts those readings into useful maintenance decisions.

## Recommended demonstration order

1. Show the dataset columns and relate each measurement to a machine.
2. Show the class-distribution graph and explain why accuracy alone is unsafe.
3. Show the torque/speed and tool-wear plots.
4. Explain the train/test split and Decision Tree settings.
5. Show the confusion matrix and calculate sensitivity and specificity from it.
6. Show cross-validation results.
7. Run the example reading and display `Normal` or `Failure risk`.
8. End with the synthetic-data limitation and real deployment idea.

## Formula reminder

- Sensitivity = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- Precision = TP / (TP + FP)
- Accuracy = (TP + TN) / Total
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

## Understanding this model's result

On the test set, the model detected 60 of 68 failures and missed 8. It also raised 134 false alarms among 1,932 normal observations. This gives 88.2% sensitivity and 93.1% specificity. Precision is only 30.9% because true failures are very rare. Therefore, the model's warning should request an engineering inspection rather than automatically shutting down the machine.

If asked why 92.9% accuracy is acceptable when predicting everything as normal gives about 96.6%, explain that the always-normal model detects zero failures. Predictive maintenance needs a useful balance between missed failures and false alarms, not accuracy alone.

## One-sentence conclusion

The project shows that operating measurements from a mechatronic system can be analysed with machine learning to support earlier and better-informed maintenance decisions.
