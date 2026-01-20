# Model Performance WITH Leakage

**Warning:** These metrics are inflated due to data leakage!

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| ROC AUC | 1.0000 |
| CV Accuracy (mean) | 1.0000 |
| CV Accuracy (std) | 0.0000 |

## Analysis

These metrics are **unrealistically high** because the model is using features 
that contain information about the target variable:

- `churn_risk_score` - Directly derived from the target
- `cancellation_survey_score` - Only collected after customer churns
- `final_bill_amount` - Calculated based on churn decision

This model would fail catastrophically in production because these features
won't be available at prediction time.
