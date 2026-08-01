def should_retrain(new_days:int, recent_auc:float|None, drift_score:float, baseline_auc:float=0.80):
    """Example retraining trigger logic; scheduling is intentionally out of scope."""
    signals={
        'enough_new_data': new_days >= 30,
        'performance_drop': recent_auc is not None and recent_auc < baseline_auc - 0.03,
        'material_drift': drift_score > 0.25,
    }
    return any(signals.values()), signals
