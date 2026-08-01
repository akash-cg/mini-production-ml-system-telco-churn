from src.retrain import should_retrain
def test_drift_triggers_retraining():
    decision,signals=should_retrain(5,0.82,0.4)
    assert decision and signals['material_drift']
