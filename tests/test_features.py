import pandas as pd
from src.features import build_features
def test_feature_engineering_columns():
    row=pd.DataFrame([{'tenure':2,'MonthlyCharges':50.,'TotalCharges':100.,'PhoneService':'Yes','MultipleLines':'No','OnlineSecurity':'Yes','OnlineBackup':'No','DeviceProtection':'No','TechSupport':'Yes','StreamingTV':'No','StreamingMovies':'No','PaymentMethod':'Credit card (automatic)','Contract':'Month-to-month'}])
    out=build_features(row)
    assert out.loc[0,'service_count']==3
    assert out.loc[0,'automatic_payment']==1
    assert out.loc[0,'avg_charge_per_month']==50
