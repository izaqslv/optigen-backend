from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.v2_legacy.predict import predict_concentration

measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")

df_pred = predict_concentration(measurements, fluids_meta)

print(df_pred.head())