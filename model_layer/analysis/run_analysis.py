import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from model_layer.analysis.data_loader import load_and_align_data
from model_layer.analysis.dataset_builder import build_dataset
import joblib
from model_layer.models.training.train_concentration_model import run_training

path_excel = "data/DadosSedimentation.xlsx"
measurements, fluids_meta = load_and_align_data(path_excel)

df = build_dataset(measurements, fluids_meta)

print("📦 Dados carregados")
print(f"Shape:{df.shape}")
print(f"Colunas:{list(df.columns)}")

# features = joblib.load("artifacts/features/concentration_model_features.pkl")

# TREINAMENTO DO MODELO DE CONCENTRAÇÃO
print("\n TREINANDO MODELO DE CONCENTRAÇÃO...\n")
model, df_imp, metrics = run_training(df)

# validação visual
df["pred"] = model.predict(df[features])
for f, g in df.groupby("fluid_id"):
    perfil_real = (
        g.groupby("altura")["concentracao"]
        .mean()
        .sort_index()
    )

    perfil_pred = (
        g.groupby("altura")["pred"]
        .mean()
        .sort_index()
    )

    plt.figure()

    plt.plot(perfil_real.values, perfil_real.index, label="Real")
    plt.plot(perfil_pred.values, perfil_pred.index, "--", label="Predito")
    plt.xlabel("Concentração")
    plt.ylabel("Altura")
    plt.title(f"Perfil Vertical - Fluido {f}")
    plt.legend()

    plt.show()
    features = joblib.load("artifacts/features/concentration_model_features.pkl")









### ABAIXO: CÓDIGO ANTERIR DE TESTES:


# print(df.head(10))   # primeiras 20 linhas
# print(df.tail(10))   # últimas 20
# print(df.groupby("fluid_id")["tempo"].max())
#
# print(df[["fluid_id", "altura", "tempo", "concentracao", "dC_dt"]].head(50))
#
# # VALIDAÇÃO 1 - DERIVADA (dC_dt): o que esperar --> (Nos pontos de queda, dC_dt deve ser NEGATIVO frte, magnitude grande
# # ex: t=45 ->dC_dt = -0.02 --> sinalizando queda; te=46 -> dC_dt=-0.03 --> sinalizando queda)
# # df_test = df[df["fluid_id"].isin([5, 6])]
# df_test = df.copy()
#
# print(df_test[[
#     "fluid_id", "altura", "tempo",
#     "concentracao", "dC_dt"
# ]].head(50))
#
# # VALIDAÇÃO 2 - querda brusca: o que esperar --> (deve aparecer: no fluidos 9 --> alturas 23, 20;
# # no fluido 10 --> alturas 23, 20, 18. E nos tempos corretos, que vimos nos gráficos)
# print(df_test[df_test["queda_brusca"] == True][[
#     "fluid_id", "altura", "tempo", "dC_dt"
# ]])
#
# # VALIDAÇÃO 3 - tempo de colapso: Esperado --> no fluido 9 (em h=23 --> t_c=~40-50, em h=18 --> t_c=~65), no fluido 10
# # (em h=23 --> t_c=~45, em h=18 --> t_c=~50-60)
# print(df_test[[
#     "fluid_id", "altura", "t_colapso"
# ]].drop_duplicates())
#
#
# # ## VALIDAR GRADIENTE DE ALTURA (GLOBAL POR FLUIDO)
# print("\n=== GRADIENTE DE ALTURA ===")
# print(df[["fluid_id", "gradiente_altura"]].drop_duplicates())
#
# # VALIDAR GRADIENTE LOCAL
# print("\n=== GRADIENTE LOCAL ===")
# print(df[["fluid_id", "altura", "gradiente_local"]].drop_duplicates().head(20))
#
# # VALIDAR PERFIL VERTICAL
# perfil = df.groupby(["fluid_id", "altura"])["concentracao"].mean().reset_index()
# print("\n=== PERFIL VERTICAL ===")
# print(perfil.head(20))
#
# # VALIDAR A DETECÇÃO DA INTERFACE DE SEDIMENTAÇÃO
# # df_interface = compute_interface_sedimentacao(df)
# print("\n=== INTERFACE DE SEDIMENTAÇÃO ===")
# print(df.groupby("fluid_id")["altura_interface"].first())
#
# # VALIDAÇÃO 4 - visual (essencial): sobrepor colapso no gráfico (o que edvemos ver é linha vermelha exatamente onde
# # ocorre a queda. Se não bater, o problema pode estar no threshold ou na derivada)
# for (f, h), g in df_test.groupby(["fluid_id", "altura"]):
#
#     plt.figure()
#     plt.plot(g["tempo"], g["concentracao"], label="Concentração")
#
#     # marcar colapso
#     t_col = g["t_colapso"].iloc[0]
#
#     # marcar interface de sedimentação
#     h_int = g["altura_interface"].iloc[0]
#     plt.text(
#         x=g["tempo"].max() * 0.6,
#         y=g["concentracao"].max() * 0.9,
#         s=f"Interface em h (cm) ≈ {h_int:.1f}",
#         fontsize=9,
#         color="green"
#     )
#     # fim da etapa de interface
#
#     if pd.notnull(t_col):
#         plt.axvline(x=t_col, color='r', linestyle='--', label='Colapso')
#
#     plt.title(f"Fluido {f} - Altura {h}")
#     plt.legend()
#
# plt.show(block=True) # para mostrar gráficos em sequência sem parar
#
#
#
# #------------------------------------------------------------------------------
# # Gráficos da interface de cada fluido: PERFIL VERTICAL MÉDIO
# df_interface = compute_interface_sedimentacao(df)
# print(df_interface.head())
# for f, g in df.groupby("fluid_id"):
#
#     # perfil vertical médio
#     perfil = (
#         g.groupby("altura")["concentracao"]
#         .mean()
#         .sort_index()
#     )
#
#     # pegar interface
#     h_int = df_interface[df_interface["fluid_id"] == f]["altura_interface"].iloc[0]
#
#     # plot
#     plt.figure()
#
#     plt.plot(perfil.values, perfil.index, label="Perfil")
#
#     # linha da interface
#     if pd.notnull(h_int):
#         plt.axhline(y=h_int, color='g', linestyle='--', label='Interface')
#
#     plt.xlabel("Concentração")
#     plt.ylabel("Altura")
#     plt.title(f"Perfil Vertical - Fluido {f}")
#     plt.legend()
#
# plt.show(block=True)
# # plt.show()



# ## TREINAMENTO DO MODELO PARA PREVER ALTURA DE INTERFACE
# from model_layer.models.training.train_interface_model import run_training
#
# model, df_imp, metrics = run_training(df)
#
# print("\n🚀 MODELO TREINADO COM SUCESSO!")
# print("📊 Métricas finais:")
# print(metrics)
# print("\n🧠 Feature Importance:")
# print(df_imp.head)





# print(df.head())
# print(df.columns)

# df = load_and_align_data("data/DadosSedimentation.xlsx")
# print(df.columns)
# print(df.groupby(["fluid_id", "altura"]).size())
# for (f, h), g in df.groupby(["fluid_id","altura"]):
#     print(f"\nFluido {f} | altura {h}")
#     print("min:", g["tempo"].min(), "max:", g["tempo"].max(), "len:", len(g))





# # ================================
# # 1. Carregar dados
# # ================================
# # df = load_and_merge_data("model_layer/data/raw/DadosSedimentation.xlsx")
# # DEBUG INICIAL (COLOCA AQUI)
# print("\n=== HEAD DO DATASET ===")
# print(df.head())
#
# print("\n=== COLUNAS ===")
# print(df.columns)
#
# print("\n=== TAMANHO POR GRUPO ===")
# print(df.groupby(["fluid_id", "altura"]).size())
# print()
#
# for fluid in df["fluid_id"].unique():
#     df_f = df[df["fluid_id"] == fluid]
#
#     print(f"\nFluido {fluid}")
#     print(sorted(df_f["tempo"].unique())[:10])
#     print("...")
#     print(sorted(df_f["tempo"].unique())[-10:])
#
# # ================================
# # 2. Rodar modelo
# # ================================
# df_pred = predict(df)
#
# # ================================
# # 3. Relatório físico (NOVO)
# # ================================
# report = generate_report(df_pred)
#
# print(report["text"])  # 👈 importante (não print(report))
#
# # ================================
# # 4. Plot
# # ================================
# # plot com modelo
# plot_by_fluid(df_pred)
# # plots experimentais only
# plot_experimental_only(df)  # desativar se desejar
#
# # ================================
# # 5. Debug (opcional - manter)
# # ================================
# print("\n=== VERIFICAÇÃO DE PONTOS ===")
#
# for fluid in df["fluid_id"].unique():
#     print(f"\nFluido {fluid}")
#
#     df_f = df[df["fluid_id"] == fluid]
#
#     for h in sorted(df_f["altura"].unique()):
#         df_h = df_f[df_f["altura"] == h]
#         print(f"Altura {h}: {len(df_h)} pontos")
#
#
# # Add em 10/04/2026
# plt.show(block=True)
