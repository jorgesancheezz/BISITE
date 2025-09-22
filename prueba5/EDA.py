import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba5\\lung_cancer_dataset.csv"
output_folder = os.path.join(os.path.dirname(__file__), "resultados_frequencies")
os.makedirs(output_folder, exist_ok=True)

from datetime import datetime
def unique_fig_name(base):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_folder, f"{base}_{timestamp}.png")

df = pd.read_csv(data_path)

# Variables categóricas a analizar
categorical_vars = ['gender','copd_diagnosis','alcohol_consumption',
                    'family_history','asbestos_exposure','secondhand_smoke_exposure']


plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, bins=20)
plt.title("Age Distribution")
plt.savefig(unique_fig_name("age_distribution"))

plt.figure(figsize=(6,4))
sns.boxplot(x="lung_cancer", y="age", data=df)
plt.title("Age vs Lung Cancer")
plt.savefig(unique_fig_name("age_vs_lung_cancer"))

plt.figure(figsize=(5,4))
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")
plt.savefig(unique_fig_name("gender_distribution"))

plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="lung_cancer", data=df)
plt.title("Gender vs Lung Cancer")
plt.savefig(unique_fig_name("gender_vs_lung_cancer"))

plt.figure(figsize=(6,4))
sns.histplot(df['pack_years'], kde=True, bins=20)
plt.title("Pack Years (Smoking Intensity)")
plt.savefig(unique_fig_name("pack_years_distribution"))

plt.figure(figsize=(6,4))
sns.boxplot(x="lung_cancer", y="pack_years", data=df)
plt.title("Pack Years vs Lung Cancer")
plt.savefig(unique_fig_name("pack_years_vs_lung_cancer"))

plt.figure(figsize=(6,4))
sns.scatterplot(x="age", y="pack_years", hue="lung_cancer", data=df)
plt.title("Age vs Pack Years (Colored by Lung Cancer)")
plt.savefig(unique_fig_name("age_vs_pack_years"))

plt.figure(figsize=(6,4))
sns.countplot(x="alcohol_consumption", data=df)
plt.title("Alcohol Consumption Distribution")
plt.savefig(unique_fig_name("alcohol_consumption_distribution"))

plt.figure(figsize=(6,4))
sns.countplot(x="alcohol_consumption", hue="lung_cancer", data=df)
plt.title("Alcohol Consumption vs Lung Cancer")
plt.savefig(unique_fig_name("alcohol_consumption_vs_lung_cancer"))

plt.figure(figsize=(6,4))
sns.countplot(x="secondhand_smoke_exposure", data=df)
plt.title("Secondhand Smoke Exposure Distribution")
plt.savefig(os.path.join(output_folder, "secondhand_smoke_exposure_distribution.png"))

plt.figure(figsize=(6,4))
sns.countplot(x="secondhand_smoke_exposure", hue="lung_cancer", data=df)
plt.title("Secondhand Smoke Exposure vs Lung Cancer")
plt.savefig(os.path.join(output_folder, "secondhand_smoke_exposure_vs_lung_cancer.png"))

plt.figure(figsize=(6,4))
sns.countplot(x="radon_exposure", hue="lung_cancer", data=df)
plt.title("Radon Exposure vs Lung Cancer")
plt.savefig(os.path.join(output_folder, "radon_exposure_vs_lung_cancer.png"))

plt.figure(figsize=(6,4))
sns.countplot(x="asbestos_exposure", hue="lung_cancer", data=df)
plt.title("Asbestos Exposure vs Lung Cancer")
plt.savefig(os.path.join(output_folder, "asbestos_exposure_vs_lung_cancer.png"))

plt.figure(figsize=(6,4))
sns.countplot(x="copd_diagnosis", hue="lung_cancer", data=df)
plt.title("COPD Diagnosis vs Lung Cancer")
plt.savefig(os.path.join(output_folder, "copd_diagnosis_vs_lung_cancer.png"))

plt.figure(figsize=(6,4))
sns.countplot(x="family_history", hue="lung_cancer", data=df)
plt.title("Family History vs Lung Cancer")
plt.savefig(os.path.join(output_folder, "family_history_vs_lung_cancer.png"))

plt.figure(figsize=(5,5))
df['lung_cancer'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Lung Cancer Cases Distribution")
plt.ylabel("")
plt.savefig(os.path.join(output_folder, "lung_cancer_cases_distribution.png"))