# 💓 Hart- en Vaatziekten Voorspellen met Machine Learning & Neurale Netwerken

Deze repository bevat een volledig pipelineproject waarin het risico op hart- en vaatziekten voorspeld wordt aan de hand van de **Cleveland Heart Disease dataset**. Zowel klassieke machine learning (logistic regression) als neurale netwerken zijn toegepast en geëvalueerd.

---

## 📁 Mappenstructuur & Inhoud

### `Data_&_Model/`
Bevat de ruwe dataset en de belangrijkste voorbereidingsstappen:
- `data.csv` – Schoongemaakte Cleveland dataset (294 patiënten)
- `DataPreparatie_Modelkeuze.ipynb` – Bevat:
  - Exploratory Data Analysis (EDA) met kernbevindingen
  - Opschoning (missende waarden, typecasting, droppen kolommen)
  - Imputatie & scaling (`StandardScaler`)
  - Train/test split (80/20, **stratified**)
  - Modelvergelijking (8 modellen)
  - Selectie van het beste model: **Logistic Regression**

> ℹ️ In deze notebook is ook een **vaste random seed** gebruikt voor reproduceerbaarheid (`random_state=42`).

---

### `TrainTestData_&_Scaler/`
Kant-en-klare datasets om het model opnieuw te trainen of testen:
- `X_ontrain.pkl` / `X_ontest.pkl` – Onbewerkte train- en testsets
- `X_ontrain_scaled.pkl` / `X_ontest_scaled.pkl` – Geschaalde versies
- `y_ontrain.pkl` / `y_ontest.pkl` – Labels
- `standard_onscaler.pkl` – De gebruikte scaler

> ✔️ Deze bestanden maken het mogelijk om het model **direct te hertrainen** zonder opnieuw preprocessing uit te voeren.

---

### `Logistic_regression/`
Uitgewerkte pipeline voor logistic regression:
- `Logistic_regression_ft.ipynb` – Hyperparameter tuning & modelselectie
- `Logistic_regression_fi.ipynb` – SHAP feature importance analyse
- `Logistic_regression_test.ipynb` – Testresultaten op de testset + test op 1 individuele deelnemer

---

### `Neuraal_Netwerk/`
Uitgewerkte pipeline voor het neuraal netwerk:
- `Neuraal_netwerk_ft.ipynb` – Fine-tuning en trainingsproces
- `Neuraal_netwerk_fi.ipynb` – SHAP feature importance
- `Neuraal_netwerk_test.ipynb` – Test op de volledige testset + test op één deelnemer

---

### `Models_&_threshold/`
Opgeslagen eindmodellen en drempelwaarden:
- `nfinal_logreg_model.pkl` – Eind Logistic Regression model
- `nfinal_logreg_threshold.pkl` – Bijbehorende optimal threshold
- `model5onnn.keras` – Eind neuraal netwerk model
- `final_onn_threshold.pkl` – Threshold voor het NN-model

> 🔒 Deze artefacten maken **hergebruik of deployment** mogelijk zonder retraining.

### `Raport_&_Logbook/`
Bevat het eind rapport, publishable paper en het logboek:
- `Logboek_LotteLampe_MAAI2025.pdf` – Logboek met de bijlage, feedback momenten en voortgang gedurende afstudeer periode
- `Publishable_Paper_LotteLampe_MAAI2025.pdf` – publiseerbare versie van het eind rapport met focus op het kiezen van het beste ML model en XAI
- `Rapport_LotteLampe_MAAI2025.pdf` – Eind rapport MAAI


---

# 🫀 Predictie van Hartziekte met Explainable AI

## 🎯 Projectdoelen

1. **Voorspellen** of een patiënt een verhoogd risico op hartziekte heeft op basis van klinische kenmerken.
2. **Vergelijken** van klassieke machine learning-modellen met een neuraal netwerk.
3. **Interpreteren** van individuele voorspellingen met behulp van SHAP (SHapley Additive exPlanations).

---

## 🧬 Belangrijkste Resultaten

- **Beste model**: Logistic Regression  
  - Recall ≈ 0.860
  - ROC AUC ≈ 0.936
  - Accuracy ≈ 0.864
- **Neuraal netwerk**:  
  - Recall ≈ 0.810
  - ROC AUC ≈ 0.936
  - Accuracy ≈ 0.831

- **Top 5 invloedrijke features (volgens SHAP):**
  1. `cp` - Chest pain type 
  2. `exang` - Exercise-induced angina  
  3. `oldpeak` - ST depression  
  4. `Sex ` - Male or female
  5. `chol` - Total cholestero

---

## 🔁 Reproduceerbaarheid

Deze studie is volledig reproduceerbaar dankzij:

- **Vaste random seed** (`random_state=42`) voor consistente resultaten
- **Modellen & datasets opgeslagen** met `joblib`, `pickle` of `keras`
- **Gedocumenteerde preprocessing** in Jupyter notebooks
- **Kant-en-klare `.pkl` bestanden** beschikbaar voor snelle hertraining of evaluatie

---

## 🚀 Zelf Uitvoeren

### 1. Clone de repository
```bash
git clone <repository-url>
cd <repository-folder>


---

## 🚀 Zelf uitvoeren

1. Clone deze repository
```bash
git clone <repository-url>
cd <repository-folder>

2. Installeer de dependencies
pip install -r requirements.txt

3. Voer de notebooks uit in volgorde:

- Data_en_Model/DataPreparatie_Modelkeuze.ipynb
- Kies vervolgens:
  - Logistic_regression/ → voor het LR-model
  - Neuraal_Netwerk/ → voor het NN-model
  - Gebruik eventueel de bestanden in Data_voor_Trainen/ om direct te starten met modeltraining of testen.


📦 Vereisten
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- shap
- tensorflow / keras
- joblib

Installeer alles via:
pip install -r requirements.txt
