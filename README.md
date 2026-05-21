# ✈️ Advanced Jet Engine Predictive Maintenance  
## CNN-LSTM Hybrid Model for Remaining Useful Life Prediction

Predicting the **Remaining Useful Life (RUL)** of aircraft engines is a critical challenge in aerospace engineering. Unscheduled maintenance is costly and risky, while premature maintenance wastes valuable engine life.

This project implements a **Hybrid CNN-LSTM Deep Learning architecture** to provide real-time engine health diagnostics using the **NASA C-MAPSS dataset**. By combining signal processing, sequential learning, and dual-head prediction logic, the system identifies degradation patterns long before failure occurs.

---

## 🚀 Project Highlights

- Predicts the **Remaining Useful Life** of aircraft engines in cycles.
- Detects whether an engine is at **failure risk within the next 30 cycles**.
- Uses a **CNN-LSTM hybrid architecture** for sensor-based degradation analysis.
- Applies **Savitzky-Golay filtering** to reduce high-frequency sensor noise.
- Provides an interactive **Streamlit dashboard** for real-time prediction and visualization.

---

## 🧠 AI Pipeline

The project follows an end-to-end machine learning workflow:

### 1. Signal Processing

A **Savitzky-Golay filter** is applied to smooth high-frequency sensor noise.

This helps the model focus on true degradation trends instead of random sensor jitter.

### 2. Feature Extraction using CNN

A **Conv1D layer** acts as a spatial feature extractor.

It identifies important relationships across multiple engine sensor readings.

### 3. Temporal Pattern Learning using LSTM

An **LSTM layer** learns how engine degradation evolves over time.

This allows the model to understand sequential patterns from historical sensor cycles.

### 4. Dual-Head Predictive Logic

The model uses two prediction heads:

#### Regression Head

Predicts the exact **Remaining Useful Life** in cycles.

#### Classification Head

Predicts whether the engine is:

- **Safe**
- **Failure Risk within 30 cycles**

---

## 📊 Key Results

| Metric | Performance |
|---|---:|
| Root Mean Square Error | 14.25 cycles |
| Binary Classification Accuracy | 95.00% |

---

## 📈 Model Reliability

The binary classifier demonstrates strong performance in distinguishing between healthy engines and engines approaching critical failure.

This makes the system useful for proactive maintenance planning and early risk detection.

---

## 📉 Degradation Trajectory

The regression model provides a continuous estimation of engine lifespan.

This enables maintenance teams to schedule inspections and repairs before failure occurs, reducing downtime and operational risk.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy, Scikit-Learn |
| Signal Analysis | SciPy, Savitzky-Golay Filter |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

## 📁 Repository Structure

```plaintext
├── app.py                  # Streamlit dashboard code
├── rul_regressor_v2.h5     # Trained regression model
├── rul_classifier_v2.h5    # Trained classification model
├── scaler_v2.pkl           # Preprocessing scaler
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

⚙️ How to Run

1. Clone the Repository
git clone [your-repo-link]
cd [your-repo-name]

2. Install Dependencies
pip install -r requirements.txt

3. Launch the Streamlit Dashboard
streamlit run app.py

🧪 Dataset

This project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset, a widely used benchmark dataset for predictive maintenance and Remaining Useful Life estimation.

The dataset contains multiple sensor readings collected over engine operating cycles until failure.

🎯 Objective

The main goal of this project is to build an intelligent predictive maintenance system that can:

Monitor engine health from sensor data.
Predict future engine degradation.
Estimate Remaining Useful Life.
Identify engines at high risk of failure.
Support safer and more cost-efficient maintenance decisions.
🧩 Model Architecture
Input Sensor Sequence
        ↓
Savitzky-Golay Filtering
        ↓
Data Scaling & Windowing
        ↓
Conv1D Feature Extraction
        ↓
LSTM Temporal Learning
        ↓
Dual Output Heads
   ┌───────────────┬────────────────────┐
   ↓               ↓
RUL Regression   Failure Risk Classification
✅ Output

The system provides:

Predicted RUL in engine cycles.
Failure risk status.
Health condition interpretation.
Degradation trend visualization.
Maintenance decision support.

📌 Future Improvements

Add real-time sensor data streaming support.
Integrate model explainability using SHAP or LIME.
Deploy the dashboard on Streamlit Cloud or Hugging Face Spaces.
Add support for multiple C-MAPSS subsets.
Improve classification using multi-class health stages.
Add alert notifications for high-risk engines.

🏁 Conclusion

This project demonstrates how deep learning can be used for aerospace predictive maintenance by combining sensor signal processing with a CNN-LSTM hybrid architecture.

By predicting both exact engine lifespan and near-failure risk, the system provides a practical foundation for safer, smarter, and more cost-effective aircraft maintenance.
