# FoodBridge AI

FoodBridge AI is a comprehensive Flask-based backend designed to minimize food waste at large events. It utilizes a **Hybrid Prediction Engine** that combines mathematical formulas, historical data, and AI to provide donors with precise preparation quantities.

## Key Features

* **Role-Based Access:** Specialized dashboards for Donors, NGOs, Individuals, and Admins.
* **Hybrid AI Engine:** Predicts wastage using a blend of ICMR standards, Random Forest regression, and Gemini AI fallbacks.
* **NGO Matching:** Automatically calculates distances using the Haversine formula to find the nearest NGOs for surplus pickup.
* **Roadside SOS:** Allows guest users to report needy individuals on the roadside, instantly alerting local NGOs.
* **Data-Driven Accuracy:** As more "Actual Wastage" data is confirmed in the database, the system shifts from formula-based to ML-dominant predictions.

## Internal Configuration

This project is configured for ease of deployment without external environment variables. 

* **Supabase:** The REST API URL and Service Key are declared directly within `app.py`.
* **Gemini AI:** The API key for fallback predictions is integrated within the prediction logic.

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server:**
    ```bash
    python app.py
    ```
    The application will start on `http://localhost:5000`.

3.  **Access the Dashboard:**
    Open `index.html` in your browser or serve it through the Flask root.

## Admin Capabilities

* **Training:** Admins can trigger model retraining directly from the dashboard.
* **Data Upload:** Support for uploading CSV datasets (like the Tamil Nadu Food Surplus dataset) to improve model accuracy.
* **User Management:** View and manage all registered donors and NGOs.

---
**Note:** For production environments, ensure the hardcoded API keys are secured or restricted within your provider's console.
