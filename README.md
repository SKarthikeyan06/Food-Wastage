# FoodBridge AI v2 🍲

FoodBridge AI is a high-performance Flask-based web application designed to combat food waste in the catering and hospitality industry. It uses a **Hybrid Prediction Engine** that blends scientific formulas, Machine Learning (Random Forest), and Generative AI (Google Gemini) to predict food wastage and coordinate surplus donations to NGOs.

## 🚀 Key Features

* **Hybrid Prediction Engine:** * *Formula Mode:* Uses ICMR per-capita standards and research-based waste factors.
    * *ML Mode:* A Scikit-Learn Random Forest Regressor trained on 6,000+ historical data points.
    * *Gemini AI Fallback:* Leverages `gemini-2.0-flash` for high-reasoning predictions when data confidence is low.
* **Role-Based Access Control:** Custom dashboards for **Donors** (Caterers/Hotels), **NGOs**, **Individuals**, and **Admins**.
* **Real-time Alerts:** * *Soft Alerts:* Notifies NGOs to be on standby based on predictions.
    * *Hard Alerts:* Instant "Pick up now" notifications once surplus is confirmed.
* **SOS Roadside Reporting:** A public, no-login-required feature to report needy individuals or orphans to the nearest NGOs.
* **Database Integration:** Fully integrated with **Supabase (PostgreSQL)** for persistence and AI training data.
* **Impact Tracking:** Visualizes CO2 prevention, people fed, and food saved metrics.

## 🛠️ Tech Stack

-   **Backend:** Python / Flask
-   **Frontend:** HTML5, CSS3 (Syne & DM Sans typography), Vanilla JavaScript
-   **Database:** Supabase (REST API)
-   **AI/ML:** Scikit-Learn (Random Forest), Google Gemini API
-   **Geospatial:** Haversine formula for NGO-Donor distance mapping

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-repo/foodbridge-ai.git](https://github.com/your-repo/foodbridge-ai.git)
    cd foodbridge-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration:**
    The application is currently configured with Supabase credentials inside `app.py`. For production, it is recommended to move these to a `.env` file:
    - `SUPABASE_URL`
    - `SUPABASE_KEY`
    - `GEMINI_API_KEY`

4.  **Run the application:**
    ```bash
    python app.py
    ```
    The app will be available at `http://localhost:5000`.

## 🧪 Admin & Testing

* **Demo Accounts:**
    * **Donor:** `donor@demo.com` | `demo123`
    * **NGO:** `ngo@demo.com` | `demo123`
    * **Admin:** `admin@demo.com` | `demo123`
* **Training Data:** Admins can upload CSV datasets (Tamil Nadu Food Surplus or Original Food Waste data) via the Admin Dashboard to retrain the ML model.

## 📊 Prediction Logic
The system calculates the **Optimal Quantity** using the following formula:
$$Optimal = (Guests \times AttendanceRate \times PerCapitaConsumption) \times Buffer$$

It then applies a **Wastage Factor ($W_f$)**:
$$W_f = BaseWaste \times MethodMultiplier \times PricingMultiplier \times SeasonMultiplier$$

The ML model then refines this result based on historical $ActualWastage$ patterns stored in the database.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any feature enhancements.

---
*Developed to bridge the gap between surplus and scarcity.*
