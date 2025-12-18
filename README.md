# ⚡ Vidyut Sanchay

### **Intelligent AI Orchestrator for Power Grid Procurement & Logistics**

**Vidyut Sanchay** (Sanskrit for *Electricity Accumulation*) is an advanced AI-driven platform designed to revolutionize the supply chain of power transmission infrastructure. By combining **Machine Learning**, **Generative AI (LLMs)**, and **Live Market Data**, it automates the complex planning of transmission lines, supplier selection, logistics routing, and risk analysis.

---

## 🚀 Key Features

* **🧠 Vidyut Brain (AI Orchestrator):** A context-aware chatbot powered by **Llama-3 (via Groq)** and **LangGraph**. It acts as a "God Mode" assistant, answering questions about project plans, live rates, and logistics feasibility in real-time.
* **📊 Smart Procurement Plans:** Uses an **ExtraTreeRegressor ML Model** to predict material quantities (Steel, Conductors, Concrete) based on project terrain, voltage, and soil type.
* **🚚 Logistics Optimization:** Calculates optimal transport routes, estimated time of arrival (ETA), and costs using **Geoapify**.
* **🌍 Live Market Intelligence:** Integrates with **Alpha Vantage** and **ExchangeRate-API** to fetch real-time commodity prices and Forex rates.
* **🛡️ Automated Risk Analysis:** Evaluates geopolitical, financial, and supply chain risks using **Google Gemini** and **SerpApi** for real-time web search.
* **📑 Instant Reporting:** Generates downloadable, invoice-style text reports summarizing the entire project analysis.

---

## 🛠️ Tech Stack

### **Backend (The Brain)**

* **Framework:** FastAPI (Python)
* **AI & LLM:** LangChain, LangGraph, Groq API (Llama-3-70b), Google Gemini
* **Machine Learning:** Scikit-learn (**ExtraTreeRegressor**), Pandas, NumPy
* **Search & Data:** Hugging Face, SerpApi

### **Frontend (The Interface)**

* **Core:** HTML5, JavaScript (ES6+)
* **Styling:** Custom CSS + Bootstrap 5.3
* **Visualization:** Dynamic Dashboard with "Surgical" DOM manipulation

### **External APIs**

* **LLM Inference:** Groq, Google Gemini, Hugging Face
* **Market Data:** Alpha Vantage, ExchangeRate-API
* **Logistics & Maps:** Geoapify
* **Real-Time Search:** SerpApi
* **Environmental:** OpenWeatherMap

### **Infrastructure**

* **Containerization:** Docker
* **Version Control:** Git

---

## 📂 Project Structure

```bash
vidyut_sanchay/
├── backend/
│   ├── main.py                 # FastAPI Entry Point
│   ├── vidyut_brain.py         # LangGraph AI Agent Logic
│   ├── price_agent/            # Live Market Data Module
│   ├── logistics_agent/        # Route Optimization Logic
│   ├── risk_agent/             # AI Risk Analysis Module
│   ├── full_ml_pipeline_1.pkl  # Trained ExtraTreeRegressor Model (3GB+)
│   ├── requirements.txt        # Python Dependencies
│   ├── Dockerfile              # Backend Container Config
│   └── .env                    # API Keys (Not committed)
├── frontend/
│   ├── index.html              # Main Dashboard UI
│   ├── css/                    # Custom Styles
│   └── js/                     # Frontend Logic (app.js)
└── README.md                   # Project Documentation

```

---

## ⚙️ Installation & Setup

### **Prerequisites**

* Python 3.9 or higher
* Docker Desktop (Optional, for containerization)

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/vidyut-sanchay.git
cd vidyut-sanchay

```

### **2. Environment Configuration**

Create a `.env` file in the `backend/` folder with your API keys:

```ini
# backend/.env
GROQ_API_KEY=gsk_...
ALPHAVANTAGE_API_KEY=...
EXCHANGERATE_API_KEY=...
GEOAPIFY_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
GOOGLE_API_KEY=AIza...
SERPAPI_KEY=...
WEATHER_API_KEY=...

```

### **3. Running Locally (Without Docker)**

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

```

*The API will be live at `http://127.0.0.1:8000`.*
*Open `frontend/index.html` in your browser to use the app.*

---

## 🐳 Docker Setup (Recommended)

Since the ML model is large and we want to persist API caches, use **Volume Mounting**.

### **1. Build the Image**

```bash
cd backend
docker build -t vidyut-backend .

```

### **2. Run the Container**

**Windows PowerShell:**

```powershell
docker run -p 8000:8000 --env-file .env `
  -v "E:/path/to/vidyut_sanchay/backend/full_ml_pipeline_1.pkl:/app/full_ml_pipeline_1.pkl" `
  -v "E:/path/to/vidyut_sanchay/backend/price_cache.json:/app/price_cache.json" `
  vidyut-backend

```

*(Replace `E:/path/to/...` with your actual absolute path).*

---

## 📖 Usage Guide

1. **Generate Plan:**
* Enter project details (City, Terrain, Voltage, Length).
* Click **"Generate Plan"**. The ML model will predict material needs, and agents will fetch prices and routes.


2. **Ask Vidyut Assistant:**
* Click the **Robot Icon** (bottom right).
* Ask questions like:
* *"Why did you choose Skipper as the supplier?"*
* *"What is the current price of Copper?"*
* *"Is the route from Kolkata to Mumbai affected by weather?"*




3. **Download Report:**
* Click the **"Download Report"** button in the navbar to get a comprehensive `.txt` summary of costs, risks, and logistics.



---

## 🚧 Troubleshooting

* **API Rate Limit Errors:** Alpha Vantage and ExchangeRate-API have free tier limits. The app caches data in `price_cache.json` to avoid this. If you see errors, wait 5 minutes and restart.
* **Docker "File Not Found":** Ensure you use **absolute paths** (e.g., `C:/Users/...`) in the `-v` volume mount command.
* **ML Model Missing:** Ensure `full_ml_pipeline_1.pkl` is present in the `backend/` folder before running.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
Made with ❤️ by <a href="[https://github.com/your-github-profile](https://www.google.com/search?q=https://github.com/your-github-profile)">Hack Mavericks</a> for Smart Grid Innovation.
</p>
