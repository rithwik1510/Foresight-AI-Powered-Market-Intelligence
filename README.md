# Foresight: AI-Powered Market Intelligence 🚀

![Project Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-4.0-38B2AC?logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**Foresight** is an institutional-grade investment analysis platform engineered for the Indian markets (NSE/BSE). It bridges the gap between retail investors and hedge-fund technology by combining an ensemble of **5 advanced Machine Learning models** with **Generative AI** to provide actionable, data-driven market insights.

---

## 🧠 Core Intelligence

Foresight goes beyond simple technical indicators. It employs a sophisticated **"Mixture of Experts"** approach to predict stock movements:

### 🤖 The ML Prediction Ensemble
We don't rely on a single algorithm. Our system aggregates predictions from 5 distinct models to ensure robustness:
1.  **ARIMA:** Statistical baseline for capturing linear trends.
2.  **Facebook Prophet:** specialized in detecting seasonality and holiday effects.
3.  **XGBoost:** Gradient boosting for detecting non-linear direction patterns (Bullish/Bearish).
4.  **LightGBM:** Optimized for predicting precise return magnitudes.
5.  **Random Forest:** Estimating probability confidence intervals.

### 📰 Multi-Source Sentiment Engine
Market moves are often driven by emotion. Foresight continuously scans and analyzes:
*   **News Aggregation:** Real-time processing of NewsAPI, Google News, and RSS feeds from top Indian financial dailies (Economic Times, Moneycontrol).
*   **Social Sentiment:** Analysis of retail sentiment from Reddit (r/IndiaInvestments, r/IndianStreetBets).
*   **Macro Indicators:** Integration of global factors (S&P 500, Oil, Gold, USD/INR) and economic data (FRED API).

### 💬 AI Investment Advisor
Integrated with **Google Gemini Pro**, Foresight offers a chat interface that understands your portfolio context. Ask questions like:
*   *"Analyze the risk factors for HDFCBANK given the recent RBI news."*
*   *"How does my portfolio exposure compare to the Nifty 50?"*

---

## ⚡ Tech Stack

### Frontend (Modern & Fast)
*   **Framework:** Next.js 16 (App Router)
*   **Language:** TypeScript
*   **Styling:** Tailwind CSS v4 + Shadcn/UI
*   **Visualization:** Recharts & Lightweight Charts (TradingView style)
*   **State Management:** Zustand & TanStack Query

### Backend (Robust & Scalable)
*   **API:** FastAPI (Python)
*   **Database:** PostgreSQL (via SQLAlchemy & AsyncPG)
*   **ML Engine:** Scikit-learn, PyPortfolioOpt, Statsmodels
*   **Data Processing:** Pandas, NumPy
*   **Caching:** Redis

---

## 🛠️ Getting Started

Follow these steps to set up Foresight locally.

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   PostgreSQL & Redis (Running locally or via cloud providers like Supabase/Upstash)

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (see Configuration section below)
cp .env.example .env

# Run the server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. API Docs: `http://localhost:8000/docs`.

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The UI will be available at `http://localhost:3000`.

---

## 🔑 Configuration (.env)

Create a `.env` file in the `backend/` directory with the following variables:

```env
# Database & Cache
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/foresight_db
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=your_super_secret_key_change_this
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI & Data APIs
GOOGLE_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key
# Optional (for enhanced data)
FINNHUB_API_KEY=your_finnhub_key
FRED_API_KEY=your_fred_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
```

---

## 📊 Features Overview

| Feature | Status | Description |
| :--- | :---: | :--- |
| **Stock Analysis** | ✅ | Deep dive technicals, fundamentals, and AI predictions for NSE/BSE stocks. |
| **Mutual Funds** | 🚧 | Track NAV history, overlapping holdings, and performance metrics. |
| **Portfolio Optimizer** | 🚧 | HRP (Hierarchical Risk Parity) optimization to minimize risk. |
| **News Feed** | ✅ | Aggregated financial news with sentiment scoring. |
| **Paper Trading** | ⏳ | Simulate trades to test strategies without real money. |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by Rithwik
</p>