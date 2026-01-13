# AI Stock & Mutual Fund Analyzer

A professional-grade AI-powered investment analysis platform for Indian markets (NSE/BSE).

## Project Status: Phase 1 - Foundation (In Progress)

### What's Built
- [x] Project planning complete
- [x] Backend config structure started (`backend/app/config.py`, `main.py`)
- [ ] Complete backend API structure
- [ ] Frontend setup
- [ ] Database integration
- [ ] Data source integrations

---

## Quick Start (Once Complete)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## Tech Stack

| Layer | Technology | Hosting |
|-------|------------|---------|
| Frontend | Next.js 14 + React + TailwindCSS | Vercel (free) |
| Backend | Python FastAPI | Railway/Render (free) |
| Database | PostgreSQL | Supabase (free tier) |
| Cache | Redis | Upstash (free tier) |
| AI | Google Gemini API | - |
| Charts | Lightweight-charts, Recharts | - |

---

## Core Features

### 1. Portfolio Analyzer
- Track holdings (stocks + mutual funds)
- Sector exposure visualization
- HRP (Hierarchical Risk Parity) optimization
- Risk metrics: VaR, CVaR, Sharpe Ratio, Beta

### 2. Stock Analyzer
- Fundamentals: P/E, P/B, ROE, Debt ratios
- Technicals: RSI, MACD, Moving Averages
- Price charts with indicators
- Peer comparison

### 3. Mutual Fund Analyzer
- Performance: CAGR, Sharpe, Sortino
- Fund overlap detection
- Style drift analysis
- SIP calculator

### 4. AI Advisor (Gemini)
- Ask investment questions
- Get AI-powered analysis with cited data
- Sentiment analysis on news

---

## Data Sources (All Free)

| Data | Source | Usage |
|------|--------|-------|
| Stock Prices | yfinance | NSE: `.NS`, BSE: `.BO` suffix |
| Mutual Funds | mfapi.in | NAV, scheme details |
| Macro Data | FRED API | Interest rates, inflation |
| News | NewsAPI | 100 req/day free |

---

## Project Structure

```
stock-analyzer/
├── frontend/                 # Next.js 14 App
│   ├── app/
│   │   ├── (auth)/          # Login, Register
│   │   ├── (dashboard)/     # Main app pages
│   │   │   ├── portfolio/
│   │   │   ├── stocks/
│   │   │   ├── funds/
│   │   │   └── ai-advisor/
│   ├── components/
│   └── lib/
│
├── backend/                  # FastAPI App
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   ├── integrations/    # External APIs
│   │   └── ml/              # ML pipelines
│   └── requirements.txt
│
├── README.md
└── CONTEXT.md               # Development context
```

---

## Environment Variables

### Backend (.env)
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://default:pass@host:6379
GOOGLE_API_KEY=your-gemini-api-key
NEWS_API_KEY=xxx
JWT_SECRET=your-secret-key
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="Stock Analyzer"
```

---

## Implementation Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation (Setup, Auth, Basic APIs) | In Progress |
| 2 | Core Data (Stock/Fund pages, Charts) | Pending |
| 3 | Portfolio Management | Pending |
| 4 | Advanced Analytics (HRP, Risk) | Pending |
| 5 | AI Integration (Gemini) | Pending |
| 6 | Polish & Production | Pending |

---

## Key Dependencies

### Backend (Python)
- fastapi, uvicorn - Web framework
- sqlalchemy, asyncpg - Database
- yfinance - Stock data
- pypfopt - Portfolio optimization (HRP)
- google-generativeai - Gemini AI
- pandas, numpy, scipy - Data processing

### Frontend (Node.js)
- next, react - Framework
- tailwindcss - Styling
- @tanstack/react-query - Data fetching
- lightweight-charts - Trading charts
- recharts - General charts
- zustand - State management

---

## Reference Document

See `Building an AI Investment Predictor.pdf` for the theoretical foundation including:
- Hierarchical Risk Parity (HRP) algorithm
- CVaR risk management
- Kelly Criterion position sizing
- Factor-based investing concepts
