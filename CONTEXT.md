# Development Context - AI Stock Analyzer

**Last Updated:** January 7, 2026
**Status:** Phase 1 - Foundation (In Progress)

---

## User Requirements Summary

1. **Target Market:** India (NSE/BSE stocks, Indian mutual funds)
2. **Frontend:** Next.js 14 + React + TailwindCSS
3. **Deployment:** Vercel (frontend) + Railway/Render (backend)
4. **Data Budget:** Free tier only
5. **AI Backend:** Google Gemini API
6. **Authentication:** Email/Password (JWT)
7. **Scope:** All features (full implementation)

---

## Files Already Created

```
backend/
├── app/
│   ├── __init__.py          ✅ Created
│   ├── config.py            ✅ Created (Settings with pydantic)
│   └── main.py              ✅ Created (FastAPI app entry)
```

---

## Next Steps (Continue From Here)

### Immediate Tasks:
1. Create API route files (`backend/app/api/v1/`)
   - `stocks.py` - Stock endpoints
   - `funds.py` - Mutual fund endpoints
   - `portfolio.py` - Portfolio endpoints
   - `analysis.py` - Analysis endpoints
   - `ai.py` - AI advisor endpoints
   - `auth.py` - Authentication endpoints

2. Create integrations (`backend/app/integrations/`)
   - `yahoo_finance.py` - yfinance wrapper for NSE/BSE
   - `mfapi.py` - mfapi.in wrapper for mutual funds

3. Create database models (`backend/app/models/`)
4. Set up Next.js frontend

---

## Key Implementation Details

### Indian Stock Symbols (yfinance)
```python
# NSE stocks: Add .NS suffix
"RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"

# BSE stocks: Add .BO suffix
"RELIANCE.BO", "TCS.BO"

# Indices
"^NSEI"  # Nifty 50
"^BSESN" # Sensex
```

### Mutual Fund API (mfapi.in)
```python
# Base URL
BASE_URL = "https://api.mfapi.in/mf"

# Get all funds
GET /mf

# Get specific fund by scheme code
GET /mf/{scheme_code}

# Response includes: scheme_name, nav, date, historical data
```

### HRP Portfolio Optimization
```python
from pypfopt import HRPOpt

def optimize_portfolio(returns_df):
    hrp = HRPOpt(returns_df)
    weights = hrp.optimize()
    return dict(weights)
```

### Gemini AI Integration
```python
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

response = model.generate_content(prompt)
```

---

## Database Schema (Supabase PostgreSQL)

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolios
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Holdings
CREATE TABLE holdings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    asset_type VARCHAR(10) NOT NULL,  -- 'STOCK' or 'MF'
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(18,4) NOT NULL,
    avg_buy_price DECIMAL(18,4) NOT NULL,
    buy_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat History
CREATE TABLE chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## API Endpoints to Implement

### Stocks
```
GET  /api/v1/stocks/search?q={query}
GET  /api/v1/stocks/{symbol}
GET  /api/v1/stocks/{symbol}/fundamentals
GET  /api/v1/stocks/{symbol}/technicals
GET  /api/v1/stocks/{symbol}/history?period=1y
```

### Mutual Funds
```
GET  /api/v1/funds/search?q={query}
GET  /api/v1/funds/{scheme_code}
GET  /api/v1/funds/{scheme_code}/nav-history
POST /api/v1/funds/overlap  (body: {fund_codes: []})
POST /api/v1/funds/sip-calculator
```

### Portfolio
```
GET  /api/v1/portfolios
POST /api/v1/portfolios
GET  /api/v1/portfolios/{id}
POST /api/v1/portfolios/{id}/holdings
GET  /api/v1/portfolios/{id}/analysis
GET  /api/v1/portfolios/{id}/optimize
```

### AI Advisor
```
POST /api/v1/ai/ask  (body: {question: string})
POST /api/v1/ai/analyze-stock/{symbol}
POST /api/v1/ai/analyze-fund/{code}
```

---

## Frontend Pages to Create

```
app/
├── (auth)/
│   ├── login/page.tsx
│   └── register/page.tsx
├── (dashboard)/
│   ├── layout.tsx           # Sidebar + header
│   ├── page.tsx             # Dashboard home
│   ├── portfolio/
│   │   ├── page.tsx         # List portfolios
│   │   ├── new/page.tsx     # Create portfolio
│   │   └── [id]/
│   │       ├── page.tsx     # Portfolio detail
│   │       └── analysis/page.tsx
│   ├── stocks/
│   │   ├── page.tsx         # Stock screener
│   │   └── [symbol]/page.tsx
│   ├── funds/
│   │   ├── page.tsx         # Fund screener
│   │   └── [code]/page.tsx
│   └── ai-advisor/
│       └── page.tsx         # Chat interface
```

---

## Free Tier Limits

| Service | Limit | Strategy |
|---------|-------|----------|
| Vercel | 100GB/mo | Image optimization |
| Railway | $5/mo | Sleep on idle |
| Supabase | 500MB DB | Cache aggressively |
| Upstash | 10K cmd/day | Batch operations |
| Gemini | 15 RPM Pro | Use Flash for simple |
| yfinance | Unofficial | 2 req/sec max |
| mfapi.in | None | Cache daily |
| NewsAPI | 100/day | Cache per stock |

---

## Reference Files

1. **PDF Document:** `Building an AI Investment Predictor.pdf`
   - Contains theoretical foundation for HRP, CVaR, Kelly Criterion

2. **Plan File:** `C:\Users\posan\.claude\plans\gleaming-orbiting-treehouse.md`
   - Full implementation plan with code examples

---

## Commands to Run

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir=false
npm install @tanstack/react-query recharts lightweight-charts zustand axios date-fns
npx shadcn-ui@latest init
npm run dev
```

---

## Key Libraries to Install

### Backend (requirements.txt)
```
fastapi==0.109.0
uvicorn==0.27.0
sqlalchemy==2.0.25
asyncpg==0.29.0
pydantic-settings==2.1.0
redis==5.0.1
yfinance==0.2.36
pandas==2.2.0
numpy==1.26.3
pypfopt==1.5.5
scipy==1.12.0
scikit-learn==1.4.0
google-generativeai==0.4.0
python-jose==3.3.0
passlib[bcrypt]==1.7.4
httpx==0.26.0
python-multipart==0.0.6
```

---

## Resume Prompt

When continuing this project, use this prompt:

> "Continue building the AI Stock Analyzer. Check CONTEXT.md and README.md for current status. We're in Phase 1 - Foundation. Next: complete backend API routes and integrations, then set up the Next.js frontend."
