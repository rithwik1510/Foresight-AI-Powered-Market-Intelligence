import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, TrendingUp, TrendingDown, Wallet, PieChart, BarChart3, Sparkles, IndianRupee, Activity, Zap } from "lucide-react"
import Link from "next/link"
import { dashboardApi } from "@/lib/api/dashboard"

// Mock trending stocks data
const trendingStocks = [
  { symbol: "RELIANCE", name: "Reliance Industries", price: 2945.50, change: 2.34 },
  { symbol: "TCS", name: "Tata Consultancy", price: 4125.00, change: 1.85 },
  { symbol: "HDFCBANK", name: "HDFC Bank", price: 1685.25, change: -0.92 },
  { symbol: "INFY", name: "Infosys", price: 1890.75, change: 1.45 },
  { symbol: "ICICIBANK", name: "ICICI Bank", price: 1245.60, change: -0.35 },
]

// Mock watchlist
const watchlist = [
  { symbol: "TATAMOTORS", price: 785.50, change: 3.25 },
  { symbol: "WIPRO", price: 465.80, change: -1.12 },
  { symbol: "SBIN", price: 825.40, change: 1.85 },
  { symbol: "BHARTIARTL", price: 1645.20, change: 0.92 },
]

export default async function Home() {
  // Portfolio mock data
  const portfolio = {
    totalValue: 1245890,
    totalInvested: 1050000,
    totalReturns: 195890,
    returnsPct: 18.65,
    todayChange: 12450,
    todayChangePct: 1.01,
    holdingsCount: 12,
  }

  // Market data - try to fetch real, fallback to mock
  let niftyPrice = 25485.50;
  let niftyChange = 0.85;
  let sensexPrice = 84180.25;
  let sensexChange = 0.72;
  let sentimentLabel = "Neutral";
  let sentimentIndex = 52;

  try {
    const indices = await dashboardApi.getTrendingStocks().catch(() => null);
    if (indices?.nifty) {
      niftyPrice = indices.nifty.current_price;
      niftyChange = ((indices.nifty.current_price - indices.nifty.previous_close) / indices.nifty.previous_close * 100);
    }
    if (indices?.sensex) {
      sensexPrice = indices.sensex.current_price;
      sensexChange = ((indices.sensex.current_price - indices.sensex.previous_close) / indices.sensex.previous_close * 100);
    }
  } catch (e) {
    // Use mock data
  }

  try {
    const sentiment = await dashboardApi.getMarketSentiment().catch(() => null);
    if (sentiment) {
      sentimentLabel = sentiment.label;
      sentimentIndex = sentiment.sentiment_index;
    }
  } catch (e) {
    // Use mock data
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Hero Section - Portfolio Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Portfolio Card */}
        <Card className="lg:col-span-2 border-border bg-gradient-to-br from-card to-secondary/30">
          <CardContent className="p-6 md:p-8">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <p className="text-muted-foreground text-sm uppercase tracking-widest font-medium mb-2">Total Portfolio Value</p>
                <h1 className="text-4xl md:text-5xl font-display font-bold text-foreground tracking-tight">
                  ₹{portfolio.totalValue.toLocaleString('en-IN')}
                </h1>
                <div className="flex items-center gap-3 mt-3">
                  <span className={`flex items-center gap-1 text-sm font-medium px-2 py-1 rounded-full ${portfolio.todayChangePct >= 0 ? 'bg-bullish/10 text-bullish' : 'bg-bearish/10 text-bearish'}`}>
                    {portfolio.todayChangePct >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                    {portfolio.todayChangePct >= 0 ? '+' : ''}{portfolio.todayChangePct.toFixed(2)}%
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {portfolio.todayChange >= 0 ? '+' : ''}₹{portfolio.todayChange.toLocaleString('en-IN')} today
                  </span>
                </div>
              </div>
              <Link href="/portfolio">
                <button className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2.5 rounded-lg font-medium hover:bg-primary/90 transition-colors">
                  <Wallet className="h-4 w-4" />
                  View Portfolio
                </button>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-1 gap-4">
          <Card className="border-border bg-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-bullish/10">
                  <TrendingUp className="h-5 w-5 text-bullish" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase">Total Returns</p>
                  <p className="text-lg font-bold text-bullish">+₹{portfolio.totalReturns.toLocaleString('en-IN')}</p>
                  <p className="text-xs text-muted-foreground">+{portfolio.returnsPct}%</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border bg-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <PieChart className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase">Holdings</p>
                  <p className="text-lg font-bold text-foreground">{portfolio.holdingsCount} Stocks</p>
                  <p className="text-xs text-muted-foreground">₹{portfolio.totalInvested.toLocaleString('en-IN')} invested</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Market Indices Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground font-medium">NIFTY 50</p>
            <p className="text-xl font-bold text-foreground mt-1">{niftyPrice.toLocaleString('en-IN', { maximumFractionDigits: 2 })}</p>
            <p className={`text-sm font-medium ${niftyChange >= 0 ? 'text-bullish' : 'text-bearish'}`}>
              {niftyChange >= 0 ? '+' : ''}{niftyChange.toFixed(2)}%
            </p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground font-medium">SENSEX</p>
            <p className="text-xl font-bold text-foreground mt-1">{sensexPrice.toLocaleString('en-IN', { maximumFractionDigits: 2 })}</p>
            <p className={`text-sm font-medium ${sensexChange >= 0 ? 'text-bullish' : 'text-bearish'}`}>
              {sensexChange >= 0 ? '+' : ''}{sensexChange.toFixed(2)}%
            </p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground font-medium">MARKET MOOD</p>
            <p className="text-xl font-bold text-foreground mt-1">{sentimentLabel}</p>
            <p className="text-sm text-muted-foreground">Index: {sentimentIndex}/100</p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground font-medium">FII/DII</p>
            <p className="text-xl font-bold text-bullish mt-1">+₹1,245 Cr</p>
            <p className="text-sm text-muted-foreground">Net buying today</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Trending & Actions */}
        <div className="lg:col-span-2 space-y-6">
          {/* Trending Stocks */}
          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg font-medium flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                Trending Today
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border">
                {trendingStocks.map((stock) => (
                  <Link key={stock.symbol} href={`/predictions/${stock.symbol}`}>
                    <div className="flex items-center justify-between p-4 hover:bg-secondary/50 transition-colors cursor-pointer">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center font-bold text-foreground text-sm">
                          {stock.symbol.slice(0, 2)}
                        </div>
                        <div>
                          <p className="font-medium text-foreground">{stock.symbol}</p>
                          <p className="text-xs text-muted-foreground">{stock.name}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-foreground">₹{stock.price.toLocaleString('en-IN')}</p>
                        <p className={`text-sm font-medium ${stock.change >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                          {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Link href="/predictions">
              <Card className="border-border bg-gradient-to-br from-primary/5 to-primary/10 hover:from-primary/10 hover:to-primary/20 transition-all cursor-pointer group h-full">
                <CardContent className="p-5 flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-colors">
                    <Sparkles className="h-6 w-6 text-primary" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-foreground">AI Predictions</p>
                    <p className="text-sm text-muted-foreground">Get buy/sell signals</p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                </CardContent>
              </Card>
            </Link>
            <Link href="/advisor">
              <Card className="border-border bg-gradient-to-br from-secondary to-secondary/50 hover:from-secondary/80 hover:to-secondary/30 transition-all cursor-pointer group h-full">
                <CardContent className="p-5 flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-foreground/5 group-hover:bg-foreground/10 transition-colors">
                    <Zap className="h-6 w-6 text-foreground" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-foreground">AI Advisor</p>
                    <p className="text-sm text-muted-foreground">Ask anything about markets</p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                </CardContent>
              </Card>
            </Link>
            <Link href="/funds">
              <Card className="border-border bg-card hover:bg-secondary/50 transition-all cursor-pointer group h-full">
                <CardContent className="p-5 flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-secondary">
                    <BarChart3 className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-foreground">Mutual Funds</p>
                    <p className="text-sm text-muted-foreground">Analyze fund overlap</p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                </CardContent>
              </Card>
            </Link>
            <Link href="/stocks">
              <Card className="border-border bg-card hover:bg-secondary/50 transition-all cursor-pointer group h-full">
                <CardContent className="p-5 flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-secondary">
                    <IndianRupee className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-foreground">Browse Stocks</p>
                    <p className="text-sm text-muted-foreground">Search & analyze</p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                </CardContent>
              </Card>
            </Link>
          </div>
        </div>

        {/* Right Column - Watchlist */}
        <div className="space-y-6">
          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg font-medium">Watchlist</CardTitle>
                <Link href="/stocks" className="text-sm text-primary hover:underline">Edit</Link>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border">
                {watchlist.map((stock) => (
                  <Link key={stock.symbol} href={`/predictions/${stock.symbol}`}>
                    <div className="flex items-center justify-between p-4 hover:bg-secondary/50 transition-colors cursor-pointer">
                      <p className="font-medium text-foreground">{stock.symbol}</p>
                      <div className="text-right">
                        <p className="font-medium text-foreground">₹{stock.price.toLocaleString('en-IN')}</p>
                        <p className={`text-xs font-medium ${stock.change >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                          {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
              <div className="p-4 border-t border-border">
                <Link href="/stocks">
                  <button className="w-full text-center text-sm text-primary hover:underline font-medium">
                    + Add to Watchlist
                  </button>
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Market News Preview */}
          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg font-medium">Latest News</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <p className="text-sm font-medium text-foreground line-clamp-2">RBI keeps repo rate unchanged at 6.5% for 11th consecutive time</p>
                <p className="text-xs text-muted-foreground">2 hours ago</p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium text-foreground line-clamp-2">FIIs turn net buyers in Indian equities after 5 months</p>
                <p className="text-xs text-muted-foreground">4 hours ago</p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium text-foreground line-clamp-2">IT sector rallies on strong Q3 earnings guidance</p>
                <p className="text-xs text-muted-foreground">6 hours ago</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
