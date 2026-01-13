import { StockSearch } from "@/components/predictions/stock-search"
import { Card, CardContent } from "@/components/ui/card"
import { TrendingUp, ArrowRight } from "lucide-react"
import Link from "next/link"

export default function PredictionsPage() {
  const trending = [
    { symbol: "RELIANCE.NS", name: "Reliance Industries", change: "+1.85%", sentiment: "Bullish" },
    { symbol: "TCS.NS", name: "Tata Consultancy Svc", change: "-0.45%", sentiment: "Neutral" },
    { symbol: "HDFCBANK.NS", name: "HDFC Bank", change: "+0.92%", sentiment: "Bullish" },
    { symbol: "ADANIENT.NS", name: "Adani Enterprises", change: "+2.15%", sentiment: "Bullish" },
    { symbol: "INFY.NS", name: "Infosys Ltd", change: "-1.20%", sentiment: "Bearish" },
  ]

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div className="text-center space-y-4 py-8">
        <h1 className="text-4xl font-display font-bold text-foreground">
          Market Predictions
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Search for any NSE/BSE stock to get instant AI-powered forecasts, sentiment analysis, and risk metrics.
        </p>
        <div className="flex justify-center pt-4">
          <StockSearch />
        </div>
      </div>

      <div className="grid gap-6">
        <h2 className="text-xl font-display font-bold text-foreground flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          Trending Now
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {trending.map((stock) => (
            <Link key={stock.symbol} href={`/predictions/${stock.symbol}`}>
              <Card className="bg-card border-border hover:border-primary/50 transition-all cursor-pointer group shadow-sm hover:shadow-md">
                <CardContent className="p-4 flex items-center justify-between">
                  <div>
                    <div className="font-display font-bold text-foreground group-hover:text-primary transition-colors">
                      {stock.symbol}
                    </div>
                    <div className="text-sm text-muted-foreground">{stock.name}</div>
                  </div>
                  <div className="text-right">
                    <div className={`font-medium ${stock.change.startsWith('+') ? 'text-bullish' : 'text-bearish'}`}>
                      {stock.change}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1 flex items-center justify-end gap-1">
                      {stock.sentiment}
                      <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
