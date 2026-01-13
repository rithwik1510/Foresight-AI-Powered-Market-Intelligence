import { notFound } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Calendar } from "lucide-react"
import Link from "next/link"
import { PriceChart } from "@/components/charts/price-chart"
import { predictionsApi } from "@/lib/api/predictions"

interface PageProps {
  params: Promise<{ symbol: string }>
}

export default async function PredictionPage({ params }: PageProps) {
  const { symbol } = await params
  const decodedSymbol = decodeURIComponent(symbol).toUpperCase()

  // Fetch real data from APIs
  let prediction, stockInfo, fundamentals, events, history
  let error = null

  try {
    [prediction, stockInfo, fundamentals, events, history] = await Promise.all([
      predictionsApi.getPrediction(decodedSymbol).catch(() => null),
      predictionsApi.getStockInfo(decodedSymbol).catch(() => null),
      predictionsApi.getFundamentals(decodedSymbol).catch(() => null),
      predictionsApi.getEvents(decodedSymbol).catch(() => null),
      predictionsApi.getStockHistory(decodedSymbol, "6mo").catch(() => null),
    ])
  } catch (e) {
    error = e
  }

  // If no prediction data, show error
  if (!prediction || !stockInfo) {
    return (
      <div className="max-w-5xl mx-auto space-y-8 pb-12">
        <div className="flex items-center gap-2">
          <Link href="/predictions">
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground pl-0 gap-2">
              <ArrowLeft className="h-4 w-4" /> Back to Search
            </Button>
          </Link>
        </div>
        <Card className="border-border bg-card shadow-sm p-12 text-center">
          <h2 className="text-2xl font-display font-bold mb-4">Unable to load prediction</h2>
          <p className="text-muted-foreground">Could not fetch data for {decodedSymbol}. Please try another symbol.</p>
        </Card>
      </div>
    )
  }

  // Determine signal based on direction
  const signalMap = {
    bullish: "BUY",
    bearish: "SELL",
    neutral: "HOLD",
  }
  const signal = signalMap[prediction.direction]
  const signalColor = prediction.direction === "bullish" ? "bg-bullish" : prediction.direction === "bearish" ? "bg-bearish" : "bg-neutral"

  // Format price change
  const priceChange = stockInfo.current_price - stockInfo.previous_close
  const priceChangePct = (priceChange / stockInfo.previous_close) * 100
  const priceChangeColor = priceChange >= 0 ? "text-bullish" : "text-bearish"

  // Format chart data
  const chartData = history?.data?.map((point: any) => ({
    date: point.date,
    price: point.close,
  })) || []

  // Generate rationale
  const rationale = `The AI model suggests a ${signal} because ${prediction.top_bullish_factors.slice(0, 2).join(", ")}. The prediction has ${(prediction.confidence * 100).toFixed(0)}% confidence with ${(prediction.model_agreement * 100).toFixed(0)}% model agreement.`

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-12">
      {/* Navigation */}
      <div className="flex items-center gap-2">
        <Link href="/predictions">
          <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground pl-0 gap-2">
            <ArrowLeft className="h-4 w-4" /> Back to Search
          </Button>
        </Link>
      </div>

      {/* Header Signal Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl md:text-4xl font-display font-bold text-foreground tracking-tight">{decodedSymbol}</h1>
              <p className="text-lg text-muted-foreground mt-1">{stockInfo.name}</p>
            </div>
            <div className="text-right">
              <div className="text-xl md:text-3xl font-body font-bold text-foreground">₹{stockInfo.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              <div className={`font-medium ${priceChangeColor}`}>
                {priceChange >= 0 ? "+" : ""}{priceChangePct.toFixed(2)}% today
              </div>
            </div>
          </div>
        </div>

        <div className="md:col-span-1 flex justify-end">
           <div className={`${signalColor} text-white font-display font-bold text-xl md:text-3xl px-4 py-2 md:px-8 md:py-4 rounded-lg tracking-wider flex items-center gap-2 shadow-lg`}>
              {signal}
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content: Chart & Rationale */}
        <div className="lg:col-span-2 space-y-8">
          {/* Rationale Block */}
          <div className="space-y-3">
            <h3 className="text-lg font-medium text-foreground">Why this signal?</h3>
            <Card className="border-border bg-card shadow-sm">
              <CardContent className="p-6">
                <p className="text-base text-foreground leading-relaxed">
                  {rationale}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Events Section */}
          {events && events.upcoming_events.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-medium text-foreground">Upcoming Events</h3>
              <Card className="border-border bg-card shadow-sm">
                <CardContent className="p-6 space-y-3">
                  {events.upcoming_events.map((event, idx) => (
                    <div key={idx} className="flex items-start gap-3">
                      <Calendar className="h-4 w-4 text-primary mt-0.5" />
                      <div>
                        <div className="font-medium text-foreground">{event.description}</div>
                        <div className="text-sm text-muted-foreground">
                          {event.date} • {event.days_until > 0 ? `In ${event.days_until} days` : event.days_until === 0 ? "Today" : `${Math.abs(event.days_until)} days ago`}
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Chart */}
          {chartData.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-medium text-foreground">Price Trend (6 months)</h3>
              <PriceChart data={chartData} height={350} className="border-border bg-card shadow-sm" />
            </div>
          )}
        </div>

        {/* Sidebar: Metrics */}
        <div className="space-y-6">
          <Card className="border-border bg-card shadow-sm">
            <CardHeader>
              <CardTitle className="text-base font-medium">Key Metrics</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-muted-foreground uppercase">Target (30d)</div>
                <div className="text-lg font-bold text-bullish">₹{prediction.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground uppercase">Potential</div>
                <div className={`text-lg font-bold ${prediction.predicted_return_pct >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                  {prediction.predicted_return_pct >= 0 ? "+" : ""}{prediction.predicted_return_pct.toFixed(2)}%
                </div>
              </div>
              {fundamentals && (
                <>
                  <div>
                    <div className="text-xs text-muted-foreground uppercase">P/E Ratio</div>
                    <div className="text-lg font-medium text-foreground">{fundamentals.pe_ratio ? fundamentals.pe_ratio.toFixed(2) : "N/A"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground uppercase">Market Cap</div>
                    <div className="text-lg font-medium text-foreground">
                      ₹{(stockInfo.market_cap / 1e12).toFixed(2)}T
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <Card className="border-border bg-card shadow-sm">
            <CardHeader>
              <CardTitle className="text-base font-medium">Prediction Confidence</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Model Confidence</span>
                  <span className="text-foreground">{(prediction.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-primary" style={{ width: `${prediction.confidence * 100}%` }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Model Agreement</span>
                  <span className="text-foreground">{(prediction.model_agreement * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-bullish" style={{ width: `${prediction.model_agreement * 100}%` }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Risk Level</span>
                  <span className="text-foreground">{prediction.risk_level}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
