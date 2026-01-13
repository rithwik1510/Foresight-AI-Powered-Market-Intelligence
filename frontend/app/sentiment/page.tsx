import { MarketMeter } from "@/components/sentiment/market-meter"
import { NewsFeed } from "@/components/sentiment/news-feed"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function SentimentPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-12 pb-12">
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">Market Sentiment</h1>
        <p className="text-muted-foreground max-w-xl mx-auto">
          Understand the overall mood of the Indian market based on news, volatility, and social signals.
        </p>
      </div>

      <div className="max-w-2xl mx-auto">
        <Card className="border-border bg-card shadow-sm">
          <CardContent className="p-8">
            <MarketMeter score={68} />
          </CardContent>
        </Card>
      </div>

      <div className="space-y-6">
        <h2 className="text-xl font-display font-bold text-foreground">Key Drivers</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
           <NewsFeed />
        </div>
      </div>
    </div>
  )
}
