"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PriceChart } from "@/components/charts/price-chart"
import { ModelBreakdown } from "@/components/predictions/model-breakdown"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2, XCircle, AlertCircle } from "lucide-react"

export function PredictionTabs({ symbol }: { symbol: string }) {
  // Mock data for chart
  const historicalData = Array.from({ length: 30 }, (_, i) => ({
    date: new Date(Date.now() - (30 - i) * 24 * 60 * 60 * 1000).toISOString(),
    price: 2400 + Math.random() * 200 + (i * 5)
  }))

  const predictionData = Array.from({ length: 30 }, (_, i) => ({
    date: new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000).toISOString(),
    price: 2550 + Math.random() * 50 + (i * 2)
  }))

  return (
    <Tabs defaultValue="overview" className="w-full space-y-6">
      <TabsList className="bg-black-surface border border-white/5 p-1">
        <TabsTrigger value="overview" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">
          Overview
        </TabsTrigger>
        <TabsTrigger value="models" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">
          Models Analysis
        </TabsTrigger>
        <TabsTrigger value="sentiment" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">
          Sentiment
        </TabsTrigger>
      </TabsList>

      <TabsContent value="overview" className="space-y-6 animate-fadeIn">
        <PriceChart 
          data={historicalData} 
          predictionData={predictionData} 
          height={400} 
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="border-white/5 bg-black-surface">
            <CardHeader>
              <CardTitle className="text-base text-bullish flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5" /> Bullish Factors
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                <li className="flex items-start gap-2 text-sm text-gray-300">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bullish shrink-0" />
                  Strong RSI momentum divergence indicating upward pressure
                </li>
                <li className="flex items-start gap-2 text-sm text-gray-300">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bullish shrink-0" />
                  Positive news sentiment (0.42 score) from major sources
                </li>
                <li className="flex items-start gap-2 text-sm text-gray-300">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bullish shrink-0" />
                  Price trading above 50-day SMA support level
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-white/5 bg-black-surface">
            <CardHeader>
              <CardTitle className="text-base text-bearish flex items-center gap-2">
                <XCircle className="h-5 w-5" /> Bearish Factors
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                <li className="flex items-start gap-2 text-sm text-gray-300">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bearish shrink-0" />
                  High P/E ratio (28x) relative to sector average
                </li>
                <li className="flex items-start gap-2 text-sm text-gray-300">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bearish shrink-0" />
                  USD/INR weakness potentially affecting import costs
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </TabsContent>

      <TabsContent value="models" className="space-y-6 animate-fadeIn">
        <div className="mb-4">
          <h3 className="text-lg font-display font-medium text-white">Ensemble Model Breakdown</h3>
          <p className="text-gray-400 text-sm">
            Our prediction is a weighted average of 5 institutional-grade models. 
            High agreement ({">"}70%) indicates a stronger signal.
          </p>
        </div>
        <ModelBreakdown />
      </TabsContent>

      <TabsContent value="sentiment" className="animate-fadeIn">
        <Card className="border-white/5 bg-black-surface p-12 text-center">
          <AlertCircle className="h-12 w-12 text-orange-500 mx-auto mb-4" />
          <h3 className="text-xl font-display font-bold text-white">Sentiment Analysis</h3>
          <p className="text-gray-400 mt-2">
            Detailed news and social sentiment analysis will be available here.
          </p>
        </Card>
      </TabsContent>
    </Tabs>
  )
}
