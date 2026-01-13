"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface ModelInfo {
  name: string
  type: string
  prediction: "bullish" | "bearish" | "neutral"
  confidence: number
  weight: number
  description: string
}

const MODELS: ModelInfo[] = [
  {
    name: "ARIMA",
    type: "Statistical",
    prediction: "bullish",
    confidence: 0.65,
    weight: 15,
    description: "Captures linear trends and momentum"
  },
  {
    name: "Prophet",
    type: "Time Series",
    prediction: "bullish",
    confidence: 0.72,
    weight: 20,
    description: "Analyzes long-term trends and seasonality"
  },
  {
    name: "XGBoost",
    type: "Gradient Boosting",
    prediction: "bullish",
    confidence: 0.78,
    weight: 20,
    description: "Detects non-linear patterns and interactions"
  },
  {
    name: "LightGBM",
    type: "Gradient Boosting",
    prediction: "neutral",
    confidence: 0.55,
    weight: 25,
    description: "Optimized for return magnitude prediction"
  },
  {
    name: "Random Forest",
    type: "Ensemble",
    prediction: "bearish",
    confidence: 0.62,
    weight: 20,
    description: "Robust probability estimation"
  }
]

export function ModelBreakdown() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {MODELS.map((model) => (
        <Card key={model.name} className="border-white/5 bg-black-surface hover:border-orange-500/20 transition-colors">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-medium text-white">{model.name}</CardTitle>
              <Badge 
                variant={model.prediction} 
                className="uppercase text-[10px] tracking-wider"
              >
                {model.prediction}
              </Badge>
            </div>
            <span className="text-xs text-gray-500">{model.type}</span>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-xs text-gray-400 min-h-[2.5em]">
              {model.description}
            </p>
            
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Confidence</span>
                  <span className="text-white font-medium">{(model.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                  <div 
                    className={cn(
                      "h-full rounded-full transition-all",
                      model.prediction === 'bullish' && "bg-bullish",
                      model.prediction === 'bearish' && "bg-bearish",
                      model.prediction === 'neutral' && "bg-neutral"
                    )}
                    style={{ width: `${model.confidence * 100}%` }}
                  />
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Ensemble Weight</span>
                  <span className="text-orange-500 font-medium">{model.weight}%</span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                  <div 
                    className="h-full bg-orange-500 rounded-full" 
                    style={{ width: `${model.weight}%` }}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
