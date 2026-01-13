"use client"

import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { DirectionBadge } from "@/components/predictions/direction-badge"
import { cn } from "@/lib/utils"

interface PredictionCardProps {
  symbol: string
  name: string
  currentPrice: number
  targetPrice: number
  direction: "bullish" | "bearish" | "neutral"
  confidence: number
  returnPct: number
  modelAgreement: number
  riskLevel: "LOW" | "MEDIUM" | "HIGH"
}

export function PredictionCard({
  symbol,
  name,
  currentPrice,
  targetPrice,
  direction,
  confidence,
  returnPct,
  modelAgreement,
  riskLevel,
}: PredictionCardProps) {
  const isPositive = returnPct > 0

  return (
    <Card className="group relative overflow-hidden border-border bg-card transition-all hover:border-primary/50 hover:shadow-md">
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
      
      <CardHeader className="flex flex-row items-start justify-between pb-2">
        <div className="space-y-1">
          <CardTitle className="text-xl font-display font-bold text-foreground">
            {symbol}
          </CardTitle>
          <p className="text-sm text-muted-foreground">{name}</p>
        </div>
        <DirectionBadge direction={direction} confidence={confidence} />
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Price Target Section */}
        <div className="flex items-end justify-between border-b border-border pb-6">
          <div className="space-y-1">
            <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Current</span>
            <div className="text-2xl font-body font-medium text-foreground">
              ₹{currentPrice.toLocaleString('en-IN')}
            </div>
          </div>
          
          <ArrowRight className="mb-2 h-5 w-5 text-muted-foreground" />

          <div className="space-y-1 text-right">
            <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Target (30d)</span>
            <div className={cn(
              "text-2xl font-body font-bold flex items-center justify-end gap-2",
              isPositive ? "text-bullish" : "text-bearish"
            )}>
              ₹{targetPrice.toLocaleString('en-IN')}
              <span className="text-sm font-medium bg-secondary px-2 py-0.5 rounded text-foreground">
                {isPositive ? "+" : ""}{returnPct.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase">Model Agreement</span>
            <div className="flex items-center gap-2">
              <div className="h-2 flex-1 rounded-full bg-secondary overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full" 
                  style={{ width: `${modelAgreement * 100}%` }}
                />
              </div>
              <span className="text-sm font-medium text-foreground">{(modelAgreement * 100).toFixed(0)}%</span>
            </div>
          </div>
          
          <div className="space-y-1 text-right">
            <span className="text-xs text-muted-foreground uppercase">Risk Level</span>
            <span className={cn(
              "inline-block px-2 py-0.5 rounded text-xs font-bold",
              riskLevel === "LOW" && "bg-bullish/10 text-bullish",
              riskLevel === "MEDIUM" && "bg-neutral/10 text-neutral",
              riskLevel === "HIGH" && "bg-bearish/10 text-bearish"
            )}>
              {riskLevel}
            </span>
          </div>
        </div>

        <Link href={`/predictions/${symbol}`} className="block">
          <Button className="w-full bg-secondary hover:bg-secondary/80 text-foreground border border-border group-hover:border-primary/30 group-hover:text-primary">
            View Full Analysis
          </Button>
        </Link>
      </CardContent>
    </Card>
  )
}