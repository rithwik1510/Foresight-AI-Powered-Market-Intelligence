"use client"

import { ArrowUpRight, ArrowDownRight, Wallet, PieChart, TrendingUp } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function PortfolioStats() {
  // Mock data
  const stats = {
    totalValue: 1245890,
    dailyPnL: 12450,
    dailyPnLPct: 1.01,
    totalReturn: 195890,
    totalReturnPct: 18.65,
    invested: 1050000
  }

  return (
    <Card className="border-border bg-card">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-display font-medium text-foreground">Your Portfolio</h3>
          <ButtonVariant href="/portfolio" />
        </div>

        <div className="space-y-6">
          <div>
            <span className="text-sm text-muted-foreground">Total Value</span>
            <div className="flex items-baseline gap-3 mt-1">
              <span className="text-3xl font-display font-bold text-foreground tracking-tight">
                ₹{stats.totalValue.toLocaleString('en-IN')}
              </span>
              <span className={cn(
                "flex items-center text-sm font-medium",
                stats.dailyPnLPct >= 0 ? "text-bullish" : "text-bearish"
              )}>
                {stats.dailyPnLPct >= 0 ? <ArrowUpRight className="h-4 w-4 mr-0.5" /> : <ArrowDownRight className="h-4 w-4 mr-0.5" />}
                {Math.abs(stats.dailyPnLPct)}%
              </span>
            </div>
            <div className="text-sm text-muted-foreground mt-1">
              +₹{stats.dailyPnL.toLocaleString('en-IN')} today
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
            <div>
              <span className="text-xs text-muted-foreground uppercase tracking-wide">Invested</span>
              <div className="text-lg font-medium text-foreground mt-1">
                ₹{stats.invested.toLocaleString('en-IN')}
              </div>
            </div>
            <div>
              <span className="text-xs text-muted-foreground uppercase tracking-wide">Total Return</span>
              <div className={cn(
                "text-lg font-medium mt-1",
                stats.totalReturn >= 0 ? "text-bullish" : "text-bearish"
              )}>
                +{stats.totalReturn.toLocaleString('en-IN')} ({stats.totalReturnPct}%)
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ButtonVariant({ href }: { href: string }) {
  const Link = require("next/link").default
  return (
    <Link href={href} className="text-xs font-medium text-primary hover:text-primary/80 transition-colors">
      View Details →
    </Link>
  )
}
