"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle, AlertTriangle, XCircle, ArrowRight } from "lucide-react"

interface PortfolioReviewProps {
  data: any
}

export function PortfolioReview({ data }: PortfolioReviewProps) {
  if (!data) return null

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-border bg-card shadow-sm">
          <CardContent className="p-6">
            <div className="text-sm text-muted-foreground uppercase tracking-wide">Total Value</div>
            <div className="text-2xl font-bold text-foreground mt-1">₹{data.totalValue.toLocaleString('en-IN')}</div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card shadow-sm">
          <CardContent className="p-6">
            <div className="text-sm text-muted-foreground uppercase tracking-wide">Risk Score</div>
            <div className="text-2xl font-bold text-bearish mt-1">{data.riskScore}</div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card shadow-sm">
          <CardContent className="p-6">
            <div className="text-sm text-muted-foreground uppercase tracking-wide">Allocation</div>
            <div className="text-lg font-medium text-foreground mt-1">
              {data.allocation.equity}% Equity / {data.allocation.debt}% Debt
            </div>
          </CardContent>
        </Card>
      </div>

      {/* The Plain English Review */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card className="border-border bg-card shadow-sm">
            <CardHeader>
              <CardTitle className="text-xl font-display font-medium text-foreground">Portfolio Diagnosis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                {data.issues.map((issue: string, i: number) => (
                  <div key={i} className="flex items-start gap-3 p-4 rounded-lg bg-secondary/50 border border-border">
                    <AlertTriangle className="h-5 w-5 text-neutral shrink-0 mt-0.5" />
                    <p className="text-foreground text-sm leading-relaxed">{issue}</p>
                  </div>
                ))}
              </div>
              
              <div className="pt-6 border-t border-border">
                <h4 className="font-medium text-foreground mb-4">AI Recommendations</h4>
                <ul className="space-y-3">
                  <li className="flex items-center gap-3 text-sm text-muted-foreground">
                    <CheckCircle className="h-4 w-4 text-bullish" />
                    <span>Reduce exposure to Axis Bluechip to lower overlap.</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm text-muted-foreground">
                    <CheckCircle className="h-4 w-4 text-bullish" />
                    <span>Add a Debt Fund to balance your high equity risk.</span>
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Actionable Suggestions */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide pl-1">Suggested Switches</h3>
          
          <Card className="border-border bg-card hover:border-primary/50 transition-colors cursor-pointer group shadow-sm">
            <CardContent className="p-4">
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-bold bg-bearish/10 text-bearish px-2 py-0.5 rounded">SELL</span>
                <span className="text-xs text-muted-foreground">High Overlap</span>
              </div>
              <h4 className="font-medium text-foreground group-hover:text-primary transition-colors">Axis Bluechip Fund</h4>
              <div className="my-3 flex justify-center">
                <ArrowRight className="h-4 w-4 text-muted-foreground rotate-90 md:rotate-0" />
              </div>
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-bold bg-bullish/10 text-bullish px-2 py-0.5 rounded">BUY</span>
                <span className="text-xs text-muted-foreground">Better Diversity</span>
              </div>
              <h4 className="font-medium text-foreground group-hover:text-primary transition-colors">HDFC Balanced Advantage</h4>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}