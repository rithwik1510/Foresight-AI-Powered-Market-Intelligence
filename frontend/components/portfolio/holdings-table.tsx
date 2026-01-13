"use client"

import * as React from "react"
import { MoreHorizontal, ArrowUpRight, ArrowDownRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface Holding {
  id: string
  symbol: string
  name: string
  type: "STOCK" | "MF"
  quantity: number
  avgPrice: number
  currentPrice: number
  invested: number
  currentValue: number
  pnl: number
  pnlPercent: number
}

const MOCK_HOLDINGS: Holding[] = [
  {
    id: "1",
    symbol: "RELIANCE.NS",
    name: "Reliance Industries Ltd",
    type: "STOCK",
    quantity: 50,
    avgPrice: 2100.00,
    currentPrice: 2450.50,
    invested: 105000,
    currentValue: 122525,
    pnl: 17525,
    pnlPercent: 16.69
  },
  {
    id: "2",
    symbol: "TCS.NS",
    name: "Tata Consultancy Services",
    type: "STOCK",
    quantity: 30,
    avgPrice: 3200.00,
    currentPrice: 3890.00,
    invested: 96000,
    currentValue: 116700,
    pnl: 20700,
    pnlPercent: 21.56
  },
  {
    id: "3",
    symbol: "HDFCBANK.NS",
    name: "HDFC Bank Ltd",
    type: "STOCK",
    quantity: 40,
    avgPrice: 1580.00,
    currentPrice: 1450.00,
    invested: 63200,
    currentValue: 58000,
    pnl: -5200,
    pnlPercent: -8.23
  },
  {
    id: "4",
    symbol: "SBI BLUECHIP",
    name: "SBI Bluechip Fund Direct Growth",
    type: "MF",
    quantity: 1250.5,
    avgPrice: 85.00,
    currentPrice: 98.50,
    invested: 106292.5,
    currentValue: 123174.25,
    pnl: 16881.75,
    pnlPercent: 15.88
  }
]

export function HoldingsTable() {
  return (
    <div className="w-full overflow-hidden rounded-xl border border-border bg-card">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-secondary text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-6 py-4 font-medium tracking-wider">Asset</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">Qty</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">Avg Price</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">Current</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">Invested</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">Current Value</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">P&L</th>
              <th className="px-6 py-4 font-medium tracking-wider text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {MOCK_HOLDINGS.map((holding) => {
              const isProfit = holding.pnl >= 0
              return (
                <tr key={holding.id} className="group hover:bg-secondary/50 transition-colors">
                  <td className="px-6 py-4">
                    <div className="flex flex-col">
                      <span className="font-display font-medium text-foreground">{holding.symbol}</span>
                      <span className="text-xs text-muted-foreground">{holding.name}</span>
                      <div className="mt-1">
                        <Badge variant="secondary" className="text-[10px] h-4 px-1">
                          {holding.type}
                        </Badge>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right font-medium text-foreground">
                    {holding.quantity.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-right text-muted-foreground">
                    ₹{holding.avgPrice.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-6 py-4 text-right font-medium text-foreground">
                    ₹{holding.currentPrice.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-6 py-4 text-right text-muted-foreground">
                    ₹{holding.invested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td className="px-6 py-4 text-right font-medium text-foreground">
                    ₹{holding.currentValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className={cn("flex flex-col items-end", isProfit ? "text-bullish" : "text-bearish")}>
                      <span className="font-medium flex items-center">
                        {isProfit ? "+" : ""}₹{holding.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </span>
                      <span className="text-xs flex items-center bg-secondary px-1 rounded mt-0.5">
                        {isProfit ? <ArrowUpRight className="h-3 w-3 mr-0.5" /> : <ArrowDownRight className="h-3 w-3 mr-0.5" />}
                        {Math.abs(holding.pnlPercent).toFixed(2)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
