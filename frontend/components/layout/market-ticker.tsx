"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { dashboardApi } from "@/lib/api/dashboard"

interface MarketItem {
  symbol: string
  price: string
  change: string
  changePercent: string
  isPositive: boolean
}

const SAMPLE_DATA: MarketItem[] = [
  { symbol: "NIFTY 50", price: "25,500.00", change: "+125.00", changePercent: "+0.49%", isPositive: true },
  { symbol: "SENSEX", price: "84,180.00", change: "+380.00", changePercent: "+0.45%", isPositive: true },
  { symbol: "USD/INR", price: "86.50", change: "+0.12", changePercent: "+0.14%", isPositive: true },
  { symbol: "S&P 500", price: "6,050.00", change: "+28.00", changePercent: "+0.46%", isPositive: true },
  { symbol: "NASDAQ", price: "19,850.00", change: "+145.00", changePercent: "+0.74%", isPositive: true },
  { symbol: "GOLD", price: "2,680.00", change: "+12.00", changePercent: "+0.45%", isPositive: true },
  { symbol: "CRUDE OIL", price: "78.50", change: "-0.85", changePercent: "-1.07%", isPositive: false },
]

export function MarketTicker() {
  const [marketData, setMarketData] = React.useState<MarketItem[]>(SAMPLE_DATA)

  React.useEffect(() => {
    const fetchMarketData = async () => {
      try {
        // Fetch all data in parallel
        const [indices, globalFactors] = await Promise.all([
          dashboardApi.getTrendingStocks().catch(() => null),
          dashboardApi.getGlobalFactors().catch(() => null),
        ])

        const newData: MarketItem[] = []

        // Nifty 50
        if (indices?.nifty) {
          const changePct = ((indices.nifty.current_price - indices.nifty.previous_close) / indices.nifty.previous_close * 100)
          const change = indices.nifty.current_price - indices.nifty.previous_close
          newData.push({
            symbol: "NIFTY 50",
            price: indices.nifty.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
            change: `${change >= 0 ? '+' : ''}${change.toFixed(2)}`,
            changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
            isPositive: changePct >= 0
          })
        }

        // Sensex
        if (indices?.sensex) {
          const changePct = ((indices.sensex.current_price - indices.sensex.previous_close) / indices.sensex.previous_close * 100)
          const change = indices.sensex.current_price - indices.sensex.previous_close
          newData.push({
            symbol: "SENSEX",
            price: indices.sensex.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
            change: `${change >= 0 ? '+' : ''}${change.toFixed(2)}`,
            changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
            isPositive: changePct >= 0
          })
        }

        // Global factors
        if (globalFactors) {
          // S&P 500
          if (globalFactors.us_markets?.sp500) {
            const changePct = globalFactors.us_markets.sp500.change_1d * 100 // Convert decimal to percentage
            newData.push({
              symbol: "S&P 500",
              price: globalFactors.us_markets.sp500.price.toLocaleString('en-US', { minimumFractionDigits: 2 }),
              change: "",
              changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
              isPositive: changePct >= 0
            })
          }

          // NASDAQ
          if (globalFactors.us_markets?.nasdaq) {
            const changePct = globalFactors.us_markets.nasdaq.change_1d * 100
            newData.push({
              symbol: "NASDAQ",
              price: globalFactors.us_markets.nasdaq.price.toLocaleString('en-US', { minimumFractionDigits: 2 }),
              change: "",
              changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
              isPositive: changePct >= 0
            })
          }

          // USD/INR
          if (globalFactors.forex?.usdinr) {
            const changePct = globalFactors.forex.usdinr.change_1d * 100
            newData.push({
              symbol: "USD/INR",
              price: globalFactors.forex.usdinr.rate.toFixed(2),
              change: "",
              changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
              isPositive: changePct >= 0
            })
          }

          // Gold
          if (globalFactors.commodities?.gold) {
            const changePct = globalFactors.commodities.gold.change_1d * 100
            newData.push({
              symbol: "GOLD",
              price: globalFactors.commodities.gold.price.toLocaleString('en-US', { minimumFractionDigits: 2 }),
              change: "",
              changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
              isPositive: changePct >= 0
            })
          }

          // Crude Oil (Brent)
          if (globalFactors.commodities?.oil_brent) {
            const changePct = globalFactors.commodities.oil_brent.change_1d * 100
            newData.push({
              symbol: "CRUDE OIL",
              price: globalFactors.commodities.oil_brent.price.toLocaleString('en-US', { minimumFractionDigits: 2 }),
              change: "",
              changePercent: `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`,
              isPositive: changePct >= 0
            })
          }
        }

        // Only update if we got some real data
        if (newData.length > 0) {
          setMarketData(newData)
        }
      } catch (error) {
        // Silently fall back to mock data
        console.log("Using mock market data")
      }
    }

    // Fetch immediately
    fetchMarketData()

    // Refresh every 5 minutes
    const interval = setInterval(fetchMarketData, 5 * 60 * 1000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="w-full overflow-hidden bg-card border-b border-border h-9 flex items-center">
      <div className="flex animate-scroll whitespace-nowrap hover:[animation-play-state:paused]">
        {[...marketData, ...marketData].map((item, i) => (
          <div key={i} className="flex items-center gap-3 px-4 border-r border-border last:border-r-0">
            <span className="text-[11px] font-bold text-muted-foreground font-display tracking-wide">{item.symbol}</span>
            <div className="flex items-center gap-2 text-[11px] font-medium font-body">
              <span className="text-foreground">{item.price}</span>
              <span className={cn(item.isPositive ? "text-bullish" : "text-bearish")}>
                {item.changePercent}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}