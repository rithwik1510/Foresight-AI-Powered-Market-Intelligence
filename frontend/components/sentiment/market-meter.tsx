"use client"

import { cn } from "@/lib/utils"

interface MarketMeterProps {
  value: number // 0-100, where 0 = Fear, 100 = Greed
  label?: string
  className?: string
}

export function MarketMeter({ value, label, className }: MarketMeterProps) {
  // Clamp value between 0-100
  const clampedValue = Math.max(0, Math.min(100, value))

  // Determine color based on value
  const getColor = (val: number) => {
    if (val < 25) return "text-bearish" // Extreme Fear
    if (val < 45) return "text-orange-500" // Fear
    if (val >= 55 && val < 75) return "text-green-500" // Greed
    if (val >= 75) return "text-bullish" // Extreme Greed
    return "text-neutral" // Neutral
  }

  const getBarColor = (val: number) => {
    if (val < 25) return "bg-bearish" // Extreme Fear
    if (val < 45) return "bg-orange-500" // Fear
    if (val >= 55 && val < 75) return "bg-green-500" // Greed
    if (val >= 75) return "bg-bullish" // Extreme Greed
    return "bg-neutral" // Neutral
  }

  const labelText = label || (
    clampedValue < 25 ? "Extreme Fear" :
    clampedValue < 45 ? "Fear" :
    clampedValue >= 55 && clampedValue < 75 ? "Greed" :
    clampedValue >= 75 ? "Extreme Greed" :
    "Neutral"
  )

  return (
    <div className={cn("space-y-3", className)}>
      {/* Value and Label */}
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-muted-foreground">Market Sentiment</span>
        <span className={cn("text-2xl font-display font-bold", getColor(clampedValue))}>
          {labelText} ({clampedValue})
        </span>
      </div>

      {/* Meter Bar */}
      <div className="relative">
        {/* Background bar */}
        <div className="h-3 w-full bg-secondary rounded-full overflow-hidden relative">
          {/* Gradient fill */}
          <div
            className="h-full transition-all duration-500 ease-out"
            style={{
              width: `${clampedValue}%`,
              background: clampedValue < 50
                ? `linear-gradient(to right, #DC2626, #F59E0B)`
                : `linear-gradient(to right, #F59E0B, #059669)`
            }}
          />
        </div>

        {/* Triangle marker */}
        <div
          className="absolute -top-1 transform -translate-x-1/2 transition-all duration-500"
          style={{ left: `${clampedValue}%` }}
        >
          <div className={cn("w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-transparent",
            getColor(clampedValue).replace('text-', 'border-t-')
          )} />
        </div>
      </div>

      {/* Labels */}
      <div className="flex justify-between text-xs text-muted-foreground font-medium">
        <span>Fear (0)</span>
        <span className="text-neutral">Neutral (50)</span>
        <span>Greed (100)</span>
      </div>
    </div>
  )
}
