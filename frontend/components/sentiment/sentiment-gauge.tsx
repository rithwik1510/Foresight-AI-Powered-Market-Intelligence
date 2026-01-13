"use client"

import { cn } from "@/lib/utils"

interface SentimentGaugeProps {
  score: number // -1 (Bearish) to 1 (Bullish)
  label: string
  className?: string
}

export function SentimentGauge({ score, label, className }: SentimentGaugeProps) {
  // Map score -1 to 1 => 0 to 180 degrees
  // -1 => 0 deg
  // 0 => 90 deg
  // 1 => 180 deg
  const rotation = (score + 1) * 90

  // Determine color based on score
  let colorClass = "text-neutral"
  let colorHex = "#F59E0B"
  
  if (score > 0.2) {
    colorClass = "text-bullish"
    colorHex = "#22C55E"
  } else if (score < -0.2) {
    colorClass = "text-bearish"
    colorHex = "#EF4444"
  }

  return (
    <div className={cn("relative flex flex-col items-center justify-center p-6", className)}>
      {/* Semi-circle Gauge */}
      <div className="relative h-32 w-64 overflow-hidden">
        <div className="absolute top-0 left-0 h-full w-full rounded-t-full bg-white/5 border border-white/10 border-b-0"></div>
        
        {/* Needle */}
        <div 
          className="absolute bottom-0 left-1/2 h-full w-1 origin-bottom bg-white transition-all duration-1000 ease-out"
          style={{ transform: `translateX(-50%) rotate(${rotation - 90}deg)` }}
        >
          <div className="absolute -top-1 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)]" />
        </div>
        
        {/* Hub */}
        <div className="absolute bottom-0 left-1/2 h-4 w-8 -translate-x-1/2 rounded-t-full bg-black-surface border-t border-white/20" />
      </div>

      {/* Labels */}
      <div className="mt-4 text-center space-y-1">
        <div className={cn("text-3xl font-display font-bold", colorClass)}>
          {score > 0 ? "+" : ""}{score.toFixed(2)}
        </div>
        <div className="text-sm font-medium text-gray-400 tracking-widest uppercase">
          {label}
        </div>
      </div>

      {/* Axis Labels */}
      <div className="absolute bottom-16 w-full flex justify-between px-8 text-xs font-bold text-gray-600 uppercase">
        <span>Bearish</span>
        <span>Neutral</span>
        <span>Bullish</span>
      </div>
    </div>
  )
}
