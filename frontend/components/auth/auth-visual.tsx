"use client"

import { cn } from "@/lib/utils"

export function AuthVisual() {
  return (
    <div className="relative w-full max-w-lg aspect-square">
      {/* Abstract Chart Container */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-full h-full">
          {/* Animated Rings */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[80%] border border-primary/10 rounded-full animate-[spin_60s_linear_infinite]" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] border border-primary/20 rounded-full animate-[spin_40s_linear_infinite_reverse]" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40%] h-[40%] border border-primary/30 rounded-full animate-[spin_20s_linear_infinite]" />

          {/* Floating Data Points */}
          <div className="absolute top-[20%] left-[20%] w-3 h-3 bg-primary rounded-full shadow-[0_0_15px_rgba(229,123,60,0.5)] animate-bounce" style={{ animationDuration: '3s' }} />
          <div className="absolute bottom-[30%] right-[25%] w-2 h-2 bg-bullish rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)] animate-bounce" style={{ animationDuration: '4s', animationDelay: '1s' }} />
          <div className="absolute top-[40%] right-[20%] w-2 h-2 bg-bearish rounded-full shadow-[0_0_10px_rgba(239,68,68,0.5)] animate-bounce" style={{ animationDuration: '5s', animationDelay: '0.5s' }} />

          {/* Central Glass Card */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-32 bg-card/80 backdrop-blur-md border border-white/20 rounded-2xl shadow-2xl flex flex-col items-center justify-center p-4">
            <div className="text-xs text-muted-foreground uppercase tracking-widest font-bold mb-1">AI Prediction</div>
            <div className="text-3xl font-display font-bold text-foreground">
              +12.4%
            </div>
            <div className="text-xs font-medium text-bullish mt-1">Confidence: High</div>
          </div>
        </div>
      </div>
    </div>
  )
}
