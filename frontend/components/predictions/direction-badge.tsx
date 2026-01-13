import * as React from "react"
import { ArrowUp, ArrowDown, Minus } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface DirectionBadgeProps {
  direction: "bullish" | "bearish" | "neutral"
  confidence: number
  className?: string
}

export function DirectionBadge({ direction, confidence, className }: DirectionBadgeProps) {
  const config = {
    bullish: {
      icon: ArrowUp,
      variant: "bullish" as const,
      label: "BULLISH",
    },
    bearish: {
      icon: ArrowDown,
      variant: "bearish" as const,
      label: "BEARISH",
    },
    neutral: {
      icon: Minus,
      variant: "neutral" as const,
      label: "NEUTRAL",
    },
  }

  const { icon: Icon, variant, label } = config[direction]

  return (
    <div className={cn("flex flex-col items-center gap-1", className)}>
      <Badge variant={variant} className="px-3 py-1 text-sm font-medium tracking-wide">
        <Icon className="mr-1 h-3.5 w-3.5" />
        {label}
      </Badge>
      <span className={cn(
        "text-xs font-medium",
        direction === "bullish" && "text-bullish",
        direction === "bearish" && "text-bearish",
        direction === "neutral" && "text-neutral"
      )}>
        {(confidence * 100).toFixed(0)}% Confidence
      </span>
    </div>
  )
}
