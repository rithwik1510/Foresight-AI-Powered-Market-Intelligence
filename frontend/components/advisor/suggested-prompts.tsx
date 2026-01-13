"use client"

import { Button } from "@/components/ui/button"
import { Sparkles, TrendingUp, AlertTriangle, PieChart } from "lucide-react"

const PROMPTS = [
  {
    icon: TrendingUp,
    text: "Should I invest in RELIANCE now?",
    category: "Analysis"
  },
  {
    icon: PieChart,
    text: "Compare TCS vs INFY for long-term",
    category: "Comparison"
  },
  {
    icon: AlertTriangle,
    text: "Analyze my portfolio risk",
    category: "Risk"
  },
  {
    icon: Sparkles,
    text: "Suggest high-growth midcap stocks",
    category: "Discovery"
  }
]

interface SuggestedPromptsProps {
  onSelect: (prompt: string) => void
}

export function SuggestedPrompts({ onSelect }: SuggestedPromptsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl mx-auto">
      {PROMPTS.map((prompt, index) => (
        <Button
          key={index}
          variant="outline"
          className="h-auto py-4 px-4 flex justify-start gap-4 border-border bg-card hover:bg-secondary/50 hover:border-primary/50 transition-all text-left whitespace-normal shadow-sm group"
          onClick={() => onSelect(prompt.text)}
        >
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary group-hover:text-primary group-hover:bg-primary/20 transition-colors">
            <prompt.icon className="h-5 w-5" />
          </div>
          <div>
            <div className="font-medium text-foreground">{prompt.text}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{prompt.category}</div>
          </div>
        </Button>
      ))}
    </div>
  )
}