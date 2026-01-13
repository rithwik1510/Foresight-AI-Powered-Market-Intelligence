"use client"

import * as React from "react"
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface NewsArticle {
  title: string
  source: string
  date: string
  summary?: string
  url?: string
  sentiment?: "bullish" | "bearish" | "neutral"
}

interface NewsListProps {
  articles: NewsArticle[]
  className?: string
}

export function NewsList({ articles, className }: NewsListProps) {
  const [expandedId, setExpandedId] = React.useState<number | null>(null)

  const getSentimentBadge = (sentiment?: string) => {
    if (!sentiment) return null

    const colors = {
      bullish: "bg-bullish/10 text-bullish border-bullish/20",
      bearish: "bg-bearish/10 text-bearish border-bearish/20",
      neutral: "bg-neutral/10 text-neutral border-neutral/20",
    }

    return (
      <span className={cn(
        "px-2 py-0.5 text-xs font-medium rounded border",
        colors[sentiment as keyof typeof colors]
      )}>
        {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
      </span>
    )
  }

  const toggleExpand = (id: number) => {
    setExpandedId(expandedId === id ? null : id)
  }

  if (articles.length === 0) {
    return (
      <Card className={cn("border-border bg-card shadow-sm", className)}>
        <CardContent className="p-12 text-center">
          <p className="text-muted-foreground">No news articles available</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className={cn("space-y-3", className)}>
      {articles.map((article, idx) => {
        const isExpanded = expandedId === idx
        const hasSummary = !!article.summary

        return (
          <Card
            key={idx}
            className="border-border bg-card hover:bg-secondary/50 transition-colors shadow-sm"
          >
            <CardContent className="p-4">
              <div
                className={cn(
                  "flex items-start gap-3",
                  hasSummary && "cursor-pointer"
                )}
                onClick={() => hasSummary && toggleExpand(idx)}
              >
                <div className="flex-1 space-y-1">
                  {/* Title and Sentiment */}
                  <div className="flex items-start justify-between gap-3">
                    <h4 className="font-medium text-foreground leading-snug">
                      {article.title}
                    </h4>
                    {getSentimentBadge(article.sentiment)}
                  </div>

                  {/* Source and Date */}
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{article.source}</span>
                    <span>•</span>
                    <span>{article.date}</span>
                    {article.url && (
                      <>
                        <span>•</span>
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 hover:text-primary transition-colors"
                          onClick={(e) => e.stopPropagation()}
                        >
                          Link <ExternalLink className="h-3 w-3" />
                        </a>
                      </>
                    )}
                  </div>

                  {/* Expandable Summary */}
                  {hasSummary && isExpanded && (
                    <p className="text-sm text-foreground pt-3 border-t border-border mt-3">
                      {article.summary}
                    </p>
                  )}
                </div>

                {/* Expand/Collapse Icon */}
                {hasSummary && (
                  <button className="text-muted-foreground hover:text-foreground transition-colors mt-1">
                    {isExpanded ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                )}
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
