"use client"

import * as React from "react"
import { ExternalLink, MessageSquare } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatDistanceToNow } from "date-fns"

interface NewsItem {
  id: string
  title: string
  source: string
  time: string
  sentiment: "positive" | "negative" | "neutral"
  url: string
  type: "news" | "social"
}

const MOCK_NEWS: NewsItem[] = [
  {
    id: "1",
    title: "Reliance Industries Q3 results beat estimates, profit jumps 11%",
    source: "Economic Times",
    time: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 mins ago
    sentiment: "positive",
    url: "#",
    type: "news"
  },
  {
    id: "2",
    title: "Oil prices dip slightly on global demand concerns",
    source: "Moneycontrol",
    time: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
    sentiment: "negative",
    url: "#",
    type: "news"
  },
  {
    id: "3",
    title: "Is RELIANCE a good buy at current levels? Detailed analysis inside.",
    source: "r/IndiaInvestments",
    time: new Date(Date.now() - 1000 * 60 * 60 * 4).toISOString(), // 4 hours ago
    sentiment: "neutral",
    url: "#",
    type: "social"
  },
  {
    id: "4",
    title: "Jio Financial Services plans major expansion in lending sector",
    source: "Livemint",
    time: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(), // 5 hours ago
    sentiment: "positive",
    url: "#",
    type: "news"
  },
  {
    id: "5",
    title: "Market volatility expected to continue ahead of budget",
    source: "Business Standard",
    time: new Date(Date.now() - 1000 * 60 * 60 * 8).toISOString(), // 8 hours ago
    sentiment: "negative",
    url: "#",
    type: "news"
  }
]

export function NewsFeed({ type = "all" }: { type?: "all" | "news" | "social" }) {
  const filteredNews = type === "all" ? MOCK_NEWS : MOCK_NEWS.filter(item => item.type === type)

  return (
    <div className="space-y-4">
      {filteredNews.map((item) => (
        <Card key={item.id} className="border-white/5 bg-black-surface hover:border-orange-500/20 transition-all group">
          <CardContent className="p-4">
            <div className="flex justify-between items-start gap-4">
              <div className="space-y-2 flex-1">
                <div className="flex items-center gap-2">
                  <Badge 
                    variant={item.sentiment === "positive" ? "bullish" : item.sentiment === "negative" ? "bearish" : "neutral"}
                    className="h-5 px-1.5 text-[10px] uppercase"
                  >
                    {item.sentiment}
                  </Badge>
                  <span className="text-xs text-gray-500 flex items-center gap-1">
                    {item.type === "social" && <MessageSquare className="h-3 w-3" />}
                    {item.source} • {formatDistanceToNow(new Date(item.time), { addSuffix: true })}
                  </span>
                </div>
                <h4 className="font-medium text-gray-200 group-hover:text-orange-500 transition-colors leading-snug">
                  {item.title}
                </h4>
              </div>
              <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-white shrink-0">
                <ExternalLink className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}
      <Button variant="outline" className="w-full border-white/10 bg-white/5 hover:bg-white/10 text-gray-400">
        Load More
      </Button>
    </div>
  )
}
