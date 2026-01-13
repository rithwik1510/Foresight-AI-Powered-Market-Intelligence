"use client"

import * as React from "react"
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts"
import { format } from "date-fns"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface PriceData {
  date: string
  price: number
}

interface PriceChartProps {
  data: PriceData[]
  predictionData?: PriceData[]
  height?: number
  className?: string
}

export function PriceChart({ data, predictionData, height = 400, className }: PriceChartProps) {
  // Combine historical and prediction data if available
  const chartData = React.useMemo(() => {
    if (!predictionData) return data

    // Add a flag to distinguish historical vs predicted
    const historical = data.map(d => ({ ...d, type: 'historical' }))
    const predicted = predictionData.map(d => ({ ...d, type: 'predicted' }))
    
    return [...historical, ...predicted]
  }, [data, predictionData])

  return (
    <Card className={cn("p-4 border-border bg-card", className)}>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#E57B3C" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#E57B3C" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis 
              dataKey="date" 
              stroke="#6B7280" 
              fontSize={12} 
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => {
                try {
                  return format(new Date(value), "MMM d")
                } catch (e) {
                  return value
                }
              }}
            />
            <YAxis 
              stroke="#6B7280" 
              fontSize={12} 
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `₹${value.toLocaleString()}`}
              domain={['auto', 'auto']}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length && label) {
                  return (
                    <div className="rounded-lg border border-border bg-card p-3 shadow-xl">
                      <p className="mb-1 text-xs font-medium text-muted-foreground">
                        {format(new Date(label), "MMM d, yyyy")}
                      </p>
                      <p className="font-display text-lg font-bold text-foreground">
                        ₹{payload[0].value?.toLocaleString()}
                      </p>
                      {payload[0].payload.type === 'predicted' && (
                        <span className="text-xs font-bold text-primary uppercase tracking-wider">
                          Forecast
                        </span>
                      )}
                    </div>
                  )
                }
                return null
              }}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#E57B3C"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorPrice)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
