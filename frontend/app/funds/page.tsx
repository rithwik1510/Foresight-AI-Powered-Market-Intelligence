"use client"

import * as React from "react"
import { FundUpload } from "@/components/funds/fund-upload"
import { PortfolioReview } from "@/components/funds/portfolio-review"

export default function FundsPage() {
  const [portfolioData, setPortfolioData] = React.useState<any>(null)

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-12">
      <div className="space-y-2">
        <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">Mutual Fund Analyzer</h1>
        <p className="text-muted-foreground">
          Get a professional health check of your mutual fund portfolio. Identify risks, overlaps, and better opportunities.
        </p>
      </div>

      {!portfolioData ? (
        <div className="max-w-3xl">
          <FundUpload onAnalyze={setPortfolioData} />
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-medium text-foreground">Analysis Report</h2>
            <button 
              onClick={() => setPortfolioData(null)}
              className="text-sm text-primary hover:underline"
            >
              Analyze Another
            </button>
          </div>
          <PortfolioReview data={portfolioData} />
        </div>
      )}
    </div>
  )
}