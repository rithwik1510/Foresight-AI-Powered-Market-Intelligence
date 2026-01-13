"use client"

import * as React from "react"
import { Upload, Plus, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface FundUploadProps {
  onAnalyze: (data: any) => void
}

export function FundUpload({ onAnalyze }: FundUploadProps) {
  const [loading, setLoading] = React.useState(false)

  const handleSimulate = () => {
    setLoading(true)
    setTimeout(() => {
      onAnalyze({
        totalValue: 540000,
        allocation: { equity: 70, debt: 20, gold: 10 },
        riskScore: "High",
        issues: [
          "High overlap (45%) between Axis Bluechip and SBI Bluechip.",
          "Over-exposure to Financial Services sector (38%).",
          "2 funds have consistently underperformed benchmark for 3 years."
        ]
      })
      setLoading(false)
    }, 1500)
  }

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card className="border-border bg-card shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-foreground">Upload Statement</CardTitle>
          <CardDescription>Upload your CAS (Consolidated Account Statement) PDF or Excel.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-border rounded-lg p-10 hover:bg-secondary/50 transition-colors cursor-pointer bg-background">
            <Upload className="h-8 w-8 text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground text-center">
              Drag & drop your file here, or click to select.
            </p>
            <Button variant="secondary" className="mt-4" onClick={handleSimulate}>
              {loading ? "Analyzing..." : "Select File"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border bg-card shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-foreground">Manual Entry</CardTitle>
          <CardDescription>Add your funds one by one for a quick check.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Fund Name</Label>
            <Input placeholder="e.g. Parag Parikh Flexi Cap" className="bg-background border-border" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Amount (₹)</Label>
              <Input placeholder="10,000" className="bg-background border-border" />
            </div>
            <div className="flex items-end">
              <Button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground" onClick={handleSimulate}>
                <Plus className="h-4 w-4 mr-2" /> Add
              </Button>
            </div>
          </div>
          
          <div className="pt-4 border-t border-border">
             <Button variant="ghost" className="w-full text-primary hover:text-primary/80 hover:bg-primary/10" onClick={handleSimulate}>
                <FileText className="h-4 w-4 mr-2" /> Load Sample Portfolio
             </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}