import { PortfolioStats } from "@/components/portfolio/portfolio-stats"
import { HoldingsTable } from "@/components/portfolio/holdings-table"
import { AddHoldingModal } from "@/components/portfolio/add-holding-modal"
import { PriceChart } from "@/components/charts/price-chart"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { PieChart, Download } from "lucide-react"

export default function PortfolioPage() {
  // Mock performance data
  const performanceData = Array.from({ length: 30 }, (_, i) => ({
    date: new Date(Date.now() - (30 - i) * 24 * 60 * 60 * 1000).toISOString(),
    price: 1000000 + Math.random() * 50000 + (i * 2000)
  }))

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">My Portfolio</h1>
          <p className="text-muted-foreground">Track and manage your investments across stocks and mutual funds.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" className="border-border bg-secondary hover:bg-secondary/80 text-muted-foreground">
            <Download className="h-4 w-4 mr-2" /> Export Report
          </Button>
          <AddHoldingModal />
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <PortfolioStats />
        </div>
        <div className="lg:col-span-2">
          <div className="rounded-xl border border-border bg-card p-4 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4 px-2">
              <h3 className="font-medium text-foreground">Portfolio Performance</h3>
              <div className="flex gap-2">
                <span className="text-xs font-bold px-2 py-1 rounded bg-primary/10 text-primary">1M</span>
                <span className="text-xs font-medium px-2 py-1 rounded hover:bg-secondary text-muted-foreground cursor-pointer">6M</span>
                <span className="text-xs font-medium px-2 py-1 rounded hover:bg-secondary text-muted-foreground cursor-pointer">1Y</span>
              </div>
            </div>
            <div className="flex-1 min-h-[200px]">
              <PriceChart data={performanceData} height={250} className="border-0 bg-transparent p-0" />
            </div>
          </div>
        </div>
      </div>

      {/* Holdings Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-display font-bold text-foreground">Holdings</h2>
          <Tabs defaultValue="all" className="w-[300px]">
            <TabsList className="grid w-full grid-cols-3 bg-secondary border border-border">
              <TabsTrigger value="all" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-xs">All</TabsTrigger>
              <TabsTrigger value="stocks" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-xs">Stocks</TabsTrigger>
              <TabsTrigger value="funds" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-xs">Funds</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        <HoldingsTable />
      </div>
    </div>
  )
}
