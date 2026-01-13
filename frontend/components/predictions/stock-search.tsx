"use client"

import * as React from "react"
import { Search, Loader2 } from "lucide-react"
import { useRouter } from "next/navigation"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

// Mock data - replace with API call later
const STOCK_SUGGESTIONS = [
  { symbol: "RELIANCE.NS", name: "Reliance Industries Ltd." },
  { symbol: "TCS.NS", name: "Tata Consultancy Services" },
  { symbol: "HDFCBANK.NS", name: "HDFC Bank Ltd." },
  { symbol: "INFY.NS", name: "Infosys Limited" },
  { symbol: "ICICIBANK.NS", name: "ICICI Bank Ltd." },
]

export function StockSearch() {
  const router = useRouter()
  const [query, setQuery] = React.useState("")
  const [isOpen, setIsOpen] = React.useState(false)
  const [isLoading, setIsLoading] = React.useState(false)
  const containerRef = React.useRef<HTMLDivElement>(null)

  // Filter suggestions based on query
  const filteredSuggestions = STOCK_SUGGESTIONS.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
      stock.name.toLowerCase().includes(query.toLowerCase())
  )

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query) {
      router.push(`/predictions/${query.toUpperCase()}`)
      setIsOpen(false)
    }
  }

  const handleSelect = (symbol: string) => {
    setQuery(symbol)
    router.push(`/predictions/${symbol}`)
    setIsOpen(false)
  }

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  return (
    <div ref={containerRef} className="relative w-full max-w-2xl">
      <form onSubmit={handleSearch} className="relative">
        <Search className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search for a stock (e.g., RELIANCE.NS)..."
          className="h-12 pl-10 pr-4 text-base bg-card border-border focus-visible:ring-primary/50 text-foreground placeholder:text-muted-foreground shadow-sm"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setIsOpen(true)
          }}
          onFocus={() => setIsOpen(true)}
        />
        <Button 
          type="submit" 
          className="absolute right-1 top-1 h-10 bg-primary hover:bg-primary/90 text-primary-foreground"
        >
          Analyze
        </Button>
      </form>

      {isOpen && query.length > 0 && (
        <Card className="absolute top-full z-50 mt-2 w-full overflow-hidden border-border bg-popover p-0 shadow-lg">
          {isLoading ? (
            <div className="flex h-20 items-center justify-center">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
            </div>
          ) : filteredSuggestions.length > 0 ? (
            <ul className="max-h-[300px] overflow-auto py-2 bg-popover">
              {filteredSuggestions.map((stock) => (
                <li key={stock.symbol}>
                  <button
                    onClick={() => handleSelect(stock.symbol)}
                    className="flex w-full flex-col items-start px-4 py-2 hover:bg-secondary text-left transition-colors"
                  >
                    <span className="font-display font-medium text-foreground">
                      {stock.symbol}
                    </span>
                    <span className="text-sm text-muted-foreground truncate w-full">
                      {stock.name}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="p-4 text-center text-sm text-muted-foreground bg-popover">
              No results found. Try entering a symbol directly.
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
