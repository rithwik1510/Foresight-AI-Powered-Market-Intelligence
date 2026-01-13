"use client"

import * as React from "react"
import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"

export function AddHoldingModal() {
  const [open, setOpen] = React.useState(false)
  const [type, setType] = React.useState("STOCK")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // TODO: Implement API call
    console.log("Submitting holding...")
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="bg-orange-600 hover:bg-orange-700 text-white gap-2">
          <Plus className="h-4 w-4" /> Add Holding
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px] bg-black-elevated border-white/10 text-white">
        <DialogHeader>
          <DialogTitle>Add New Holding</DialogTitle>
          <DialogDescription className="text-gray-400">
            Enter the details of your investment to track it in your portfolio.
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4 pt-4">
          <Tabs defaultValue="STOCK" onValueChange={setType} className="w-full">
            <TabsList className="grid w-full grid-cols-2 bg-black-surface border border-white/10">
              <TabsTrigger value="STOCK" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">Stock</TabsTrigger>
              <TabsTrigger value="MF" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">Mutual Fund</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="space-y-2">
            <Label htmlFor="symbol">Symbol / Scheme Code</Label>
            <Input id="symbol" placeholder={type === "STOCK" ? "e.g., RELIANCE.NS" : "e.g., 123456"} className="bg-black-surface border-white/10 focus-visible:ring-orange-500/50" required />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="quantity">Quantity</Label>
              <Input id="quantity" type="number" step="any" placeholder="0.00" className="bg-black-surface border-white/10 focus-visible:ring-orange-500/50" required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="price">Avg. Buy Price</Label>
              <Input id="price" type="number" step="any" placeholder="₹0.00" className="bg-black-surface border-white/10 focus-visible:ring-orange-500/50" required />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="date">Purchase Date</Label>
            <Input id="date" type="date" className="bg-black-surface border-white/10 focus-visible:ring-orange-500/50" required />
          </div>

          <DialogFooter className="pt-4">
            <Button type="submit" className="bg-orange-600 hover:bg-orange-700 text-white w-full">
              Add to Portfolio
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
