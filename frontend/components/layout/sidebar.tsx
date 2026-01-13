"use client"

import * as React from "react"
import Link from "next/link"
import Image from "next/image"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  LayoutDashboard,
  LineChart,
  PieChart,
  Bot,
  Settings,
  Menu,
  ChevronLeft,
  Search
} from "lucide-react"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = React.useState(false)

  const navItems = [
    {
      title: "Dashboard",
      href: "/",
      icon: LayoutDashboard,
    },
    {
      title: "Stocks",
      href: "/predictions",
      icon: LineChart,
    },
    {
      title: "Mutual Funds",
      href: "/funds",
      icon: PieChart,
    },
    {
      title: "AI Advisor",
      href: "/advisor",
      icon: Bot,
    },
  ]

  return (
    <div
      className={cn(
        "relative flex flex-col border-r border-border bg-card transition-all duration-200 ease-in-out shadow-sm",
        collapsed ? "w-[72px]" : "w-[240px]",
        className
      )}
    >
      {/* Header */}
      <div className="flex h-14 items-center border-b border-border px-4">
        {collapsed ? (
          <div className="flex items-center justify-center w-full">
            <Image
              src="/logos/foresight-favicon.svg"
              alt="Foresight"
              width={28}
              height={28}
              className="shrink-0"
            />
          </div>
        ) : (
          <div className="flex items-center gap-2 overflow-hidden transition-all w-auto opacity-100">
            <Image
              src="/logos/foresight-favicon.svg"
              alt="Foresight"
              width={28}
              height={28}
              className="shrink-0"
            />
            <span className="font-display text-base font-bold tracking-tight bg-gradient-to-r from-orange-400 via-orange-500 to-orange-600 bg-clip-text text-transparent whitespace-nowrap">
              Foresight
            </span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className={cn("ml-auto text-muted-foreground hover:text-foreground h-8 w-8", collapsed && "absolute right-2 top-3")}
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed ? <Menu className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-4">
        <nav className="grid gap-1 px-2">
          {navItems.map((item, index) => {
            const isActive = pathname === item.href || (item.href !== "/" && pathname?.startsWith(item.href))
            return (
              <Link
                key={index}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-secondary text-foreground font-semibold"
                    : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground",
                  collapsed && "justify-center px-2"
                )}
              >
                <item.icon className={cn("h-4 w-4 shrink-0", isActive ? "text-primary" : "text-muted-foreground")} />
                {!collapsed && <span>{item.title}</span>}
              </Link>
            )
          })}
        </nav>
      </div>

      {/* Footer */}
      <div className="border-t border-border p-2">
        <Link
          href="/settings"
          className={cn(
            "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary/50 hover:text-foreground",
            collapsed && "justify-center px-2"
          )}
        >
          <Settings className="h-4 w-4 shrink-0" />
          {!collapsed && <span>Settings</span>}
        </Link>
      </div>
    </div>
  )
}
