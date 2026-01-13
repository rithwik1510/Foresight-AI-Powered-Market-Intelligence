"use client"

import { MarketTicker } from "@/components/layout/market-ticker"
import { AuthVisual } from "@/components/auth/auth-visual"
import Link from "next/link"
import Image from "next/image"

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen w-full flex flex-col bg-background">
      {/* Top Bar */}
      <div className="h-14 border-b border-border flex items-center px-8 justify-between">
        <Link href="/" className="flex items-center gap-2">
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
        </Link>
        <Link href="/" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
          Back to Home
        </Link>
      </div>

      <div className="flex-1 flex">
        {/* Left: Form Side */}
        <div className="w-full lg:w-[480px] flex flex-col justify-center px-8 md:px-16 py-12 bg-card relative z-10 shadow-xl lg:shadow-none border-r border-border">
          {children}
        </div>

        {/* Right: Visual Side */}
        <div className="hidden lg:flex flex-1 bg-secondary/30 relative overflow-hidden items-center justify-center p-12">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/10 via-background to-background pointer-events-none" />
          <AuthVisual />
        </div>
      </div>
    </div>
  )
}
