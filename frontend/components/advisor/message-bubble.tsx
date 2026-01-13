"use client"

import * as React from "react"
import { Bot, User } from "lucide-react"
import { cn } from "@/lib/utils"
import ReactMarkdown from 'react-markdown'

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div className={cn("flex w-full gap-4", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 border border-primary/20">
          <Bot className="h-5 w-5 text-primary" />
        </div>
      )}

      <div
        className={cn(
          "relative max-w-[80%] rounded-2xl px-5 py-3 text-sm shadow-sm",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-none"
            : "bg-card border border-border text-foreground rounded-tl-none"
        )}
      >
        <div className={cn("prose prose-sm max-w-none break-words", isUser ? "prose-invert" : "text-foreground")}>
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        <span className={cn(
            "text-[10px] opacity-70 mt-1 block",
            isUser ? "text-primary-foreground/80" : "text-muted-foreground"
        )}>
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary border border-border">
          <User className="h-5 w-5 text-muted-foreground" />
        </div>
      )}
    </div>
  )
}