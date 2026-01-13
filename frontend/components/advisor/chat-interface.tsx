"use client"

import * as React from "react"
import { Send, Loader2, Eraser } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MessageBubble } from "@/components/advisor/message-bubble"
import { SuggestedPrompts } from "@/components/advisor/suggested-prompts"
import { cn } from "@/lib/utils"
import { aiApi } from "@/lib/api/ai"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export function ChatInterface() {
  const [input, setInput] = React.useState("")
  const [messages, setMessages] = React.useState<Message[]>([])
  const [isLoading, setIsLoading] = React.useState(false)
  const messagesEndRef = React.useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  React.useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (text: string) => {
    if (!text.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // Call real AI API
      const response = await aiApi.ask(text, { history: messages })

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.response,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiResponse])
    } catch (error) {
      // Fallback error message
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm sorry, I encountered an error processing your request. Please try again or check if the backend server is running.",
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorResponse])
      console.error("AI API Error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    handleSend(input)
  }

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] max-h-[800px] border border-border rounded-xl bg-card overflow-hidden shadow-sm">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-muted/50 scrollbar-track-transparent">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center space-y-8">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-display font-bold text-foreground">
                How can I help you today?
              </h2>
              <p className="text-muted-foreground">
                Ask about stocks, mutual funds, or get personalized investment advice.
              </p>
            </div>
            <SuggestedPrompts onSelect={handleSend} />
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && (
              <div className="flex justify-start gap-4">
                 <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 border border-primary/20">
                  <Loader2 className="h-4 w-4 text-primary animate-spin" />
                </div>
                <div className="bg-secondary/50 border border-border rounded-2xl rounded-tl-none px-5 py-3 flex items-center">
                  <span className="text-sm text-muted-foreground animate-pulse">Analyzing market data...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-card border-t border-border">
        <form onSubmit={handleSubmit} className="flex gap-4 max-w-4xl mx-auto relative">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask your AI financial advisor..."
            className="flex-1 bg-background border-border focus-visible:ring-primary/50 min-h-[50px] pr-12 text-base text-foreground placeholder:text-muted-foreground"
            disabled={isLoading}
          />
          <Button 
            type="submit" 
            size="icon"
            className="absolute right-2 top-1.5 h-9 w-9 bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm"
            disabled={!input.trim() || isLoading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <div className="text-center mt-2">
            <span className="text-[10px] text-muted-foreground">
                AI can make mistakes. Please check important information.
            </span>
        </div>
      </div>
    </div>
  )
}