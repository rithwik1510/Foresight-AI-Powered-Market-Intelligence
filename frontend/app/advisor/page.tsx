import { ChatInterface } from "@/components/advisor/chat-interface"
import { Bot } from "lucide-react"

export default function AdvisorPage() {
  return (
    <div className="max-w-5xl mx-auto space-y-6 pb-8 h-full flex flex-col">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center border border-primary/20">
            <Bot className="h-6 w-6 text-primary" />
        </div>
        <div>
            <h1 className="text-2xl font-display font-bold text-foreground tracking-tight">AI Advisor</h1>
            <p className="text-sm text-muted-foreground">Powered by Gemini Pro & Custom Market Models</p>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <ChatInterface />
      </div>
    </div>
  )
}