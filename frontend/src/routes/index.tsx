import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Menu } from "lucide-react";
import logo from "@/components/ledger-logo.png";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { UploadPanel } from "@/components/upload-panel";
import { ChatPanel } from "@/components/chat-panel";
import { StatsSidebar } from "@/components/stats-sidebar";
import { useHealth, useStats } from "@/lib/hooks";
import type { UploadResponse } from "@/lib/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Ledger — Document Q&A, powered by AI" },
      { name: "description", content: "Upload PDFs, DOCX, TXT, CSV, or Markdown and ask questions grounded in your document." },
      { property: "og:title", content: "Ledger — Document Q&A, powered by AI" },
      { property: "og:description", content: "Ask questions grounded in your documents." },
    ],
  }),
  component: LedgerApp,
});

function LedgerApp() {
  const [doc, setDoc] = useState<UploadResponse | null>(null);
  const [statsKey, setStatsKey] = useState(0);
  const { online } = useHealth();
  const stats = useStats(statsKey + (doc ? 1 : 0));

  const refreshStats = () => setStatsKey((k) => k + 1);

  const handleReset = () => {
    setDoc(null);
    refreshStats();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50/40">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <img src={logo} alt="Ledger logo" width={36} height={36} className="rounded" />
            <div>
              <h1 className="text-lg font-semibold leading-tight">Ledger</h1>
              <p className="text-xs text-muted-foreground">Document Q&A, powered by AI</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={
                online === null
                  ? ""
                  : online
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                    : "border-destructive/30 bg-destructive/10 text-destructive"
              }
            >
              <span
                className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full ${
                  online === null ? "bg-muted-foreground" : online ? "bg-emerald-500" : "bg-destructive"
                }`}
              />
              {online === null ? "Checking…" : online ? "Online" : "Offline"}
            </Badge>
            {stats && (
              <Badge variant={stats.llm_active ? "default" : "secondary"} className="hidden sm:inline-flex">
                {stats.llm_active ? "AI: Active" : "AI: Fallback Mode"}
              </Badge>
            )}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" className="lg:hidden">
                  <Menu className="h-4 w-4" />
                </Button>
              </SheetTrigger>
              <SheetContent>
                <SheetHeader>
                  <SheetTitle>Session</SheetTitle>
                </SheetHeader>
                <div className="mt-6">
                  <StatsSidebar stats={stats} onReset={handleReset} />
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          <UploadPanel
            currentDoc={doc}
            onUploaded={(u) => {
              setDoc(u);
              refreshStats();
            }}
          />
          <ChatPanel enabled={!!doc} />
        </div>
        <aside className="hidden lg:block">
          <div className="sticky top-24 rounded-lg border bg-card p-5 shadow-sm">
            <StatsSidebar stats={stats} onReset={handleReset} />
          </div>
        </aside>
      </main>
    </div>
  );
}
