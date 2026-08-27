import { useEffect, useRef, useState } from "react";
import { Send, Loader2, ChevronDown, ChevronRight, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { api, ApiError, NetworkError, type Source } from "@/lib/api";
import logo from "./ledger-logo.png";

type Msg =
  | { role: "user"; content: string; id: string }
  | { role: "assistant"; content: string; sources?: Source[]; id: string }
  | { role: "system"; content: string; id: string };

export function ChatPanel({ enabled }: { enabled: boolean }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading || !enabled) return;
    const uid = crypto.randomUUID();
    setMessages((m) => [...m, { role: "user", content: q, id: uid }]);
    setInput("");
    setLoading(true);
    try {
      const res = await api.ask(q);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, sources: res.sources, id: crypto.randomUUID() },
      ]);
    } catch (e) {
      const msg =
        e instanceof NetworkError
          ? "Can't reach the server."
          : e instanceof ApiError
            ? e.message
            : "Something failed.";
      setMessages((m) => [...m, { role: "system", content: msg, id: crypto.randomUUID() }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="flex h-[70vh] min-h-[500px] flex-col overflow-hidden">
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4 sm:p-6">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center text-muted-foreground">
            <img src={logo} alt="Ledger" width={64} height={64} className="mb-3 opacity-80" />
            <p className="font-medium text-foreground">
              {enabled ? "Ask Ledger anything about your document" : "Upload a document to start chatting"}
            </p>
            <p className="mt-1 text-sm">Answers are grounded in the document you upload.</p>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} msg={m} />
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Thinking…
          </div>
        )}
      </div>
      <div className="border-t bg-muted/30 p-3">
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={enabled ? "Ask a question…" : "Upload a document first"}
            disabled={!enabled || loading}
          />
          <Button type="submit" disabled={!enabled || loading || !input.trim()}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </form>
      </div>
    </Card>
  );
}

function MessageBubble({ msg }: { msg: Msg }) {
  if (msg.role === "system") {
    return (
      <div className="mx-auto max-w-md rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-center text-xs text-destructive">
        {msg.content}
      </div>
    );
  }
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <img src={logo} alt="" width={20} height={20} />}
      </div>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-2`}>
        <div
          className={`whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-foreground"
          }`}
        >
          {msg.content}
        </div>
        {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
          <SourcesBlock sources={msg.sources} />
        )}
      </div>
    </div>
  );
}

function SourcesBlock({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Sources ({sources.length})
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <SourceItem key={i} source={s} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceItem({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const meta = source.metadata || {};
  const page = (meta as Record<string, unknown>).page;
  const row = (meta as Record<string, unknown>).row;
  const truncated = source.content.length > 240;
  const shown = expanded ? source.content : source.content.slice(0, 240);
  return (
    <div className="rounded-md border bg-background p-3 text-xs">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <Badge variant="secondary">#{index + 1}</Badge>
        <Badge variant="outline">{Math.round(source.similarity * 100)}% match</Badge>
        {page !== undefined && <Badge variant="outline">Page {String(page)}</Badge>}
        {row !== undefined && <Badge variant="outline">Row {String(row)}</Badge>}
      </div>
      <p className="whitespace-pre-wrap text-muted-foreground">
        {shown}
        {truncated && !expanded && "…"}
      </p>
      {truncated && (
        <button
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 text-primary hover:underline"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
