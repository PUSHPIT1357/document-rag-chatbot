import { useState } from "react";
import { RotateCcw, Loader2, Database, Cpu, FileText, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { toast } from "sonner";
import { api, ApiError, NetworkError, type StatsResponse } from "@/lib/api";

type Props = {
  stats: StatsResponse | null;
  onReset: () => void;
};

export function StatsSidebar({ stats, onReset }: Props) {
  const [resetting, setResetting] = useState(false);

  const doReset = async () => {
    setResetting(true);
    try {
      await api.reset();
      toast.success("Session reset");
      onReset();
    } catch (e) {
      const msg =
        e instanceof NetworkError
          ? "Can't reach the server"
          : e instanceof ApiError
            ? e.message
            : "Reset failed";
      toast.error(msg);
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold">Session</h3>
        <div className="mt-3 space-y-3 text-sm">
          <StatRow
            icon={<FileText className="h-4 w-4" />}
            label="Document loaded"
            value={stats?.pdf_loaded ? "Yes" : "No"}
          />
          <StatRow
            icon={<FileText className="h-4 w-4" />}
            label="Document name"
            value={stats?.pdf_name ?? "—"}
          />
          <StatRow
            icon={<Database className="h-4 w-4" />}
            label="Chunks"
            value={stats ? String(stats.chunks_count) : "—"}
          />
        </div>
      </div>

      <Separator />

      <div>
        <h3 className="text-sm font-semibold">Engine</h3>
        <div className="mt-3 space-y-3 text-sm">
          <StatRow
            icon={<Cpu className="h-4 w-4" />}
            label="Embeddings"
            value={stats?.embedding_model ?? "—"}
          />
          <StatRow
            icon={<Database className="h-4 w-4" />}
            label="Vector DB"
            value={stats?.database ?? "—"}
          />
          <StatRow
            icon={<Sparkles className="h-4 w-4" />}
            label="LLM"
            value={
              stats ? (
                <Badge variant={stats.llm_active ? "default" : "secondary"}>
                  {stats.llm_status}
                </Badge>
              ) : (
                "—"
              )
            }
          />
        </div>
      </div>

      <Separator />

      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button variant="outline" className="w-full" disabled={resetting}>
            {resetting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RotateCcw className="mr-2 h-4 w-4" />
            )}
            Reset session
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Reset session?</AlertDialogTitle>
            <AlertDialogDescription>
              This clears the loaded document and chat history on the server.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={doReset}>Reset</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function StatRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span>{label}</span>
      </div>
      <div className="max-w-[60%] truncate text-right font-medium text-foreground">
        {value}
      </div>
    </div>
  );
}
