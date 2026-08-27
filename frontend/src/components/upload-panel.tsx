import { useCallback, useRef, useState } from "react";
import { Upload, FileText, FileSpreadsheet, FileCode, File as FileIcon, Loader2, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api, ACCEPTED_EXTENSIONS, ACCEPT_ATTR, ApiError, NetworkError, type UploadResponse } from "@/lib/api";

function iconFor(name: string) {
  const ext = name.toLowerCase().split(".").pop();
  const cls = "h-8 w-8 text-primary";
  switch (ext) {
    case "pdf":
      return <FileText className={cls} />;
    case "docx":
      return <FileText className={cls} />;
    case "csv":
      return <FileSpreadsheet className={cls} />;
    case "md":
      return <FileCode className={cls} />;
    case "txt":
      return <FileIcon className={cls} />;
    default:
      return <FileIcon className={cls} />;
  }
}

type Props = {
  onUploaded: (info: UploadResponse) => void;
  currentDoc: UploadResponse | null;
};

export function UploadPanel({ onUploaded, currentDoc }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const doUpload = useCallback(
    async (file: File) => {
      const ext = "." + (file.name.toLowerCase().split(".").pop() ?? "");
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        toast.error(`Unsupported file type: ${ext}`, {
          description: `Allowed: ${ACCEPTED_EXTENSIONS.join(", ")}`,
        });
        return;
      }
      setUploading(true);
      try {
        const res = await api.upload(file);
        toast.success("Document processed", {
          description: `${res.filename} · ${res.chunks} chunks`,
        });
        onUploaded(res);
      } catch (e) {
        if (e instanceof NetworkError) {
          toast.error("Can't reach the server");
        } else if (e instanceof ApiError) {
          toast.error(e.message);
        } else {
          toast.error("Upload failed");
        }
      } finally {
        setUploading(false);
      }
    },
    [onUploaded],
  );

  return (
    <Card
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) void doUpload(file);
      }}
      className={`border-2 border-dashed p-8 transition-colors ${
        dragOver ? "border-primary bg-primary/5" : "border-border"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void doUpload(f);
          e.target.value = "";
        }}
      />
      <div className="flex flex-col items-center justify-center gap-3 text-center">
        {uploading ? (
          <>
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Processing document…</p>
          </>
        ) : currentDoc ? (
          <>
            <div className="flex items-center gap-3">
              {iconFor(currentDoc.filename)}
              <div className="text-left">
                <div className="flex items-center gap-2 font-medium">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  {currentDoc.filename}
                </div>
                <p className="text-xs text-muted-foreground">
                  {currentDoc.chunks} chunks indexed
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => inputRef.current?.click()}
            >
              Replace document
            </Button>
          </>
        ) : (
          <>
            <Upload className="h-10 w-10 text-muted-foreground" />
            <div>
              <p className="font-medium">Drop a document to get started</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {ACCEPTED_EXTENSIONS.join(" · ")} · up to a few MB
              </p>
            </div>
            <Button onClick={() => inputRef.current?.click()}>Choose file</Button>
          </>
        )}
      </div>
    </Card>
  );
}
