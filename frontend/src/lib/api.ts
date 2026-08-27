export const API_URL =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ||
  "http://localhost:8000";

export type UploadResponse = {
  message: string;
  filename: string;
  chunks: number;
  status: string;
  free: boolean;
  llm_active: boolean;
};

export type Source = {
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
};

export type AskResponse = {
  answer: string;
  sources: Source[];
  query: string;
  chunks_found: number;
};

export type HealthResponse = {
  status: string;
  bot_initialized: boolean;
  bot_ready: boolean;
  chunks: number;
};

export type StatsResponse = {
  pdf_loaded: boolean;
  pdf_name: string | null;
  chunks_count: number;
  database: string;
  embedding_model: string;
  llm_status: string;
  llm_active: boolean;
  free: boolean;
  api_keys_needed: boolean;
};

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
export class NetworkError extends Error {}

async function handle<T>(res: Response): Promise<T> {
  let data: any = null;
  try {
    data = await res.json();
  } catch {
    /* ignore */
  }
  if (!res.ok) {
    const msg = data?.error || data?.detail || `Request failed (${res.status})`;
    throw new ApiError(msg, res.status);
  }
  return data as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_URL}${path}`, init);
    return await handle<T>(res);
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new NetworkError("Can't reach the server");
  }
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  stats: () => request<StatsResponse>("/stats"),
  reset: () =>
    request<{ message: string; status: string }>("/reset", { method: "DELETE" }),
  ask: (question: string) =>
    request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<UploadResponse>("/upload", { method: "POST", body: fd });
  },
};

export const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".md"];
export const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(",");
