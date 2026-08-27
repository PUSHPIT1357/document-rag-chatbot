import { useEffect, useState } from "react";
import { api, type HealthResponse, type StatsResponse } from "@/lib/api";

export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealth(h);
        setOnline(true);
      } catch {
        if (cancelled) return;
        setOnline(false);
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { health, online };
}

export function useStats(refreshKey: number) {
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await api.stats();
        if (!cancelled) setStats(s);
      } catch {
        if (!cancelled) setStats(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  return stats;
}
