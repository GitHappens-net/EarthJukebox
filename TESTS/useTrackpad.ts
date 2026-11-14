import type { PointerEvent } from "react";
import { useEffect, useState } from "react";

export type PadState = {
  x: number;
  y: number;
  touching: boolean;
  fingers: number;
};

type PadUpdateEvent = CustomEvent<PadState>;

const initialPad: PadState = {
  x: 0.5,
  y: 0.5,
  touching: false,
  fingers: 0,
};

export function useTrackpad() {
  const [isDesktopApp, setIsDesktopApp] = useState(false);
  const [pad, setPad] = useState<PadState>(initialPad);

  useEffect(() => {
    const onPadUpdate = (event: Event) => {
      setIsDesktopApp(true);
      setPad((event as PadUpdateEvent).detail);
    };

    window.addEventListener("pad-update", onPadUpdate);
    return () => window.removeEventListener("pad-update", onPadUpdate);
  }, []);

  const onFallbackPointer = (event: PointerEvent<HTMLElement>, touching: boolean) => {
    if (isDesktopApp) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setPad({
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
      touching,
      fingers: touching ? 1 : 0,
    });
  };

  return { pad, isDesktopApp, onFallbackPointer };
}
