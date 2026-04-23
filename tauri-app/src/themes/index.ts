import type { Theme } from "./types";
import { slate } from "./slate";
import { atomMaterial } from "./atom-material";

export const themes = {
  slate,
  "atom-material": atomMaterial,
} as const;

export type ThemeName = keyof typeof themes;

export const ACTIVE_THEME: ThemeName = "atom-material";

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  const map: Record<keyof Theme["tokens"], string> = {
    background: "--background",
    foreground: "--foreground",
    card: "--card",
    cardForeground: "--card-foreground",
    popover: "--popover",
    popoverForeground: "--popover-foreground",
    primary: "--primary",
    primaryForeground: "--primary-foreground",
    secondary: "--secondary",
    secondaryForeground: "--secondary-foreground",
    muted: "--muted",
    mutedForeground: "--muted-foreground",
    accent: "--accent",
    accentForeground: "--accent-foreground",
    destructive: "--destructive",
    destructiveForeground: "--destructive-foreground",
    border: "--border",
    input: "--input",
    ring: "--ring",
    radius: "--radius",
  };
  for (const [key, value] of Object.entries(theme.tokens)) {
    root.style.setProperty(map[key as keyof Theme["tokens"]], value);
  }
}

export type { Theme };
