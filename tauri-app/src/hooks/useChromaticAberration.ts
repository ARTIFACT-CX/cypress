/*
 * AREA: ui · HOOKS · CHROMATIC-ABERRATION
 *
 * Drives the `.chromatic-aberration` CSS filter from the mouse position.
 * The hook returns a ref you attach to the target element; it then listens
 * to window `mousemove` and writes three CSS custom properties on that
 * element:
 *
 *   --ca-dx   : horizontal channel offset   (unitless, used in calc())
 *   --ca-dy   : vertical channel offset     (unitless, used in calc())
 *   --ca-blur : drop-shadow blur radius     (unitless, used in calc())
 *
 * The CSS side converts them to px via calc(var(--ca-dx) * 1px) etc.
 *
 * Intensity falls off linearly with distance from the element center, so
 * the effect is strongest when the cursor is right on top of the logo and
 * fades to zero past `falloff` pixels away.
 */
import { useEffect, useRef } from "react";

type Options = {
  // SWAP: tune these to taste. Defaults aim for "noticeable but not tacky".
  maxOffset?: number; // peak horizontal channel separation, px
  maxBlur?: number; // peak drop-shadow blur, px
  falloff?: number; // distance from center at which intensity hits 0, px
};

export function useChromaticAberration<T extends HTMLElement>(
  opts: Options = {},
) {
  const { maxOffset = 6, maxBlur = 3, falloff = 400 } = opts;
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const onMove = (e: MouseEvent) => {
      // STEP 1: measure element center in viewport coords. Recomputed every
      // move so window resizes / layout shifts don't desync the effect.
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;

      // STEP 2: vector from center to cursor. dx/dy sign drives which side
      // each color channel shifts toward.
      const dx = e.clientX - cx;
      const dy = e.clientY - cy;
      const dist = Math.hypot(dx, dy);

      // STEP 3: intensity ramps from 1 at center to 0 at `falloff`.
      const intensity = Math.max(0, 1 - dist / falloff);

      // STEP 4: normalize direction so offset magnitude is controlled by
      // intensity alone, not by raw distance. Guard div-by-zero when the
      // cursor is exactly on the center.
      const nx = dist === 0 ? 0 : dx / dist;
      const ny = dist === 0 ? 0 : dy / dist;

      const ox = nx * intensity * maxOffset;
      const oy = ny * intensity * maxOffset;
      const blur = intensity * maxBlur;

      el.style.setProperty("--ca-dx", ox.toFixed(2));
      el.style.setProperty("--ca-dy", oy.toFixed(2));
      el.style.setProperty("--ca-blur", blur.toFixed(2));
    };

    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [maxOffset, maxBlur, falloff]);

  return ref;
}
