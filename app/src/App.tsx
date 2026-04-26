import logo from "./assets/logo.png";
import { ModelPicker } from "./components/ModelPicker";
import { ServerControl } from "./components/ServerControl";
import { VoiceButton } from "./components/VoiceButton";
import { useChromaticAberration } from "./hooks/useChromaticAberration";
import { useVoiceStore } from "./store/voiceStore";

function App() {
  const [logoRef, setLogoAudio] = useChromaticAberration<HTMLImageElement>();
  // Read voice state directly from the store — no prop drilling.
  // Each select is single-field so unrelated changes don't re-render
  // the page chrome.
  const sessionState = useVoiceStore((s) => s.state);
  const micLevel = useVoiceStore((s) => s.micLevel);
  const playbackLevel = useVoiceStore((s) => s.playbackLevel);
  const live = sessionState === "live" || sessionState === "connecting";
  // Pump audio level + live flag into the logo effect. While live,
  // the hook ignores mouse and reacts only to audio (silent → no
  // effect). While not live, it falls back to the original
  // mouse-reactive mode.
  setLogoAudio(Math.max(micLevel, playbackLevel), live);
  return (
    <>
      <div data-tauri-drag-region className="titlebar-drag" />
      <main className="flex min-h-screen flex-col items-center justify-center gap-6">
        <img ref={logoRef} src={logo} alt="Cypress" className="h-32 w-32 chromatic-aberration" />
        <h1 className="text-2xl font-medium tracking-tight text-foreground">
          Cypress
        </h1>
        {/* Tagline gives way to the conversation UX once the user
            starts a session — the transcript becomes the focal copy. */}
        {!live && (
          <p className="text-sm text-muted-foreground">
            Local voice inference, on your machine.
          </p>
        )}
      </main>
      <ModelPicker />
      <ServerControl />
      <VoiceButton />
    </>
  );
}

export default App;
