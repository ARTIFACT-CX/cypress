import logo from "./assets/logo.png";
import { ServerControl } from "./components/ServerControl";

function App() {
  return (
    <>
      <div data-tauri-drag-region className="titlebar-drag" />
      <main className="flex min-h-screen flex-col items-center justify-center gap-6">
        <img src={logo} alt="Cypress" className="h-32 w-32" />
        <h1 className="text-2xl font-medium tracking-tight text-foreground">
          Cypress
        </h1>
        <p className="text-sm text-muted-foreground">
          Local voice inference, on your machine.
        </p>
      </main>
      <ServerControl />
    </>
  );
}

export default App;
