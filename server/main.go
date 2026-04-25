// AREA: server · ENTRY · COMPOSITION-ROOT
// Cypress orchestration server. Boots the HTTP listener, constructs the
// long-lived feature components, wires them together, and routes the
// shutdown signal back through them.
//
// This file is intentionally thin — every concrete handler, business-
// logic struct, and adapter lives inside its feature package. main.go
// is the *only* place where features are introduced to each other,
// which keeps the cross-feature wiring grep-able to one location.

package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/audio"
	"github.com/ARTIFACT-CX/cypress/server/inference"
)

// SETUP: default listen address. The UI connects here from localhost; we don't
// bind externally because all traffic is intra-machine.
const listenAddr = "127.0.0.1:7842"

func main() {
	// STEP 1: build the long-lived feature components. Each one is created
	// in "idle" state — starting the Python worker or opening an audio
	// pipeline happens later, in response to explicit UI commands.
	inferenceMgr := inference.NewManager()
	// REASON: inferenceMgr satisfies audio.InferenceClient (the interface
	// declared inside the audio feature). Passing it as an interface
	// keeps audio decoupled from inference's concrete type.
	audioPipeline := audio.NewPipeline(inferenceMgr)
	wsHandler := audio.NewWSHandler(audioPipeline)

	// STEP 2: wire HTTP routes. Each feature's inbound adapter registers
	// its own routes onto the shared mux; main.go just owns the mux.
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.Handle("/ws", wsHandler)
	inference.RegisterRoutes(mux, inferenceMgr)

	srv := &http.Server{
		Addr:              listenAddr,
		Handler:           withCORS(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	// STEP 3: bind the port first, then announce ready. Binding before
	// logging means a bind failure (port in use, permission) is visible at
	// the top of the log rather than buried after an optimistic "listening"
	// line. Accept() is deferred to the goroutine below.
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("bind %s failed: %v", listenAddr, err)
	}
	log.Printf("cypress-server listening on %s", listenAddr)

	go func() {
		if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server failed: %v", err)
		}
	}()

	// STEP 4: block until we get a shutdown signal.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	log.Println("shutting down...")

	// STEP 5: graceful shutdown with a hard deadline. Python worker goes first
	// because it holds the big GPU handle; HTTP server second.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	inferenceMgr.Shutdown(shutdownCtx)
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("http shutdown error: %v", err)
	}
	log.Println("bye")
}

// withCORS wraps the mux with permissive CORS headers. Safe because the
// server binds to 127.0.0.1 only — no remote origin can reach us anyway,
// but the browser still enforces same-origin on the Tauri UI's dev server
// (http://localhost:1420). Without these headers fetch() from the UI fails
// with an opaque "access control checks" error.
//
// Lives in main.go (the composition root) rather than a feature because
// CORS is a cross-cutting concern of the HTTP listener itself, not of any
// individual feature's adapter. Promote to shared/httpx if a second
// listener ever needs it.
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
