// AREA: inference · WORKER · WIRE
// Conversion between the Manager's plain map[string]any contract and
// the typed proto messages on the wire. Living in one place keeps the
// (cmd, extra) ↔ ClientMsg / ServerMsg switches grep-able and out of
// the Manager + grpcWorker plumbing.
//
// REASON: the workerHandle interface still hands callers map[string]any
// so we don't have to refactor Manager + Stream + every test in this
// pass. Future cleanup can pull typed replies up through the interface;
// the gRPC schema is already there to support it.

package inference

import (
	"fmt"

	pb "github.com/ARTIFACT-CX/cypress/proto/dist/go/workerpb"
)

// buildClientMsg packs (cmd, extra) into a ClientMsg with the right
// oneof variant. Unknown commands are an internal-bug error — the
// Manager should only emit names from this whitelist.
func buildClientMsg(id uint64, cmd string, extra map[string]any) (*pb.ClientMsg, error) {
	out := &pb.ClientMsg{Id: id}
	switch cmd {
	case "status":
		out.Payload = &pb.ClientMsg_Status{Status: &pb.StatusReq{}}
	case "load_model":
		out.Payload = &pb.ClientMsg_LoadModel{LoadModel: &pb.LoadModelReq{
			Name: stringField(extra, "name"),
		}}
	case "unload":
		out.Payload = &pb.ClientMsg_Unload{Unload: &pb.UnloadReq{}}
	case "shutdown":
		out.Payload = &pb.ClientMsg_Shutdown{Shutdown: &pb.ShutdownReq{}}
	case "run_wav":
		out.Payload = &pb.ClientMsg_RunWav{RunWav: &pb.RunWavReq{
			Input:  stringField(extra, "input"),
			Output: stringField(extra, "output"),
		}}
	case "start_stream":
		out.Payload = &pb.ClientMsg_StartStream{StartStream: &pb.StartStreamReq{}}
	case "audio_in":
		out.Payload = &pb.ClientMsg_AudioIn{AudioIn: &pb.AudioInReq{
			Pcm: bytesField(extra, "pcm"),
		}}
	case "stop_stream":
		out.Payload = &pb.ClientMsg_StopStream{StopStream: &pb.StopStreamReq{}}
	case "download_model":
		req := &pb.DownloadModelReq{
			Name:  stringField(extra, "name"),
			Repo:  stringField(extra, "repo"),
			Files: stringSliceField(extra, "files"),
		}
		if rev, ok := extra["revision"].(string); ok && rev != "" {
			req.Revision = &rev
		}
		out.Payload = &pb.ClientMsg_DownloadModel{DownloadModel: req}
	case "cancel_download":
		out.Payload = &pb.ClientMsg_CancelDownload{CancelDownload: &pb.CancelDownloadReq{}}
	default:
		return nil, fmt.Errorf("buildClientMsg: unknown command %q", cmd)
	}
	return out, nil
}

// replyToMap converts a typed Reply back into the legacy dict shape.
// Empty/zero fields are deliberately included for the keys callers
// expect (e.g. "ok": true) so the Manager's map lookups don't have
// to change.
func replyToMap(r *pb.Reply) map[string]any {
	out := map[string]any{"ok": true}
	switch v := r.GetResult().(type) {
	case *pb.Reply_Status:
		if v.Status.GetModel() != "" {
			out["model"] = v.Status.GetModel()
		} else {
			out["model"] = nil
		}
		if v.Status.GetDevice() != "" {
			out["device"] = v.Status.GetDevice()
		} else {
			out["device"] = nil
		}
	case *pb.Reply_LoadModel:
		out["model"] = v.LoadModel.GetModel()
		if d := v.LoadModel.GetDevice(); d != "" {
			out["device"] = d
		}
	case *pb.Reply_StartStream:
		// Manager.StartStream reads sample_rate as float64 (legacy
		// JSON shape); preserve that so its type assertion still works.
		out["sample_rate"] = float64(v.StartStream.GetSampleRate())
	case *pb.Reply_DownloadStarted:
		out["started"] = v.DownloadStarted.GetStarted()
	case *pb.Reply_CancelDownload:
		out["active"] = v.CancelDownload.GetActive()
	case *pb.Reply_Ok:
		// nothing to add beyond "ok": true
	}
	return out
}

// eventToMap converts a typed Event into the legacy dict shape with the
// "event" string field that handleEvent + dispatchStreamEvent +
// handleDownloadEvent all key off. Unknown variants return nil so the
// recv loop drops them silently rather than calling onEvent with
// something it can't dispatch.
func eventToMap(e *pb.Event) map[string]any {
	switch v := e.GetPayload().(type) {
	case *pb.Event_ModelPhase:
		m := map[string]any{
			"event": "model_phase",
			"phase": v.ModelPhase.GetPhase(),
		}
		if d := v.ModelPhase.GetDevice(); d != "" {
			m["device"] = d
		}
		return m
	case *pb.Event_AudioOut:
		return map[string]any{
			"event": "audio_out",
			"pcm":   v.AudioOut.GetPcm(),
			"text":  v.AudioOut.GetText(),
		}
	case *pb.Event_StreamError:
		return map[string]any{
			"event": "stream_error",
			"error": v.StreamError.GetError(),
		}
	case *pb.Event_DownloadProgress:
		p := v.DownloadProgress
		// Numeric fields cast to float64 to match the JSON-era shape
		// (encoding/json decoded all numbers as float64). The download
		// event handler in downloads.go already does float64 type
		// assertions; keeping the shape stable means no churn there.
		return map[string]any{
			"event":      "download_progress",
			"name":       p.GetName(),
			"phase":      p.GetPhase(),
			"downloaded": float64(p.GetDownloaded()),
			"total":      float64(p.GetTotal()),
			"file":       p.GetFile(),
			"fileIndex":  float64(p.GetFileIndex()),
			"fileCount":  float64(p.GetFileCount()),
		}
	case *pb.Event_DownloadDone:
		d := v.DownloadDone
		out := map[string]any{
			"event":      "download_done",
			"name":       d.GetName(),
			"repo":       d.GetRepo(),
			"files":      stringSliceToAny(d.GetFiles()),
			"totalBytes": float64(d.GetTotalBytes()),
		}
		if d.Revision != nil {
			out["revision"] = d.GetRevision()
		}
		return out
	case *pb.Event_DownloadError:
		return map[string]any{
			"event": "download_error",
			"name":  v.DownloadError.GetName(),
			"error": v.DownloadError.GetError(),
		}
	}
	return nil
}

// --- small typed helpers (keep buildClientMsg readable) -------------
// stringField / stringSliceField live in downloads.go (shared).

func bytesField(m map[string]any, k string) []byte {
	if v, ok := m[k].([]byte); ok {
		return v
	}
	return nil
}

func stringSliceToAny(in []string) []any {
	out := make([]any, len(in))
	for i, s := range in {
		out[i] = s
	}
	return out
}
