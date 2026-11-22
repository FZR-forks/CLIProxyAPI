package cliproxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/MoeForge/moego/ojson"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
)

func TestBuildClaudeConfigModelsRetainsOutputLimit(t *testing.T) {
	for _, name := range []string{"claude-haiku-5-5", "claude-opus-5-5"} {
		t.Run(name, func(t *testing.T) {
			info := buildClaudeConfigModels(&config.ClaudeKey{Models: []config.ClaudeModel{{
				Name: name, Alias: "configured-" + name, MaxContextLength: 750000,
				Thinking: &registry.ThinkingSupport{Levels: []string{"xhigh"}},
			}}})[0]
			if info.MaxCompletionTokens != 128000 {
				t.Errorf("configured output limit = %d, want catalog limit 128000", info.MaxCompletionTokens)
			}
			if info.ID != "configured-"+name || info.MetadataModelID != name || info.ContextLength != 750000 || info.MaxContextLength != 750000 {
				t.Error("output limit inheritance changed configured identity or context override")
			}
			if !info.ExplicitThinking || len(info.Thinking.Levels) != 1 || info.Thinking.Levels[0] != "xhigh" {
				t.Error("output limit inheritance changed explicit thinking levels")
			}
		})
	}
	unknown := buildClaudeConfigModels(&config.ClaudeKey{Models: []config.ClaudeModel{{Name: "private-unknown-claude"}}})[0]
	if unknown.MaxCompletionTokens != 0 {
		t.Fatal("unknown model inherited an unrelated output limit")
	}
}

func TestConfiguredClaudeResponsesUsesModelOutputLimit(t *testing.T) {
	var seen []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		response := ojson.NewDoc([]byte(`{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`))
		_ = response.Set("message.model", ojson.GetBytes(seen, "model").String())
		_, _ = io.WriteString(w, "event: message_start\ndata: "+response.String()+"\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()
	for _, model := range []string{"claude-haiku-5-5", "claude-opus-5-5"} {
		t.Run(model, func(t *testing.T) {
			clientID := "configured-output-limit-" + model
			models := buildClaudeConfigModels(&config.ClaudeKey{Models: []config.ClaudeModel{{
				Name: model, Alias: model, MaxContextLength: 1000000,
				Thinking: &registry.ThinkingSupport{Levels: []string{"low", "medium", "high", "xhigh", "max"}},
			}}})
			reg := registry.GetGlobalRegistry()
			reg.RegisterClient(clientID, "claude", models)
			t.Cleanup(func() { reg.UnregisterClient(clientID) })
			for _, tc := range []struct {
				name   string
				limit  string
				want   int64
				config *config.Config
			}{
				{name: "omitted", want: 128000},
				{name: "null", limit: `,"max_output_tokens":null`, want: 128000},
				{name: "explicit", limit: `,"max_output_tokens":4096`, want: 4096},
				{name: "above-model-limit", limit: `,"max_output_tokens":256000`, want: 128000},
				{name: "payload-default", want: 16000, config: &config.Config{Payload: config.PayloadConfig{
					Default: []config.PayloadRule{{Models: []config.PayloadModelRule{{Name: model, Protocol: "claude"}}, Params: map[string]any{"max_tokens": 16000}}},
				}}},
				{name: "payload-default-preserves-explicit", limit: `,"max_output_tokens":4096`, want: 4096, config: &config.Config{Payload: config.PayloadConfig{
					Default: []config.PayloadRule{{Models: []config.PayloadModelRule{{Name: model, Protocol: "claude"}}, Params: map[string]any{"max_tokens": 16000}}},
				}}},
				{name: "payload-override", want: 8000, config: &config.Config{Payload: config.PayloadConfig{
					Override: []config.PayloadRule{{Models: []config.PayloadModelRule{{Name: model, Protocol: "claude"}}, Params: map[string]any{"max_tokens": 8000}}},
				}}},
			} {
				for _, stream := range []bool{false, true} {
					t.Run(tc.name+map[bool]string{false: "/execute", true: "/stream"}[stream], func(t *testing.T) {
						payload := []byte(`{"model":"` + model + `","input":"offline test","reasoning":{"effort":"xhigh"}` + tc.limit + `}`)
						req := cliproxyexecutor.Request{Model: model, Payload: payload}
						opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse, ResponseFormat: sdktranslator.FormatOpenAIResponse, OriginalRequest: payload, Stream: stream}
						auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "offline-test-only", "base_url": server.URL}}
						cfg := tc.config
						if cfg == nil {
							cfg = &config.Config{}
						}
						e := executor.NewClaudeExecutor(cfg)
						if stream {
							result, err := e.ExecuteStream(context.Background(), auth, req, opts)
							if err != nil {
								t.Fatal(err)
							}
							for chunk := range result.Chunks {
								if chunk.Err != nil {
									t.Fatal(chunk.Err)
								}
							}
						} else if _, err := e.Execute(context.Background(), auth, req, opts); err != nil {
							t.Fatal(err)
						}
						if got := ojson.GetBytes(seen, "max_tokens").Int(); got != tc.want {
							t.Errorf("upstream max_tokens = %d, want %d", got, tc.want)
						}
						if got := ojson.GetBytes(seen, "output_config.effort").String(); got != "xhigh" {
							t.Errorf("upstream effort = %q, want unchanged xhigh", got)
						}
					})
				}
			}
		})
	}
}
