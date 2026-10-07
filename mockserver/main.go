// mockserver — имитация vLLM (OpenAI SSE + /tokenize + /metrics) для пробного прогона llm-loadtest.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

var (
	running, waiting          atomic.Int64
	promptTok, genTok, drafts atomic.Int64
	draftTok, accTok          atomic.Int64
	slots                     chan struct{}
)

func main() {
	addr := flag.String("addr", ":8000", "адрес")
	maxSeqs := flag.Int("max-seqs", 8, "одновременно обрабатываемых запросов (остальные ждут в очереди)")
	flag.Parse()
	slots = make(chan struct{}, *maxSeqs)

	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"id":"mock-model","max_model_len":16384}]}`)
	})
	http.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		var b struct{ Prompt string }
		json.NewDecoder(r.Body).Decode(&b)
		fmt.Fprintf(w, `{"count":%d}`, utf8.RuneCountInString(b.Prompt)*10/33)
	})
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "# HELP x\nvllm:num_requests_running{engine=\"0\"} %d\nvllm:num_requests_waiting{engine=\"0\"} %d\n", running.Load(), waiting.Load())
		fmt.Fprintf(w, "vllm:kv_cache_usage_perc{engine=\"0\"} %f\n", float64(running.Load())/float64(cap(slots)))
		fmt.Fprintf(w, "vllm:prompt_tokens_total{engine=\"0\"} %d\nvllm:generation_tokens_total{engine=\"0\"} %d\n", promptTok.Load(), genTok.Load())
		fmt.Fprintf(w, "vllm:spec_decode_num_drafts_total %d\nvllm:spec_decode_num_draft_tokens_total %d\nvllm:spec_decode_num_accepted_tokens_total %d\n", drafts.Load(), draftTok.Load(), accTok.Load())
		fmt.Fprint(w, "vllm:prefix_cache_queries_total 1000\nvllm:prefix_cache_hits_total 600\nvllm:num_preemptions_total 0\n")
	})
	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			http.Error(w, `{"error":"unauthorized"}`, 401)
			return
		}
		var req struct {
			Messages  []struct{ Content string } `json:"messages"`
			MaxTokens int                        `json:"max_tokens"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		chars := 0
		for _, m := range req.Messages {
			chars += utf8.RuneCountInString(m.Content)
		}
		pt := chars*10/33 + 6*len(req.Messages)
		waiting.Add(1)
		select {
		case slots <- struct{}{}:
		case <-r.Context().Done():
			waiting.Add(-1)
			return
		}
		waiting.Add(-1)
		running.Add(1)
		defer func() { running.Add(-1); <-slots }()

		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		time.Sleep(time.Duration(pt/20) * time.Millisecond) // prefill ~20k tok/s
		n := req.MaxTokens/2 + rand.Intn(req.MaxTokens/2+1)
		perTok := time.Duration(15+3*running.Load()) * time.Millisecond
		words := []string{"Это ", "очень ", "интересный ", "отрывок, ", "в котором ", "герой ", "размышляет ", "о жизни. "}
		for i := 0; i < n; {
			k := 1 + rand.Intn(3) // «спекулятивный» шаг: 1..3 токена за раз
			if i+k > n {
				k = n - i
			}
			drafts.Add(1)
			draftTok.Add(3)
			accTok.Add(int64(k - 1))
			time.Sleep(perTok)
			txt := ""
			for j := 0; j < k; j++ {
				txt += words[(i+j)%len(words)]
			}
			b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": txt}}}})
			if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
				return
			}
			fl.Flush()
			i += k
		}
		genTok.Add(int64(n))
		promptTok.Add(int64(pt))
		fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"length"}]}`+"\n\n")
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\n", pt, n)
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
	})
	log.Printf("mock vLLM на %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
