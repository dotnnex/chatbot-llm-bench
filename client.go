package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Result — измерения одного запроса.
type Result struct {
	Stage            int       `json:"stage"`
	Concurrency      int       `json:"concurrency"`
	DialogID         int64     `json:"dialog_id"`
	Turn             int       `json:"turn"`
	StartTime        time.Time `json:"start_time"`
	OffsetSec        float64   `json:"offset_sec"`
	OK               bool      `json:"ok"`
	ErrKind          string    `json:"error_kind,omitempty"`
	ErrMsg           string    `json:"error,omitempty"`
	HTTPStatus       int       `json:"http_status,omitempty"`
	TTFTms           float64   `json:"ttft_ms"`
	E2Ems            float64   `json:"e2e_ms"`
	TPOTms           float64   `json:"tpot_ms"`
	ITLMeanMs        float64   `json:"itl_mean_ms"`
	ITLMaxMs         float64   `json:"itl_max_ms"`
	Chunks           int       `json:"chunks"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	CachedTokens     int       `json:"cached_tokens,omitempty"`
	UsageEstimated   bool      `json:"usage_estimated,omitempty"`
	PromptChars      int       `json:"prompt_chars"`
	FinishReason     string    `json:"finish_reason,omitempty"`

	TTFT time.Duration `json:"-"`
	E2E  time.Duration `json:"-"`
	ITLs []float32     `json:"-"` // мс между соседними чанками
}

func (r *Result) fail(kind, msg string) {
	r.OK = false
	r.ErrKind = kind
	r.ErrMsg = msg
}

func (r *Result) finalize() {
	r.TTFTms = ms(r.TTFT)
	r.E2Ems = ms(r.E2E)
	if r.OK && r.CompletionTokens > 1 {
		r.TPOTms = ms(r.E2E-r.TTFT) / float64(r.CompletionTokens-1)
	}
	if len(r.ITLs) > 0 {
		var sum, mx float64
		for _, v := range r.ITLs {
			f := float64(v)
			sum += f
			if f > mx {
				mx = f
			}
		}
		r.ITLMeanMs = sum / float64(len(r.ITLs))
		r.ITLMaxMs = mx
	}
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

type Client struct {
	chatURL     string
	modelsURL   string
	tokenizeURL string
	RootURL     string
	apiKey      string
	Model       string
	maxTokens   int
	temperature float64
	ignoreEOS   bool
	extra       map[string]any
	timeout     time.Duration
	http        *http.Client
	Inflight    atomic.Int64
}

func NewClient(cfg *Config) (*Client, error) {
	u := strings.TrimRight(cfg.URL, "/")
	var base string
	switch {
	case strings.HasSuffix(u, "/chat/completions"):
		base = strings.TrimSuffix(u, "/chat/completions")
	case strings.HasSuffix(u, "/v1"):
		base = u
	default:
		base = u + "/v1"
	}
	root := strings.TrimSuffix(base, "/v1")

	var extra map[string]any
	if cfg.ExtraBody != "" {
		if err := json.Unmarshal([]byte(cfg.ExtraBody), &extra); err != nil {
			return nil, fmt.Errorf("--extra-body: невалидный JSON: %w", err)
		}
	}
	tr := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		MaxIdleConnsPerHost: cfg.EndDialogs*2 + 16,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true, // gzip буферизует SSE и искажает TTFT
		ForceAttemptHTTP2:   true,
		TLSClientConfig:     &tls.Config{InsecureSkipVerify: cfg.Insecure},
	}
	return &Client{
		chatURL:     base + "/chat/completions",
		modelsURL:   base + "/models",
		tokenizeURL: root + "/tokenize",
		RootURL:     root,
		apiKey:      cfg.APIKey,
		Model:       cfg.Model,
		maxTokens:   cfg.MaxTokens,
		temperature: cfg.Temperature,
		ignoreEOS:   cfg.IgnoreEOS,
		extra:       extra,
		timeout:     cfg.RequestTimeout,
		http:        &http.Client{Transport: tr},
	}, nil
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

type ModelInfo struct {
	ID          string `json:"id"`
	MaxModelLen int    `json:"max_model_len"`
}

func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.modelsURL, nil)
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("GET %s: HTTP %d: %s", c.modelsURL, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var body struct {
		Data []ModelInfo `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, err
	}
	return body.Data, nil
}

// Tokenize — vLLM-эндпоинт /tokenize; нужен для точной калибровки размера сообщений.
func (c *Client) Tokenize(ctx context.Context, text string) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	payload, _ := json.Marshal(map[string]any{"model": c.Model, "prompt": text, "add_special_tokens": false})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.tokenizeURL, bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	c.setHeaders(req)
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var body struct {
		Count int `json:"count"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return 0, err
	}
	if body.Count <= 0 {
		return 0, errors.New("пустой ответ")
	}
	return body.Count, nil
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			Reasoning        string `json:"reasoning"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (c *Client) buildBody(msgs []Message) ([]byte, error) {
	body := map[string]any{
		"model":          c.Model,
		"messages":       msgs,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"max_tokens":     c.maxTokens,
		"temperature":    c.temperature,
	}
	if c.ignoreEOS {
		body["ignore_eos"] = true
	}
	for k, v := range c.extra {
		body[k] = v
	}
	return json.Marshal(body)
}

func classifyErr(ctx context.Context, err error) string {
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(ctx.Err(), context.Canceled) || errors.Is(err, context.Canceled):
		return "canceled"
	default:
		return "conn_error"
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// Chat выполняет потоковый запрос и возвращает измерения и текст ответа.
func (c *Client) Chat(ctx context.Context, msgs []Message) (*Result, string) {
	res := &Result{}
	for _, m := range msgs {
		res.PromptChars += len([]rune(m.Content))
	}
	defer res.finalize()

	body, err := c.buildBody(msgs)
	if err != nil {
		res.fail("client_error", err.Error())
		return res, ""
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatURL, bytes.NewReader(body))
	if err != nil {
		res.fail("client_error", err.Error())
		return res, ""
	}
	c.setHeaders(req)
	req.Header.Set("Accept", "text/event-stream")

	c.Inflight.Add(1)
	defer c.Inflight.Add(-1)

	t0 := time.Now()
	res.StartTime = t0
	resp, err := c.http.Do(req)
	if err != nil {
		res.E2E = time.Since(t0)
		res.fail(classifyErr(ctx, err), err.Error())
		return res, ""
	}
	defer resp.Body.Close()
	res.HTTPStatus = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		res.E2E = time.Since(t0)
		msg := strings.TrimSpace(string(b))
		kind := fmt.Sprintf("http_%d", resp.StatusCode)
		low := strings.ToLower(msg)
		if resp.StatusCode == http.StatusBadRequest &&
			(strings.Contains(low, "context length") || strings.Contains(low, "maximum context") ||
				strings.Contains(low, "max_model_len") || strings.Contains(low, "too long")) {
			kind = "context_overflow"
		}
		res.fail(kind, truncate(msg, 300))
		return res, ""
	}

	var answer strings.Builder
	rd := bufio.NewReaderSize(resp.Body, 64<<10)
	var last time.Time
	gotDone := false
	var streamErr error
	for {
		line, rerr := rd.ReadBytes('\n')
		if line = bytes.TrimSpace(line); bytes.HasPrefix(line, []byte("data:")) {
			payload := bytes.TrimSpace(line[5:])
			if bytes.Equal(payload, []byte("[DONE]")) {
				gotDone = true
				break
			}
			var ch streamChunk
			if json.Unmarshal(payload, &ch) == nil {
				if ch.Error != nil {
					streamErr = errors.New(ch.Error.Message)
					break
				}
				now := time.Now()
				for _, choice := range ch.Choices {
					piece := choice.Delta.Content + choice.Delta.ReasoningContent + choice.Delta.Reasoning
					if piece != "" {
						if res.Chunks == 0 {
							res.TTFT = now.Sub(t0)
						} else {
							res.ITLs = append(res.ITLs, float32(ms(now.Sub(last))))
						}
						last = now
						res.Chunks++
						answer.WriteString(choice.Delta.Content)
					}
					if choice.FinishReason != nil && *choice.FinishReason != "" {
						res.FinishReason = *choice.FinishReason
					}
				}
				if u := ch.Usage; u != nil {
					res.PromptTokens = u.PromptTokens
					res.CompletionTokens = u.CompletionTokens
					if u.PromptTokensDetails != nil {
						res.CachedTokens = u.PromptTokensDetails.CachedTokens
					}
				}
			}
		}
		if rerr != nil {
			if !errors.Is(rerr, io.EOF) {
				streamErr = rerr
			}
			break
		}
	}
	res.E2E = time.Since(t0)

	switch {
	case streamErr != nil:
		kind := classifyErr(ctx, streamErr)
		if kind == "conn_error" {
			kind = "stream_broken"
		}
		res.fail(kind, truncate(streamErr.Error(), 300))
	case res.Chunks == 0:
		res.fail("empty_response", "сервер не вернул ни одного токена")
	case !gotDone && res.FinishReason == "":
		res.fail("stream_broken", "поток оборвался без [DONE]")
	default:
		res.OK = true
		if res.CompletionTokens == 0 {
			res.CompletionTokens = res.Chunks
			res.UsageEstimated = true
		}
	}
	return res, answer.String()
}
