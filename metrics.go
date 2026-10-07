package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Scraper периодически читает Prometheus-метрики vLLM (/metrics):
// KV-cache, очередь, preemption, prefix cache и speculative decoding acceptance.
type Scraper struct {
	url     string
	apiKey  string
	http    *http.Client
	period  time.Duration
	mu      sync.Mutex
	samples []MetricSample
	cancel  context.CancelFunc
	done    chan struct{}
}

type MetricSample struct {
	T   time.Time
	Sum map[string]float64 // сумма по всем label-наборам (engine, model…)
	Max map[string]float64 // максимум по label-наборам
}

func NewScraper(url, apiKey string, insecure bool, period time.Duration) *Scraper {
	return &Scraper{
		url:    url,
		apiKey: apiKey,
		period: period,
		http: &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: insecure},
		}},
	}
}

func (s *Scraper) scrape(ctx context.Context) (MetricSample, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		return MetricSample{}, err
	}
	if s.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.apiKey)
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return MetricSample{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return MetricSample{}, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	sum, max := parseProm(resp.Body)
	if len(sum) == 0 {
		return MetricSample{}, fmt.Errorf("в ответе нет метрик vllm:*")
	}
	return MetricSample{T: time.Now(), Sum: sum, Max: max}, nil
}

func parseProm(r io.Reader) (map[string]float64, map[string]float64) {
	sum, max := map[string]float64{}, map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "vllm:") {
			continue
		}
		var name, rest string
		if i := strings.IndexByte(line, '{'); i >= 0 {
			j := strings.LastIndexByte(line, '}')
			if j < i {
				continue
			}
			name, rest = line[:i], line[j+1:]
		} else if i := strings.IndexByte(line, ' '); i >= 0 {
			name, rest = line[:i], line[i:]
		} else {
			continue
		}
		if strings.HasSuffix(name, "_bucket") || strings.HasSuffix(name, "_created") || strings.Contains(name, "per_pos") {
			continue
		}
		f := strings.Fields(rest)
		if len(f) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(f[0], 64)
		if err != nil {
			continue
		}
		sum[name] += v
		if cur, ok := max[name]; !ok || v > cur {
			max[name] = v
		}
	}
	return sum, max
}

// Probe проверяет доступность метрик.
func (s *Scraper) Probe(ctx context.Context) error {
	smp, err := s.scrape(ctx)
	if err != nil {
		return err
	}
	s.add(smp)
	return nil
}

func (s *Scraper) add(smp MetricSample) {
	s.mu.Lock()
	s.samples = append(s.samples, smp)
	s.mu.Unlock()
}

// ScrapeNow — внеочередной снимок (на границах ступеней).
func (s *Scraper) ScrapeNow(ctx context.Context) {
	if smp, err := s.scrape(ctx); err == nil {
		s.add(smp)
	}
}

func (s *Scraper) Start(ctx context.Context) {
	ctx, s.cancel = context.WithCancel(ctx)
	s.done = make(chan struct{})
	go func() {
		defer close(s.done)
		t := time.NewTicker(s.period)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				s.ScrapeNow(ctx)
			}
		}
	}()
}

func (s *Scraper) Stop() {
	if s.cancel != nil {
		s.cancel()
		<-s.done
	}
}

func (s *Scraper) Samples() []MetricSample {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]MetricSample(nil), s.samples...)
}

type ServerStats struct {
	RunningMax       *float64 `json:"running_max,omitempty"`
	WaitingMax       *float64 `json:"waiting_max,omitempty"`
	WaitingMean      *float64 `json:"waiting_mean,omitempty"`
	KVUsageMaxPct    *float64 `json:"kv_usage_max_pct,omitempty"`
	KVUsageMeanPct   *float64 `json:"kv_usage_mean_pct,omitempty"`
	Preemptions      *float64 `json:"preemptions,omitempty"`
	PrefixHitRatePct *float64 `json:"prefix_cache_hit_rate_pct,omitempty"`
	SpecAcceptPct    *float64 `json:"spec_decode_acceptance_pct,omitempty"`
	SpecMeanAccLen   *float64 `json:"spec_decode_mean_accept_len,omitempty"`
	GenTokPerSec     *float64 `json:"server_gen_tok_per_sec,omitempty"`
	PromptTokPerSec  *float64 `json:"server_prompt_tok_per_sec,omitempty"`
}

func fp(v float64) *float64 { return &v }

func lookup(m map[string]float64, names ...string) (float64, bool) {
	for _, n := range names {
		if v, ok := m[n]; ok {
			return v, true
		}
	}
	return 0, false
}

var (
	mRunning   = []string{"vllm:num_requests_running"}
	mWaiting   = []string{"vllm:num_requests_waiting"}
	mKV        = []string{"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}
	mPreempt   = []string{"vllm:num_preemptions_total", "vllm:num_preemptions"}
	mPrefixQ   = []string{"vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries"}
	mPrefixH   = []string{"vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits"}
	mDrafts    = []string{"vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"}
	mDraftTok  = []string{"vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"}
	mAccTok    = []string{"vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"}
	mGenTok    = []string{"vllm:generation_tokens_total"}
	mPromptTok = []string{"vllm:prompt_tokens_total"}
)

// Window считает серверную статистику за интервал [a, b].
func (s *Scraper) Window(a, b time.Time) *ServerStats {
	all := s.Samples()
	if len(all) < 2 {
		return nil
	}
	i0, i1 := 0, len(all)-1
	for i, smp := range all {
		if !smp.T.After(a) {
			i0 = i
		}
	}
	for i := len(all) - 1; i >= 0; i-- {
		if !all[i].T.Before(b) {
			i1 = i
		}
	}
	if i1 <= i0 {
		return nil
	}
	s0, s1 := all[i0], all[i1]
	dt := s1.T.Sub(s0.T).Seconds()
	delta := func(names []string) (float64, bool) {
		v0, ok0 := lookup(s0.Sum, names...)
		v1, ok1 := lookup(s1.Sum, names...)
		if !ok0 || !ok1 || v1 < v0 {
			return 0, false
		}
		return v1 - v0, true
	}

	st := &ServerStats{}
	// gauges — по снимкам внутри окна
	var nIn int
	var runMax, waitMax, waitSum, kvMax, kvSum float64
	var haveRun, haveWait, haveKV bool
	for _, smp := range all[i0 : i1+1] {
		if smp.T.Before(a) || smp.T.After(b) {
			continue
		}
		nIn++
		if v, ok := lookup(smp.Sum, mRunning...); ok {
			haveRun = true
			runMax = max(runMax, v)
		}
		if v, ok := lookup(smp.Sum, mWaiting...); ok {
			haveWait = true
			waitMax = max(waitMax, v)
			waitSum += v
		}
		if v, ok := lookup(smp.Max, mKV...); ok {
			haveKV = true
			kvMax = max(kvMax, v)
			kvSum += v
		}
	}
	if nIn > 0 {
		if haveRun {
			st.RunningMax = fp(runMax)
		}
		if haveWait {
			st.WaitingMax = fp(waitMax)
			st.WaitingMean = fp(waitSum / float64(nIn))
		}
		if haveKV {
			st.KVUsageMaxPct = fp(kvMax * 100)
			st.KVUsageMeanPct = fp(kvSum / float64(nIn) * 100)
		}
	}
	if v, ok := delta(mPreempt); ok {
		st.Preemptions = fp(v)
	}
	if q, ok := delta(mPrefixQ); ok && q > 0 {
		if h, ok := delta(mPrefixH); ok {
			st.PrefixHitRatePct = fp(h / q * 100)
		}
	}
	if dTok, ok := delta(mDraftTok); ok && dTok > 0 {
		if acc, ok := delta(mAccTok); ok {
			st.SpecAcceptPct = fp(acc / dTok * 100)
			if drafts, ok := delta(mDrafts); ok && drafts > 0 {
				st.SpecMeanAccLen = fp(1 + acc/drafts)
			}
		}
	}
	if dt > 0 {
		if v, ok := delta(mGenTok); ok {
			st.GenTokPerSec = fp(v / dt)
		}
		if v, ok := delta(mPromptTok); ok {
			st.PromptTokPerSec = fp(v / dt)
		}
	}
	return st
}

// TimelinePoint — для графика в HTML-отчёте.
type TimelinePoint struct {
	OffsetSec float64  `json:"t"`
	Running   *float64 `json:"running,omitempty"`
	Waiting   *float64 `json:"waiting,omitempty"`
	KVPct     *float64 `json:"kv,omitempty"`
}

func (s *Scraper) Timeline(start time.Time) []TimelinePoint {
	var out []TimelinePoint
	for _, smp := range s.Samples() {
		p := TimelinePoint{OffsetSec: smp.T.Sub(start).Seconds()}
		if v, ok := lookup(smp.Sum, mRunning...); ok {
			p.Running = fp(v)
		}
		if v, ok := lookup(smp.Sum, mWaiting...); ok {
			p.Waiting = fp(v)
		}
		if v, ok := lookup(smp.Max, mKV...); ok {
			p.KVPct = fp(v * 100)
		}
		out = append(out, p)
	}
	return out
}
