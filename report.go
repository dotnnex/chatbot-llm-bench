package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

type Meta struct {
	Model         string    `json:"model"`
	URL           string    `json:"url"`
	StartedAt     time.Time `json:"started_at"`
	FinishedAt    time.Time `json:"finished_at"`
	MaxModelLen   int       `json:"max_model_len,omitempty"`
	MaxContext    int       `json:"max_context"`
	CharsPerToken float64   `json:"chars_per_token"`
	CPTSource     string    `json:"chars_per_token_source"`
	MetricsURL    string    `json:"metrics_url,omitempty"`
	Corpus        string    `json:"corpus"`
	StoppedReason string    `json:"stopped_reason,omitempty"`
}

type Summary struct {
	Meta     Meta              `json:"meta"`
	Config   map[string]string `json:"config"`
	Stages   []StageSummary    `json:"stages"`
	Analysis Analysis          `json:"analysis"`
	Timeline []TimelinePoint   `json:"server_timeline,omitempty"`
}

func na(p *float64, format string) string {
	if p == nil {
		return "—"
	}
	return fmt.Sprintf(format, *p)
}

func f0(v float64) string { return strconv.FormatFloat(v, 'f', 0, 64) }
func f1(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) }
func f2(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) }

func PrintStageLine(s StageSummary) {
	srv := ""
	if s.Server != nil {
		srv = fmt.Sprintf(" | KV max %s, очередь max %s, spec acc %s",
			na(s.Server.KVUsageMaxPct, "%.0f%%"), na(s.Server.WaitingMax, "%.0f"), na(s.Server.SpecAcceptPct, "%.0f%%"))
	}
	fmt.Printf("✔ Ступень %d: %d диалогов за %s | запросов %d, ошибок %.1f%% | RPS %.2f | вых %.0f tok/s | TTFT p50/p95 %.0f/%.0f мс | TPOT p50/p95 %.1f/%.1f мс%s\n\n",
		s.Stage, s.Concurrency, fmtDur(s.DurationSec), s.Requests, s.ErrorRatePct, s.RPS, s.OutputTokPerSec,
		s.TTFT.P50, s.TTFT.P95, s.TPOT.P50, s.TPOT.P95, srv)
}

func table(w io.Writer, title string, header []string, rows [][]string) {
	fmt.Fprintf(w, "\n▸ %s\n", title)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, strings.Join(header, "\t")+"\t")
	for _, r := range rows {
		fmt.Fprintln(tw, strings.Join(r, "\t")+"\t")
	}
	tw.Flush()
}

func pickTurns(maxTurn, limit int) []int {
	if maxTurn <= limit {
		out := make([]int, maxTurn)
		for i := range out {
			out[i] = i + 1
		}
		return out
	}
	seen := map[int]bool{}
	var out []int
	for i := 0; i < limit; i++ {
		t := 1 + int(float64(i)*float64(maxTurn-1)/float64(limit-1)+0.5)
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

func PrintReport(w io.Writer, sum *Summary) {
	st := sum.Stages
	m := sum.Meta
	line := strings.Repeat("═", 100)
	fmt.Fprintf(w, "\n%s\n ИТОГИ НАГРУЗОЧНОГО ТЕСТА · модель %s · %s · длительность %s\n%s\n",
		line, m.Model, m.URL, m.FinishedAt.Sub(m.StartedAt).Truncate(time.Second), line)
	if m.StoppedReason != "" {
		fmt.Fprintf(w, " Тест остановлен досрочно: %s\n", m.StoppedReason)
	}
	if len(st) == 0 {
		fmt.Fprintln(w, " Нет данных.")
		return
	}

	var rows [][]string
	for _, s := range st {
		rows = append(rows, []string{
			strconv.Itoa(s.Concurrency), fmtDur(s.DurationSec), strconv.Itoa(s.Requests), f1(s.ErrorRatePct) + "%",
			f2(s.RPS), f0(s.InputTokPerSec), f0(s.OutputTokPerSec),
			f1(s.StreamTokSec.P50), f1(s.StreamTokSec.Min), f1(s.GoodputPct) + "%",
		})
	}
	table(w, "Пропускная способность", []string{"Диалоги", "Время", "Запросов", "Ошибки", "RPS", "Вход tok/s", "Выход tok/s", "tok/s/поток p50", "мин", "Goodput"}, rows)

	rows = nil
	for _, s := range st {
		rows = append(rows, []string{
			strconv.Itoa(s.Concurrency),
			f0(s.TTFT.P50), f0(s.TTFT.P90), f0(s.TTFT.P95), f0(s.TTFT.P99), f0(s.TTFT.Max),
			f1(s.TPOT.P50), f1(s.TPOT.P95), f1(s.TPOT.P99),
			f1(s.ITL.P50), f1(s.ITL.P99), f0(s.ITL.Max),
			f0(s.E2E.P50), f0(s.E2E.P95), f0(s.E2E.Max),
		})
	}
	table(w, "Задержки, мс  (TTFT — до первого токена, TPOT — на выходной токен, ITL — между чанками, E2E — весь ответ)",
		[]string{"Диалоги", "TTFT p50", "p90", "p95", "p99", "max", "TPOT p50", "p95", "p99", "ITL p50", "p99", "max", "E2E p50", "p95", "max"}, rows)

	rows = nil
	for _, s := range st {
		cached := "—"
		if s.CachedPct > 0 {
			cached = f1(s.CachedPct) + "%"
		}
		rows = append(rows, []string{
			strconv.Itoa(s.Concurrency), f0(s.PromptTokens.Mean), f0(s.PromptTokens.P95), f0(s.PromptTokens.Max),
			f0(s.OutputTokens.Mean), f1(s.FinishLengthPct) + "%", f2(s.TokensPerChunk), cached,
			strconv.Itoa(s.MaxTurn), fmt.Sprintf("%d/%d", s.DialogsStarted, s.DialogsCompleted),
		})
	}
	table(w, "Контекст и диалоги", []string{"Диалоги", "Prompt ср.", "p95", "max", "Ответ ср.", "Обрезано", "Ток/чанк", "Cached", "Макс ход", "Начато/заверш."}, rows)

	hasSrv := false
	for _, s := range st {
		hasSrv = hasSrv || s.Server != nil
	}
	if hasSrv {
		rows = nil
		for _, s := range st {
			v := s.Server
			if v == nil {
				v = &ServerStats{}
			}
			rows = append(rows, []string{
				strconv.Itoa(s.Concurrency), na(v.RunningMax, "%.0f"), na(v.WaitingMax, "%.0f"), na(v.WaitingMean, "%.1f"),
				na(v.KVUsageMaxPct, "%.1f%%"), na(v.KVUsageMeanPct, "%.1f%%"), na(v.Preemptions, "%.0f"),
				na(v.PrefixHitRatePct, "%.1f%%"), na(v.SpecAcceptPct, "%.1f%%"), na(v.SpecMeanAccLen, "%.2f"),
				na(v.PromptTokPerSec, "%.0f"), na(v.GenTokPerSec, "%.0f"),
			})
		}
		table(w, "Сервер vLLM (/metrics)", []string{"Диалоги", "Running max", "Waiting max", "Waiting ср.", "KV max", "KV ср.", "Preempt", "Prefix hit", "Spec accept", "Accept len", "Prompt tok/s", "Gen tok/s"}, rows)
	}

	maxTurn := 0
	for _, s := range st {
		for _, t := range s.ByTurn {
			maxTurn = max(maxTurn, t.Turn)
		}
	}
	if maxTurn > 1 {
		turns := pickTurns(maxTurn, 14)
		matrix := func(title string, val func(TurnSummary) string) {
			header := []string{"Диалоги \\ ход"}
			for _, t := range turns {
				header = append(header, strconv.Itoa(t))
			}
			var rows [][]string
			for _, s := range st {
				byT := map[int]TurnSummary{}
				for _, t := range s.ByTurn {
					byT[t.Turn] = t
				}
				row := []string{strconv.Itoa(s.Concurrency)}
				for _, t := range turns {
					if ts, ok := byT[t]; ok && ts.Requests > ts.Errors {
						row = append(row, val(ts))
					} else {
						row = append(row, "·")
					}
				}
				rows = append(rows, row)
			}
			table(w, title, header, rows)
		}
		matrix("Средний размер prompt (токенов) по номеру хода в диалоге", func(t TurnSummary) string { return f0(t.PromptTokensMean) })
		matrix("TTFT p50 (мс) по номеру хода — влияние роста контекста", func(t TurnSummary) string { return f0(t.TTFTp50) })
		matrix("TPOT p50 (мс) по номеру хода", func(t TurnSummary) string { return f1(t.TPOTp50) })
	}

	var errLines []string
	for _, s := range st {
		kinds := make([]string, 0, len(s.ErrorKinds))
		for k := range s.ErrorKinds {
			kinds = append(kinds, k)
		}
		sort.Strings(kinds)
		for _, k := range kinds {
			errLines = append(errLines, fmt.Sprintf("  %4d диалогов: %-18s ×%-5d %s", s.Concurrency, k, s.ErrorKinds[k], truncate(s.ErrorSamples[k], 120)))
		}
	}
	if len(errLines) > 0 {
		fmt.Fprintln(w, "\n▸ Ошибки")
		fmt.Fprintln(w, strings.Join(errLines, "\n"))
	}

	fmt.Fprintln(w, "\n▸ Выводы")
	for _, n := range sum.Analysis.Notes {
		fmt.Fprintf(w, "  • %s\n", n)
	}
	fmt.Fprintln(w)
}

func WriteFiles(dir string, sum *Summary, results []*Result) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(sum, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "summary.json"), b, 0o644); err != nil {
		return err
	}

	f, err := os.Create(filepath.Join(dir, "requests.jsonl"))
	if err != nil {
		return err
	}
	bw := bufio.NewWriter(f)
	enc := json.NewEncoder(bw)
	for _, r := range results {
		if err := enc.Encode(r); err != nil {
			f.Close()
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	f.Close()

	cf, err := os.Create(filepath.Join(dir, "stages.csv"))
	if err != nil {
		return err
	}
	cw := csv.NewWriter(cf)
	cw.Write([]string{"stage", "concurrency", "duration_sec", "requests", "errors", "error_rate_pct", "rps",
		"input_tok_s", "output_tok_s", "goodput_pct",
		"ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99", "ttft_max",
		"tpot_p50", "tpot_p95", "tpot_p99", "itl_p50", "itl_p99", "e2e_p50", "e2e_p95", "e2e_p99",
		"stream_tok_s_p50", "prompt_tokens_mean", "prompt_tokens_max", "output_tokens_mean", "tokens_per_chunk",
		"kv_max_pct", "waiting_max", "preemptions", "prefix_hit_pct", "spec_accept_pct", "spec_accept_len"})
	for _, s := range sum.Stages {
		v := s.Server
		if v == nil {
			v = &ServerStats{}
		}
		opt := func(p *float64) string {
			if p == nil {
				return ""
			}
			return f2(*p)
		}
		cw.Write([]string{strconv.Itoa(s.Stage), strconv.Itoa(s.Concurrency), f1(s.DurationSec), strconv.Itoa(s.Requests),
			strconv.Itoa(s.Errors), f2(s.ErrorRatePct), f2(s.RPS), f1(s.InputTokPerSec), f1(s.OutputTokPerSec), f2(s.GoodputPct),
			f1(s.TTFT.P50), f1(s.TTFT.P90), f1(s.TTFT.P95), f1(s.TTFT.P99), f1(s.TTFT.Max),
			f2(s.TPOT.P50), f2(s.TPOT.P95), f2(s.TPOT.P99), f2(s.ITL.P50), f2(s.ITL.P99),
			f1(s.E2E.P50), f1(s.E2E.P95), f1(s.E2E.P99),
			f1(s.StreamTokSec.P50), f1(s.PromptTokens.Mean), f0(s.PromptTokens.Max), f1(s.OutputTokens.Mean), f2(s.TokensPerChunk),
			opt(v.KVUsageMaxPct), opt(v.WaitingMax), opt(v.Preemptions), opt(v.PrefixHitRatePct), opt(v.SpecAcceptPct), opt(v.SpecMeanAccLen)})
	}
	cw.Flush()
	cf.Close()
	if err := cw.Error(); err != nil {
		return err
	}
	return WriteHTML(filepath.Join(dir, "report.html"), sum, results)
}
