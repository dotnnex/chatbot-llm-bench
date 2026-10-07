package main

import (
	"fmt"
	"sort"
	"time"
)

type Dist struct {
	N    int     `json:"n"`
	Mean float64 `json:"mean"`
	Min  float64 `json:"min"`
	P50  float64 `json:"p50"`
	P90  float64 `json:"p90"`
	P95  float64 `json:"p95"`
	P99  float64 `json:"p99"`
	Max  float64 `json:"max"`
}

func pct(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := p / 100 * float64(len(sorted)-1)
	lo := int(rank)
	if lo+1 >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	return sorted[lo] + (sorted[lo+1]-sorted[lo])*(rank-float64(lo))
}

func NewDist(v []float64) Dist {
	if len(v) == 0 {
		return Dist{}
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	var sum float64
	for _, x := range s {
		sum += x
	}
	return Dist{
		N: len(s), Mean: sum / float64(len(s)), Min: s[0], Max: s[len(s)-1],
		P50: pct(s, 50), P90: pct(s, 90), P95: pct(s, 95), P99: pct(s, 99),
	}
}

type TurnSummary struct {
	Turn             int     `json:"turn"`
	Requests         int     `json:"requests"`
	Errors           int     `json:"errors"`
	PromptTokensMean float64 `json:"prompt_tokens_mean"`
	CachedTokensMean float64 `json:"cached_tokens_mean"`
	TTFTp50          float64 `json:"ttft_p50_ms"`
	TTFTp95          float64 `json:"ttft_p95_ms"`
	TPOTp50          float64 `json:"tpot_p50_ms"`
	TPOTp95          float64 `json:"tpot_p95_ms"`
	E2Ep50           float64 `json:"e2e_p50_ms"`
}

type StageSummary struct {
	Stage            int               `json:"stage"`
	Concurrency      int               `json:"concurrency"`
	DurationSec      float64           `json:"duration_sec"`
	Requests         int               `json:"requests"`
	OK               int               `json:"ok"`
	Errors           int               `json:"errors"`
	ErrorRatePct     float64           `json:"error_rate_pct"`
	ErrorKinds       map[string]int    `json:"error_kinds,omitempty"`
	ErrorSamples     map[string]string `json:"error_samples,omitempty"`
	DialogsStarted   int64             `json:"dialogs_started"`
	DialogsCompleted int64             `json:"dialogs_completed"`
	MaxTurn          int               `json:"max_turn"`

	RPS             float64 `json:"rps"`
	InputTokPerSec  float64 `json:"input_tok_per_sec"`
	OutputTokPerSec float64 `json:"output_tok_per_sec"`
	TotalPromptTok  int64   `json:"total_prompt_tokens"`
	TotalOutputTok  int64   `json:"total_output_tokens"`
	TokensPerChunk  float64 `json:"tokens_per_chunk"`
	CachedPct       float64 `json:"cached_prompt_pct"`
	GoodputPct      float64 `json:"goodput_pct"`
	FinishLengthPct float64 `json:"finish_length_pct"`
	UsageEstimated  bool    `json:"usage_estimated,omitempty"`

	TTFT         Dist `json:"ttft_ms"`
	TPOT         Dist `json:"tpot_ms"`
	ITL          Dist `json:"itl_ms"`
	E2E          Dist `json:"e2e_ms"`
	StreamTokSec Dist `json:"stream_tok_per_sec"`
	PromptTokens Dist `json:"prompt_tokens"`
	OutputTokens Dist `json:"output_tokens"`

	ByTurn []TurnSummary `json:"by_turn"`
	Server *ServerStats  `json:"server,omitempty"`
}

func Summarize(st *StageRun, all []*Result, cfg *Config, srv *ServerStats) StageSummary {
	winStart := st.Start.Add(cfg.Warmup)
	if !winStart.Before(st.End) {
		winStart = st.Start
	}
	span := st.End.Sub(winStart).Seconds()
	s := StageSummary{
		Stage: st.Index, Concurrency: st.Concurrency, DurationSec: span,
		DialogsStarted: st.dialogsStarted.Load(), DialogsCompleted: st.dialogsCompleted.Load(),
		MaxTurn:    int(st.maxTurnReached.Load()),
		ErrorKinds: map[string]int{}, ErrorSamples: map[string]string{},
		Server: srv,
	}
	var ttft, tpot, itl, e2e, speed, ptok, otok []float64
	var chunks, cached int64
	var good, finishLen int
	byTurn := map[int][]*Result{}
	sloTTFT, sloTPOT := ms(cfg.SLOTTFT), ms(cfg.SLOTPOT)

	for _, r := range all {
		if r.ErrKind == "canceled" || r.StartTime.Before(winStart) {
			continue
		}
		s.Requests++
		byTurn[r.Turn] = append(byTurn[r.Turn], r)
		if !r.OK {
			s.Errors++
			s.ErrorKinds[r.ErrKind]++
			if _, ok := s.ErrorSamples[r.ErrKind]; !ok {
				s.ErrorSamples[r.ErrKind] = r.ErrMsg
			}
			continue
		}
		s.OK++
		s.TotalPromptTok += int64(r.PromptTokens)
		s.TotalOutputTok += int64(r.CompletionTokens)
		chunks += int64(r.Chunks)
		cached += int64(r.CachedTokens)
		s.UsageEstimated = s.UsageEstimated || r.UsageEstimated
		if r.FinishReason == "length" {
			finishLen++
		}
		ttft = append(ttft, r.TTFTms)
		e2e = append(e2e, r.E2Ems)
		ptok = append(ptok, float64(r.PromptTokens))
		otok = append(otok, float64(r.CompletionTokens))
		if r.CompletionTokens > 1 {
			tpot = append(tpot, r.TPOTms)
			if dec := (r.E2E - r.TTFT).Seconds(); dec > 0 {
				speed = append(speed, float64(r.CompletionTokens-1)/dec)
			}
		}
		for _, v := range r.ITLs {
			itl = append(itl, float64(v))
		}
		if r.TTFTms <= sloTTFT && (r.CompletionTokens <= 1 || r.TPOTms <= sloTPOT) {
			good++
		}
	}
	if s.Requests > 0 {
		s.ErrorRatePct = float64(s.Errors) / float64(s.Requests) * 100
		s.GoodputPct = float64(good) / float64(s.Requests) * 100
	}
	if s.OK > 0 {
		s.FinishLengthPct = float64(finishLen) / float64(s.OK) * 100
	}
	if span > 0 {
		s.RPS = float64(s.OK) / span
		s.InputTokPerSec = float64(s.TotalPromptTok) / span
		s.OutputTokPerSec = float64(s.TotalOutputTok) / span
	}
	if chunks > 0 {
		s.TokensPerChunk = float64(s.TotalOutputTok) / float64(chunks)
	}
	if s.TotalPromptTok > 0 {
		s.CachedPct = float64(cached) / float64(s.TotalPromptTok) * 100
	}
	s.TTFT, s.TPOT, s.ITL, s.E2E = NewDist(ttft), NewDist(tpot), NewDist(itl), NewDist(e2e)
	s.StreamTokSec, s.PromptTokens, s.OutputTokens = NewDist(speed), NewDist(ptok), NewDist(otok)

	turns := make([]int, 0, len(byTurn))
	for t := range byTurn {
		turns = append(turns, t)
	}
	sort.Ints(turns)
	for _, t := range turns {
		var tt, tp, ee, pt, ct []float64
		ts := TurnSummary{Turn: t}
		for _, r := range byTurn[t] {
			ts.Requests++
			if !r.OK {
				ts.Errors++
				continue
			}
			tt = append(tt, r.TTFTms)
			ee = append(ee, r.E2Ems)
			pt = append(pt, float64(r.PromptTokens))
			ct = append(ct, float64(r.CachedTokens))
			if r.CompletionTokens > 1 {
				tp = append(tp, r.TPOTms)
			}
		}
		dt, dp, de := NewDist(tt), NewDist(tp), NewDist(ee)
		ts.TTFTp50, ts.TTFTp95, ts.TPOTp50, ts.TPOTp95, ts.E2Ep50 = dt.P50, dt.P95, dp.P50, dp.P95, de.P50
		ts.PromptTokensMean, ts.CachedTokensMean = NewDist(pt).Mean, NewDist(ct).Mean
		s.ByTurn = append(s.ByTurn, ts)
	}
	return s
}

// --- выводы ---

type Analysis struct {
	BestThroughputConc int      `json:"best_throughput_concurrency"`
	BestThroughput     float64  `json:"best_output_tok_per_sec"`
	SaturationConc     int      `json:"saturation_concurrency,omitempty"`
	MaxSLOConc         int      `json:"max_concurrency_within_slo"`
	Notes              []string `json:"notes"`
}

func Analyze(stages []StageSummary, cfg *Config) Analysis {
	a := Analysis{}
	if len(stages) == 0 {
		a.Notes = append(a.Notes, "Нет завершённых ступеней.")
		return a
	}
	for _, s := range stages {
		if s.OutputTokPerSec > a.BestThroughput {
			a.BestThroughput, a.BestThroughputConc = s.OutputTokPerSec, s.Concurrency
		}
		if s.OK > 0 && s.ErrorRatePct < 1 && s.TTFT.P95 <= ms(cfg.SLOTTFT) && s.TPOT.P95 <= ms(cfg.SLOTPOT) {
			a.MaxSLOConc = s.Concurrency
		}
	}
	a.Notes = append(a.Notes, fmt.Sprintf("Максимальная пропускная способность: %.0f вых. tok/s при %d одновременных диалогах.",
		a.BestThroughput, a.BestThroughputConc))

	if a.MaxSLOConc > 0 {
		a.Notes = append(a.Notes, fmt.Sprintf("SLO (p95 TTFT ≤ %v, p95 TPOT ≤ %v, ошибок < 1%%) выдерживается до %d диалогов включительно.",
			cfg.SLOTTFT, cfg.SLOTPOT, a.MaxSLOConc))
	} else {
		a.Notes = append(a.Notes, fmt.Sprintf("SLO (p95 TTFT ≤ %v, p95 TPOT ≤ %v) не выполнен ни на одной ступени.", cfg.SLOTTFT, cfg.SLOTPOT))
	}

	// насыщение: рост нагрузки дал < 10% прироста throughput
	for i := 1; i < len(stages); i++ {
		p, c := stages[i-1], stages[i]
		if p.OutputTokPerSec <= 0 {
			continue
		}
		loadGain := float64(c.Concurrency)/float64(p.Concurrency) - 1
		tputGain := c.OutputTokPerSec/p.OutputTokPerSec - 1
		if loadGain > 0 && tputGain < 0.10 {
			a.SaturationConc = p.Concurrency
			a.Notes = append(a.Notes, fmt.Sprintf("Насыщение: при переходе %d → %d диалогов throughput вырос лишь на %.0f%% (нагрузка +%.0f%%), TTFT p50 %.0f → %.0f мс. Практический предел ≈ %d диалогов.",
				p.Concurrency, c.Concurrency, tputGain*100, loadGain*100, p.TTFT.P50, c.TTFT.P50, p.Concurrency))
			break
		}
	}
	if a.SaturationConc == 0 && len(stages) > 1 {
		a.Notes = append(a.Notes, "Явного насыщения throughput не обнаружено — можно повышать --end.")
	}

	// деградация с ростом контекста (на последней ступени, где был хотя бы 3-й ход)
	for i := len(stages) - 1; i >= 0; i-- {
		bt := stages[i].ByTurn
		if len(bt) >= 3 && bt[0].TTFTp50 > 0 {
			l := bt[len(bt)-1]
			a.Notes = append(a.Notes, fmt.Sprintf("Рост контекста (ступень %d диалогов): ход 1 → ход %d — prompt %.0f → %.0f ток., TTFT p50 %.0f → %.0f мс (×%.1f), TPOT p50 %.1f → %.1f мс.",
				stages[i].Concurrency, l.Turn, bt[0].PromptTokensMean, l.PromptTokensMean, bt[0].TTFTp50, l.TTFTp50, l.TTFTp50/bt[0].TTFTp50, bt[0].TPOTp50, l.TPOTp50))
			break
		}
	}

	last := stages[len(stages)-1]
	if srv := last.Server; srv != nil {
		if srv.SpecAcceptPct != nil {
			n := fmt.Sprintf("Speculative decoding: acceptance %.1f%%", *srv.SpecAcceptPct)
			if srv.SpecMeanAccLen != nil {
				n += fmt.Sprintf(", средняя длина принятия %.2f ток./шаг", *srv.SpecMeanAccLen)
			}
			a.Notes = append(a.Notes, n+" (последняя ступень).")
		}
		if srv.Preemptions != nil && *srv.Preemptions > 0 {
			a.Notes = append(a.Notes, fmt.Sprintf("Внимание: %.0f preemption-ов на последней ступени — не хватает KV-cache.", *srv.Preemptions))
		}
		if srv.KVUsageMaxPct != nil && *srv.KVUsageMaxPct > 90 {
			a.Notes = append(a.Notes, fmt.Sprintf("KV-cache заполнялся до %.0f%% — память GPU — вероятный ограничитель.", *srv.KVUsageMaxPct))
		}
		if srv.WaitingMax != nil && *srv.WaitingMax > 0 {
			a.Notes = append(a.Notes, fmt.Sprintf("Очередь vLLM на последней ступени: до %.0f ожидающих запросов.", *srv.WaitingMax))
		}
	}
	if last.TokensPerChunk > 1.2 {
		a.Notes = append(a.Notes, fmt.Sprintf("В среднем %.2f токена на SSE-чанк — сервер отдаёт пачками (признак работающего speculative decoding).", last.TokensPerChunk))
	}
	if last.FinishLengthPct > 50 && !cfg.IgnoreEOS {
		a.Notes = append(a.Notes, fmt.Sprintf("%.0f%% ответов обрезаны по max_tokens=%d.", last.FinishLengthPct, cfg.MaxTokens))
	}
	for _, s := range stages {
		if s.ErrorRatePct > 0 {
			a.Notes = append(a.Notes, fmt.Sprintf("Ошибки начались со ступени %d диалогов (%.1f%%).", s.Concurrency, s.ErrorRatePct))
			break
		}
	}
	return a
}

func fmtDur(sec float64) string {
	return (time.Duration(sec * float64(time.Second))).Truncate(time.Second).String()
}
