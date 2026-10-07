package main

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// Collector хранит результаты всех запросов.
type Collector struct {
	mu      sync.Mutex
	start   time.Time
	results []*Result
}

func (c *Collector) Add(r *Result) {
	r.OffsetSec = r.StartTime.Sub(c.start).Seconds()
	c.mu.Lock()
	c.results = append(c.results, r)
	c.mu.Unlock()
}

// Since возвращает результаты, добавленные после позиции from, и новую позицию.
func (c *Collector) Since(from int) ([]*Result, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*Result(nil), c.results[from:]...), len(c.results)
}

func (c *Collector) All() []*Result {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*Result(nil), c.results...)
}

func (c *Collector) Stage(idx int) []*Result {
	var out []*Result
	for _, r := range c.All() {
		if r.Stage == idx {
			out = append(out, r)
		}
	}
	return out
}

type StageRun struct {
	Index            int
	Concurrency      int
	Start, End       time.Time
	remaining        atomic.Int64 // режим «по диалогам»: сколько диалогов ещё можно начать
	dialogsStarted   atomic.Int64
	dialogsCompleted atomic.Int64
	activeDialogs    atomic.Int64
	maxTurnReached   atomic.Int64
}

type Runner struct {
	cfg        *Config
	client     *Client
	corpus     *Corpus
	calib      *Calibrator
	collector  *Collector
	maxContext int
	dialogSeq  atomic.Int64
	warnedCtx  atomic.Bool
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func (r *Runner) thinkTime(rng *rand.Rand) time.Duration {
	d := float64(r.cfg.ThinkTime)
	if j := r.cfg.ThinkJitter; j > 0 {
		d *= 1 + j*(2*rng.Float64()-1)
	}
	return time.Duration(d)
}

// RunStage держит conc одновременных диалогов до окончания ступени.
func (r *Runner) RunStage(ctx context.Context, idx, total, conc int) *StageRun {
	st := &StageRun{Index: idx, Concurrency: conc, Start: time.Now()}
	var issueCtx context.Context
	var cancel context.CancelFunc
	if r.cfg.StageDialogs > 0 {
		st.remaining.Store(int64(r.cfg.StageDialogs))
		issueCtx, cancel = context.WithCancel(ctx)
	} else {
		issueCtx, cancel = context.WithDeadline(ctx, st.Start.Add(r.cfg.StageDuration))
	}
	defer cancel()

	progDone := make(chan struct{})
	progStop := make(chan struct{})
	go r.progress(st, total, progStop, progDone)

	stagger := r.cfg.Stagger
	if stagger < 0 {
		stagger = r.cfg.ThinkTime
	}
	var wg sync.WaitGroup
	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(r.cfg.Seed + int64(idx)*1_000_003 + int64(w)*7919))
			if stagger > 0 && !sleepCtx(issueCtx, time.Duration(rng.Int63n(int64(stagger)))) {
				return
			}
			for issueCtx.Err() == nil {
				if r.cfg.StageDialogs > 0 && st.remaining.Add(-1) < 0 {
					return
				}
				r.runDialog(ctx, issueCtx, st, rng)
			}
		}(w)
	}
	wg.Wait()
	st.End = time.Now()
	close(progStop)
	<-progDone
	return st
}

func (r *Runner) runDialog(ctx, issueCtx context.Context, st *StageRun, rng *rand.Rand) {
	did := r.dialogSeq.Add(1)
	st.dialogsStarted.Add(1)
	st.activeDialogs.Add(1)
	defer st.activeDialogs.Add(-1)

	cur := r.corpus.NewCursor(rng)
	var msgs []Message
	histChars := 0
	if r.cfg.SystemPrompt != "" {
		msgs = append(msgs, Message{Role: "system", Content: r.cfg.SystemPrompt})
		histChars += utf8.RuneCountInString(r.cfg.SystemPrompt)
	}

	for turn := 1; r.cfg.MaxTurns <= 0 || turn <= r.cfg.MaxTurns; turn++ {
		if issueCtx.Err() != nil {
			return
		}
		cpt := r.calib.CharsPerToken()
		target := r.cfg.AddTokens
		if turn == 1 {
			target = r.cfg.InitialTokens
		}
		user := BuildUserMessage(cur, rng, turn == 1, target, cpt)
		userChars := utf8.RuneCountInString(user)

		if r.maxContext > 0 {
			est := int(float64(histChars+userChars)/cpt) + 8*(len(msgs)+1) + r.cfg.MaxTokens
			if est > r.maxContext {
				if turn == 1 && r.warnedCtx.CompareAndSwap(false, true) {
					fmt.Printf("  ! стартовое сообщение (~%d ток.) + max_tokens не влезает в max-context=%d\n", est, r.maxContext)
				}
				break // контекст исчерпан — диалог завершён, начинаем новый
			}
		}

		msgs = append(msgs, Message{Role: "user", Content: user})
		res, answer := r.client.Chat(ctx, msgs)
		res.Stage, res.Concurrency, res.DialogID, res.Turn = st.Index, st.Concurrency, did, turn
		r.collector.Add(res)
		if res.ErrKind == "canceled" {
			return
		}
		if !res.OK {
			// как реальный пользователь: после ошибки пауза и новый диалог
			sleepCtx(issueCtx, r.thinkTime(rng))
			return
		}
		r.calib.Observe(histChars+userChars, res.PromptTokens, len(msgs))
		for {
			m := st.maxTurnReached.Load()
			if int64(turn) <= m || st.maxTurnReached.CompareAndSwap(m, int64(turn)) {
				break
			}
		}
		if answer == "" {
			answer = "…"
		}
		msgs = append(msgs, Message{Role: "assistant", Content: answer})
		histChars += userChars + utf8.RuneCountInString(answer)

		if !sleepCtx(issueCtx, r.thinkTime(rng)) {
			return
		}
	}
	st.dialogsCompleted.Add(1)
}

func (r *Runner) progress(st *StageRun, total int, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	t := time.NewTicker(r.cfg.ReportInterval)
	defer t.Stop()
	_, pos := r.collector.Since(0)
	last := time.Now()
	var reqs, errs int
	for {
		select {
		case <-stop:
			return
		case now := <-t.C:
			batch, np := r.collector.Since(pos)
			pos = np
			var ttft []float64
			var outTok int
			newErr := 0
			for _, res := range batch {
				if res.OK {
					ttft = append(ttft, res.TTFTms)
					outTok += res.CompletionTokens
				} else if res.ErrKind != "canceled" {
					newErr++
				}
			}
			reqs += len(batch)
			errs += newErr
			sort.Float64s(ttft)
			p50 := "—"
			if len(ttft) > 0 {
				p50 = fmt.Sprintf("%.0fms", pct(ttft, 50))
			}
			elapsed := now.Sub(st.Start).Truncate(time.Second)
			var when string
			if r.cfg.StageDialogs > 0 {
				left := max(st.remaining.Load(), 0)
				when = fmt.Sprintf("%v, диалогов осталось %d", elapsed, left)
			} else {
				when = fmt.Sprintf("%v/%v", elapsed, r.cfg.StageDuration)
			}
			fmt.Printf("  [%d/%d · %d дл.] %s | активно %d, в полёте %d | запросов %d (+%d) ошибок %d | TTFT p50 %s | вых %.0f tok/s | ход ≤%d\n",
				st.Index, total, st.Concurrency, when, st.activeDialogs.Load(), r.client.Inflight.Load(),
				reqs, len(batch), errs, p50, float64(outTok)/now.Sub(last).Seconds(), st.maxTurnReached.Load())
			last = now
		}
	}
}
