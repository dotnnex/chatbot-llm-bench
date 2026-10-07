// llm-loadtest — нагрузочное тестирование LLM-чат-бота (OpenAI-совместимый API, vLLM)
// многоходовыми диалогами на естественном русском языке.
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

type Config struct {
	URL            string
	APIKey         string
	Model          string
	StartDialogs   int
	EndDialogs     int
	Step           int
	InitialTokens  int
	AddTokens      int
	ThinkTime      time.Duration
	ThinkJitter    float64
	Stagger        time.Duration
	StageDuration  time.Duration
	StageDialogs   int
	MaxTurns       int
	MaxContext     int
	MaxTokens      int
	Temperature    float64
	IgnoreEOS      bool
	ExtraBody      string
	SystemPrompt   string
	CorpusPath     string
	CharsPerToken  float64
	RequestTimeout time.Duration
	Warmup         time.Duration
	Cooldown       time.Duration
	SLOTTFT        time.Duration
	SLOTPOT        time.Duration
	MaxErrorRate   float64
	StopOnSLO      bool
	MetricsURL     string
	OutDir         string
	Seed           int64
	Insecure       bool
	ReportInterval time.Duration
	PrintSample    bool
}

const defaultSystemPrompt = "Ты — вежливый и внимательный ассистент-собеседник. Отвечай по-русски, развёрнуто и по существу, опираясь на присланные пользователем фрагменты текста и на предыдущую часть разговора."

func parseFlags() *Config {
	c := &Config{}
	fs := flag.CommandLine
	fs.StringVar(&c.URL, "url", "", "точка входа OpenAI-совместимого API: http://host:8000 | .../v1 | .../v1/chat/completions (обязательно)")
	fs.StringVar(&c.APIKey, "api-key", os.Getenv("LLM_API_KEY"), "секретный ключ (Bearer); по умолчанию из env LLM_API_KEY")
	fs.StringVar(&c.Model, "model", "", "имя модели; пусто — первая из /v1/models")

	fs.IntVar(&c.StartDialogs, "start", 1, "с какого количества одновременных диалогов начинаем")
	fs.IntVar(&c.EndDialogs, "end", 16, "каким количеством одновременных диалогов заканчиваем")
	fs.IntVar(&c.Step, "step", 0, "шаг прироста диалогов между ступенями; 0 — удвоение (1,2,4,8…)")
	fs.IntVar(&c.InitialTokens, "initial-tokens", 500, "размер стартового сообщения пользователя, токенов")
	fs.IntVar(&c.AddTokens, "add-tokens", 200, "сколько токенов добавляет каждое следующее сообщение пользователя")
	fs.DurationVar(&c.ThinkTime, "think-time", 3*time.Second, "пауза между ответом модели и следующим сообщением в диалоге")
	fs.Float64Var(&c.ThinkJitter, "think-jitter", 0.3, "случайный разброс паузы, доля (0.3 = ±30%)")
	fs.DurationVar(&c.Stagger, "stagger", -1, "разнос старта диалогов в начале ступени; -1 — равен think-time")

	fs.DurationVar(&c.StageDuration, "stage-duration", 2*time.Minute, "длительность ступени (режим «по времени»)")
	fs.IntVar(&c.StageDialogs, "stage-dialogs", 0, "если > 0 — режим «по количеству диалогов»: ступень заканчивается, когда отработано столько диалогов")
	fs.IntVar(&c.MaxTurns, "max-turns", 10, "максимум ходов в диалоге (0 — без лимита, до заполнения контекста)")
	fs.IntVar(&c.MaxContext, "max-context", 0, "лимит контекста диалога, токенов; 0 — max_model_len модели из /v1/models")

	fs.IntVar(&c.MaxTokens, "max-tokens", 300, "max_tokens ответа модели")
	fs.Float64Var(&c.Temperature, "temperature", 0.7, "temperature")
	fs.BoolVar(&c.IgnoreEOS, "ignore-eos", false, "vLLM ignore_eos: всегда генерировать ровно max-tokens (стабильная длина ответа)")
	fs.StringVar(&c.ExtraBody, "extra-body", "", `доп. поля запроса JSON, напр. '{"chat_template_kwargs":{"enable_thinking":false}}'`)
	fs.StringVar(&c.SystemPrompt, "system-prompt", defaultSystemPrompt, "системный промпт (пусто — без него)")
	fs.StringVar(&c.CorpusPath, "corpus", "", "свой UTF-8 текстовый корпус вместо встроенного")
	fs.Float64Var(&c.CharsPerToken, "chars-per-token", 3.3, "начальная оценка символов на токен (уточняется через /tokenize или usage)")

	fs.DurationVar(&c.RequestTimeout, "request-timeout", 5*time.Minute, "таймаут одного запроса")
	fs.DurationVar(&c.Warmup, "warmup", 0, "не учитывать в статистике запросы, начатые в первые N секунд ступени")
	fs.DurationVar(&c.Cooldown, "cooldown", 5*time.Second, "пауза между ступенями")
	fs.DurationVar(&c.SLOTTFT, "slo-ttft", 2*time.Second, "SLO на TTFT (для goodput и выводов)")
	fs.DurationVar(&c.SLOTPOT, "slo-tpot", 100*time.Millisecond, "SLO на TPOT (время на выходной токен)")
	fs.Float64Var(&c.MaxErrorRate, "max-error-rate", 0.2, "остановить тест, если доля ошибок на ступени выше (0.2 = 20%)")
	fs.BoolVar(&c.StopOnSLO, "stop-on-slo", false, "остановить тест, когда p95 TTFT превысил SLO")

	fs.StringVar(&c.MetricsURL, "metrics-url", "auto", "Prometheus-метрики vLLM: auto (<host>/metrics) | URL | none")
	fs.StringVar(&c.OutDir, "out", "results", "каталог для отчётов (внутри создаётся подкаталог с датой)")
	fs.Int64Var(&c.Seed, "seed", 0, "seed генератора; 0 — случайный")
	fs.BoolVar(&c.Insecure, "insecure", false, "не проверять TLS-сертификат")
	fs.DurationVar(&c.ReportInterval, "report-interval", 5*time.Second, "период live-вывода")
	fs.BoolVar(&c.PrintSample, "print-sample", false, "показать пример первых двух сообщений пользователя и выйти (без запросов)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Нагрузочный тест LLM-чат-бота многоходовыми диалогами на русском.\n\nПример:\n  %s --url http://vllm:8000 --api-key $KEY --start 1 --end 64 --initial-tokens 800 --add-tokens 300 --think-time 5s\n\nПараметры:\n", filepath.Base(os.Args[0]))
		fs.PrintDefaults()
	}
	flag.Parse()

	fail := func(msg string) {
		fmt.Fprintln(os.Stderr, "ошибка:", msg)
		os.Exit(2)
	}
	switch {
	case c.URL == "" && !c.PrintSample:
		fail("не задан --url")
	case c.StartDialogs < 1:
		fail("--start должен быть ≥ 1")
	case c.EndDialogs < c.StartDialogs:
		fail("--end должен быть ≥ --start")
	case c.InitialTokens < 1 || c.AddTokens < 1:
		fail("--initial-tokens и --add-tokens должны быть ≥ 1")
	case c.StageDialogs <= 0 && c.StageDuration <= 0:
		fail("задайте --stage-duration > 0 или --stage-dialogs > 0")
	case c.StageDialogs > 0 && c.MaxTurns <= 0:
		fail("в режиме --stage-dialogs нужен --max-turns > 0")
	case c.ReportInterval <= 0:
		fail("--report-interval должен быть > 0")
	}
	if c.Seed == 0 {
		c.Seed = time.Now().UnixNano()
	}
	return c
}

func buildLevels(start, end, step int) []int {
	var l []int
	for c := start; c < end; {
		l = append(l, c)
		if step > 0 {
			c += step
		} else {
			c *= 2
		}
	}
	return append(l, end)
}

func configMap(c *Config) map[string]string {
	m := map[string]string{}
	flag.VisitAll(func(f *flag.Flag) {
		if f.Name == "api-key" {
			if f.Value.String() != "" {
				m[f.Name] = "***"
			}
			return
		}
		m[f.Name] = f.Value.String()
	})
	m["seed"] = fmt.Sprint(c.Seed)
	return m
}

func main() {
	cfg := parseFlags()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if cfg.PrintSample {
		printSample(cfg)
		return
	}

	client, err := NewClient(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ошибка:", err)
		os.Exit(2)
	}

	// модель и max_model_len
	models, err := client.ListModels(ctx)
	maxModelLen := 0
	switch {
	case err != nil && cfg.Model == "":
		fmt.Fprintf(os.Stderr, "ошибка: не удалось получить список моделей (%v); укажите --model\n", err)
		os.Exit(1)
	case err != nil:
		fmt.Printf("! /v1/models недоступен (%v), продолжаю с --model=%s\n", err, cfg.Model)
	default:
		if len(models) == 0 && cfg.Model == "" {
			fmt.Fprintln(os.Stderr, "ошибка: сервер не вернул ни одной модели; укажите --model")
			os.Exit(1)
		}
		if cfg.Model == "" {
			cfg.Model = models[0].ID
		}
		for _, m := range models {
			if m.ID == cfg.Model {
				maxModelLen = m.MaxModelLen
			}
		}
	}
	client.Model = cfg.Model
	maxContext := cfg.MaxContext
	if maxContext == 0 {
		maxContext = maxModelLen
	}

	corpus, err := LoadCorpus(cfg.CorpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ошибка корпуса:", err)
		os.Exit(1)
	}
	corpusName := "встроенный (русская классика)"
	if cfg.CorpusPath != "" {
		corpusName = cfg.CorpusPath
	}

	calib := NewCalibrator(cfg.CharsPerToken)
	sample := corpus.Sample(rand.New(rand.NewSource(cfg.Seed)), 20000)
	if n, err := client.Tokenize(ctx, sample); err == nil {
		calib.Fix(float64(len([]rune(sample)))/float64(n), "vLLM /tokenize")
	}

	// пробный запрос: ловим неверный ключ / URL / модель до начала теста
	pre, _ := client.Chat(ctx, []Message{{Role: "user", Content: "Привет! Как дела? Ответь одной фразой."}})
	if !pre.OK {
		fmt.Fprintf(os.Stderr, "ошибка: пробный запрос не прошёл: %s %s\n", pre.ErrKind, pre.ErrMsg)
		os.Exit(1)
	}

	var scraper *Scraper
	metricsURL := ""
	if mu := strings.TrimSpace(cfg.MetricsURL); mu != "" && mu != "none" {
		if mu == "auto" {
			mu = client.RootURL + "/metrics"
		}
		s := NewScraper(mu, cfg.APIKey, cfg.Insecure, 2*time.Second)
		if err := s.Probe(ctx); err != nil {
			fmt.Printf("! метрики vLLM недоступны (%s: %v) — серверная статистика отключена\n", mu, err)
		} else {
			scraper, metricsURL = s, mu
		}
	}

	levels := buildLevels(cfg.StartDialogs, cfg.EndDialogs, cfg.Step)
	mode := fmt.Sprintf("по времени, %v на ступень", cfg.StageDuration)
	if cfg.StageDialogs > 0 {
		mode = fmt.Sprintf("по количеству, %d диалогов на ступень", cfg.StageDialogs)
	}
	fmt.Printf("\nМодель:        %s (max_model_len %d, лимит контекста диалога %d)\n", cfg.Model, maxModelLen, maxContext)
	fmt.Printf("Endpoint:      %s\n", client.chatURL)
	fmt.Printf("Ступени:       %v (%s)\n", levels, mode)
	fmt.Printf("Диалог:        старт %d ток., +%d ток./ход, до %d ходов, пауза %v ±%.0f%%, max_tokens %d\n",
		cfg.InitialTokens, cfg.AddTokens, cfg.MaxTurns, cfg.ThinkTime, cfg.ThinkJitter*100, cfg.MaxTokens)
	fmt.Printf("Токенизация:   %.2f симв./токен (%s); корпус: %s, %d символов\n", calib.CharsPerToken(), calib.Source(), corpusName, corpus.Len())
	if metricsURL != "" {
		fmt.Printf("Метрики vLLM:  %s\n", metricsURL)
	}
	fmt.Println()

	testStart := time.Now()
	collector := &Collector{start: testStart}
	runner := &Runner{cfg: cfg, client: client, corpus: corpus, calib: calib, collector: collector, maxContext: maxContext}
	if scraper != nil {
		scraper.Start(ctx)
	}

	var stages []StageSummary
	stopped := ""
	for i, conc := range levels {
		fmt.Printf("▶ Ступень %d/%d: %d одновременных диалогов\n", i+1, len(levels), conc)
		if scraper != nil {
			scraper.ScrapeNow(ctx)
		}
		st := runner.RunStage(ctx, i+1, len(levels), conc)
		var srv *ServerStats
		if scraper != nil {
			scraper.ScrapeNow(context.Background())
			srv = scraper.Window(st.Start.Add(cfg.Warmup), st.End)
		}
		sum := Summarize(st, collector.Stage(i+1), cfg, srv)
		stages = append(stages, sum)
		PrintStageLine(sum)

		if ctx.Err() != nil {
			stopped = "прервано пользователем"
			break
		}
		if sum.Requests > 0 && sum.ErrorRatePct/100 > cfg.MaxErrorRate {
			stopped = fmt.Sprintf("доля ошибок %.1f%% > %.0f%% на %d диалогах", sum.ErrorRatePct, cfg.MaxErrorRate*100, conc)
			break
		}
		if cfg.StopOnSLO && sum.TTFT.P95 > ms(cfg.SLOTTFT) {
			stopped = fmt.Sprintf("p95 TTFT %.0f мс > SLO %v на %d диалогах", sum.TTFT.P95, cfg.SLOTTFT, conc)
			break
		}
		if i < len(levels)-1 && !sleepCtx(ctx, cfg.Cooldown) {
			stopped = "прервано пользователем"
			break
		}
	}
	if scraper != nil {
		scraper.Stop()
	}

	summary := &Summary{
		Meta: Meta{
			Model: cfg.Model, URL: client.chatURL, StartedAt: testStart, FinishedAt: time.Now(),
			MaxModelLen: maxModelLen, MaxContext: maxContext,
			CharsPerToken: calib.CharsPerToken(), CPTSource: calib.Source(),
			MetricsURL: metricsURL, Corpus: corpusName, StoppedReason: stopped,
		},
		Config:   configMap(cfg),
		Stages:   stages,
		Analysis: Analyze(stages, cfg),
	}
	if scraper != nil {
		summary.Timeline = scraper.Timeline(testStart)
	}
	PrintReport(os.Stdout, summary)

	dir := filepath.Join(cfg.OutDir, testStart.Format("20060102-150405"))
	if err := WriteFiles(dir, summary, collector.All()); err != nil {
		fmt.Fprintln(os.Stderr, "ошибка записи отчётов:", err)
		os.Exit(1)
	}
	fmt.Printf("Отчёты: %s/{report.html, summary.json, stages.csv, requests.jsonl}\n", dir)
}

func printSample(cfg *Config) {
	corpus, err := LoadCorpus(cfg.CorpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ошибка корпуса:", err)
		os.Exit(1)
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	cur := corpus.NewCursor(rng)
	fmt.Printf("[system]\n%s\n\n[user, ход 1, ~%d ток.]\n%s\n\n[user, ход 2, ~%d ток.]\n%s\n",
		cfg.SystemPrompt,
		cfg.InitialTokens, BuildUserMessage(cur, rng, true, cfg.InitialTokens, cfg.CharsPerToken),
		cfg.AddTokens, BuildUserMessage(cur, rng, false, cfg.AddTokens, cfg.CharsPerToken))
}
