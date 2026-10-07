package main

import (
	_ "embed"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// Встроенный корпус: русская классическая проза (общественное достояние, ru.wikisource.org).
//
//go:embed corpus/ru_classics.txt
var embeddedCorpus string

type Corpus struct {
	text []rune
}

func LoadCorpus(path string) (*Corpus, error) {
	src := embeddedCorpus
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		if !utf8.Valid(b) {
			return nil, fmt.Errorf("%s: файл должен быть в UTF-8", path)
		}
		src = string(b)
	}
	var paras []string
	for _, line := range strings.Split(src, "\n") {
		line = strings.Join(strings.Fields(line), " ")
		if line != "" {
			paras = append(paras, line)
		}
	}
	text := []rune(strings.Join(paras, "\n"))
	if len(text) < 5000 {
		return nil, fmt.Errorf("корпус слишком мал (%d символов), нужно хотя бы 5000", len(text))
	}
	return &Corpus{text: text}, nil
}

func (c *Corpus) Len() int { return len(c.text) }

// Sample возвращает связный кусок текста (для калибровки токенизатора).
func (c *Corpus) Sample(rng *rand.Rand, n int) string {
	return c.NewCursor(rng).Next(n)
}

// Cursor последовательно читает корпус со случайной позиции —
// диалог «читает книгу» дальше от хода к ходу, текст остаётся связным.
type Cursor struct {
	c   *Corpus
	pos int
}

func (c *Corpus) NewCursor(rng *rand.Rand) *Cursor {
	cu := &Cursor{c: c, pos: rng.Intn(len(c.text))}
	cu.alignToSentence()
	return cu
}

func (cu *Cursor) at(i int) rune { return cu.c.text[i%len(cu.c.text)] }

func isTerminator(r rune) bool { return r == '.' || r == '!' || r == '?' || r == '…' }

// alignToSentence сдвигает курсор на начало следующего предложения или абзаца.
func (cu *Cursor) alignToSentence() {
	L := len(cu.c.text)
	for i := 0; i < 2000; i++ {
		r := cu.at(cu.pos + i)
		if r == '\n' || (isTerminator(r) && unicode.IsSpace(cu.at(cu.pos+i+1))) {
			cu.pos = (cu.pos + i + 1) % L
			break
		}
	}
	cu.skipSpace()
}

func (cu *Cursor) skipSpace() {
	for i := 0; i < 100 && unicode.IsSpace(cu.at(cu.pos)); i++ {
		cu.pos = (cu.pos + 1) % len(cu.c.text)
	}
}

// Next возвращает примерно n символов: дочитывает до конца предложения,
// если оно близко (≤10% от n), иначе обрывает по границе слова.
func (cu *Cursor) Next(n int) string {
	if n < 1 {
		n = 1
	}
	var b strings.Builder
	b.Grow(n * 2)
	i := 0
	for ; i < n; i++ {
		b.WriteRune(cu.at(cu.pos + i))
	}
	limit := n/10 + 20
	clean := false
	for extra := 0; extra < limit+60; extra++ {
		prev, next := cu.at(cu.pos+i-1), cu.at(cu.pos+i)
		if next == '\n' || (isTerminator(prev) && unicode.IsSpace(next)) {
			clean = true
			break
		}
		if extra >= limit && unicode.IsSpace(next) {
			break
		}
		b.WriteRune(next)
		i++
	}
	cu.pos = (cu.pos + i) % len(cu.c.text)
	out := strings.TrimSpace(b.String())
	if clean {
		cu.skipSpace()
	} else {
		// предложение оборвано: помечаем и следующий фрагмент начинаем с нового предложения
		out = strings.TrimRight(out, ",;:—- ") + "…"
		cu.alignToSentence()
	}
	return out
}

// --- шаблоны реплик пользователя ---

var firstTemplates = []string{
	"Привет! Я читаю книгу и хочу обсудить с тобой отрывок. Вот он:\n\n%s\n\nО чём этот фрагмент? Расскажи своими словами.",
	"Помоги мне разобраться в тексте. Прочитай, пожалуйста:\n\n%s\n\nКакие чувства испытывают герои и почему?",
	"Вот отрывок из рассказа:\n\n%s\n\nОпиши главного героя: какой у него характер, что им движет?",
	"Я готовлюсь к уроку литературы. Вот фрагмент:\n\n%s\n\nВыдели основную мысль и приведи пару примеров из текста.",
	"Добрый день! Мне задали прочитать вот этот текст:\n\n%s\n\nОбъясни простыми словами, что здесь происходит.",
}

var nextTemplates = []string{
	"Спасибо! Читаем дальше:\n\n%s\n\nКак изменилось настроение по сравнению с предыдущим отрывком?",
	"Интересно. Вот продолжение:\n\n%s\n\nЧто здесь самое важное для сюжета?",
	"Хорошо, продолжим:\n\n%s\n\nСогласен ли ты с поступками героев? Объясни.",
	"А вот что было дальше:\n\n%s\n\nКратко перескажи и предположи, что случится потом.",
	"Следующий фрагмент:\n\n%s\n\nКакие художественные детали тебе запомнились?",
	"Понятно. Теперь вот этот кусок:\n\n%s\n\nСвяжи его с тем, что мы уже обсудили.",
}

// BuildUserMessage собирает реплику пользователя ≈ targetTokens токенов.
func BuildUserMessage(cu *Cursor, rng *rand.Rand, first bool, targetTokens int, cpt float64) string {
	tmpls := nextTemplates
	if first {
		tmpls = firstTemplates
	}
	t := tmpls[rng.Intn(len(tmpls))]
	overhead := float64(utf8.RuneCountInString(t)-2) / cpt
	fragTokens := float64(targetTokens) - overhead
	if fragTokens < 16 {
		fragTokens = 16
	}
	return fmt.Sprintf(t, cu.Next(int(fragTokens*cpt)))
}

// --- калибровка «символов на токен» ---

type Calibrator struct {
	mu     sync.Mutex
	cpt    float64
	fixed  bool
	source string
}

func NewCalibrator(initial float64) *Calibrator {
	return &Calibrator{cpt: initial, source: "по умолчанию"}
}

func (c *Calibrator) CharsPerToken() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cpt
}

func (c *Calibrator) Source() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.source
}

func (c *Calibrator) Fix(cpt float64, source string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cpt, c.fixed, c.source = cpt, true, source
}

// Observe уточняет коэффициент по usage.prompt_tokens (если /tokenize недоступен).
func (c *Calibrator) Observe(chars, promptTokens, nMsgs int) {
	adj := promptTokens - 5*nMsgs - 5 // грубая поправка на chat template
	if adj < 50 || chars <= 0 {
		return
	}
	ratio := float64(chars) / float64(adj)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.fixed {
		return
	}
	c.cpt = 0.9*c.cpt + 0.1*ratio
	c.source = "по usage ответов"
}
