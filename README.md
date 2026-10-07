# chatbot-llm-bench (llm-loadtest)

Нагрузочное тестирование LLM чат-бота (OpenAI-совместимый API, vLLM) многоходовыми диалогами
на естественном русском языке. Один бинарник, без внешних зависимостей.

## Сборка

```bash
go build -o llm-loadtest .
# под Linux-сервер:
GOOS=linux GOARCH=amd64 go build -o llm-loadtest-linux .
```

## Запуск

```bash
./llm-loadtest --url http://vllm-host:8000 --api-key "$KEY" \
  --start 1 --end 64 --step 0 \
  --initial-tokens 800 --add-tokens 300 --think-time 5s \
  --stage-duration 3m --max-turns 12 --max-tokens 400
```

Основные параметры:

| Флаг | Смысл |
|---|---|
| `--url` | точка входа: `http://host:8000`, `.../v1` или `.../v1/chat/completions` |
| `--api-key` | секретный ключ (или env `LLM_API_KEY`) |
| `--start` / `--end` | с какого / каким количеством одновременных диалогов начинаем / заканчиваем |
| `--step` | шаг между ступенями; `0` — удвоение (1, 2, 4, 8 … end) |
| `--initial-tokens` | размер стартового сообщения, токенов |
| `--add-tokens` | сколько токенов добавляет каждое следующее сообщение пользователя |
| `--think-time` | пауза между ответом модели и следующим сообщением (`--think-jitter` — разброс) |
| `--stage-duration` | режим ступеней «по времени» |
| `--stage-dialogs N` | режим «по количеству»: ступень заканчивается, когда отработано N диалогов |
| `--max-turns` | ходов в одном диалоге; потом диалог начинается заново |
| `--max-context` | лимит контекста; по умолчанию `max_model_len` модели |
| `--max-tokens`, `--temperature`, `--ignore-eos` | параметры генерации |
| `--extra-body` | доп. поля запроса, напр. `'{"chat_template_kwargs":{"enable_thinking":false}}'` |
| `--slo-ttft`, `--slo-tpot` | SLO для goodput и выводов |
| `--max-error-rate`, `--stop-on-slo` | досрочная остановка |
| `--metrics-url` | Prometheus vLLM: `auto` / URL / `none` |
| `--print-sample` | показать пример генерируемых сообщений и выйти |

Полный список: `./llm-loadtest -h`.

## Как устроен диалог

* Каждый диалог стартует со случайного места встроенного корпуса русской прозы (Чехов, Толстой,
  Пушкин, Достоевский — общественное достояние) и «читает книгу» дальше от хода к ходу.
* Ход 1: system + сообщение ≈ `initial-tokens`. Каждый следующий ход: вся история +
  **реальный ответ модели** + новое сообщение ≈ `add-tokens`.
* Размер в токенах считается через vLLM `/tokenize` (точно), иначе — по `usage` ответов.
* Ошибка запроса → пауза и новый диалог. Переполнение контекста → новый диалог.
* Свой корпус: `--corpus file.txt` (UTF-8), например выгрузка реальных чатов.

## Что в отчёте

Консоль и `results/<дата>/`:

* `report.html` — графики и таблицы;
* `summary.json` — все агрегаты;
* `stages.csv` — по ступеням;
* `requests.jsonl` — каждый запрос.

По каждой ступени:

* **Пропускная способность:** RPS, входные и выходные tok/s, скорость генерации одного потока, goodput (доля запросов в SLO).
* **Задержки:** TTFT, TPOT, ITL, E2E — p50/p90/p95/p99/max.
* **Контекст:** размер prompt, доля обрезанных по `max_tokens`, токенов на SSE-чанк, cached tokens.
* **Рост контекста:** матрицы «ступень × номер хода» для prompt, TTFT и TPOT.
* **vLLM `/metrics`:** running/waiting, KV-cache, preemptions, prefix cache hit rate,
  **speculative decoding acceptance rate и средняя длина принятия**, серверные tok/s.
* **Ошибки:** разбивка по типам (`timeout`, `http_429`, `http_5xx`, `context_overflow`, `stream_broken`, …).
* **Выводы:** максимум throughput, предел по SLO, точка насыщения, деградация от контекста, KV/очередь.

Если vLLM стоит за балансировщиком с несколькими репликами, `/metrics` покажет только одну
из них — укажите `--metrics-url` конкретной реплики или `none`.

## Пробный прогон без GPU

```bash
go run ./mockserver -addr 127.0.0.1:18731 -max-seqs 6 &
./llm-loadtest --url http://127.0.0.1:18731 --api-key secret --end 8 --stage-duration 20s --think-time 1s
```
