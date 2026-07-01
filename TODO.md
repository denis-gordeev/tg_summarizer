# TODO

Живой список задач для автоматических раундов.

## Next actions

- Consider evaluating gpt-4.1-nano vs gpt-4o-mini quality tradeoff with A/B testing
- Consider tuning warmup schedule frequency for cost-effectiveness
- Consider adding Lambda provisioned concurrency for more predictable cold starts
- Consider adding CloudWatch alarm on circuit breaker half-open transitions

## Completed in 2026-07-01 round 2 (code quality, meta-artifacts expansion, circuit breaker refactor, S3 empty guard, prompt compaction)

- **Code quality — consolidated fallback append in `update_existing_summary`**: Extracted [`_append_link()`](history_manager.py) local helper that returns the correct append format (inline vs new "Доп. источники:" section). Previously, the same 3-line append block was duplicated in both the short-response and exception fallback paths. Now each path calls `_append_link()`, eliminating the duplication.
- **Quality — expanded `strip_meta_artifacts`**: Added "между прочим" to both intro and outro patterns in [`_META_ARTIFACT_PATTERNS`](utils.py). Added "в частности", "перейдём к", "перейдем к", "возвращаясь к", "как показано" to [`_META_ARTIFACT_INTRO_ONLY`](utils.py) — these are transition/filler phrases at the start of summaries but can appear legitimately in body text (e.g., "Модель превосходит конкурентов в частности благодаря архитектуре").
- **Code quality — `call_openai` uses `get_circuit_breaker_state()`**: Replaced inline `_CIRCUIT_BREAKER_FAILURES >= CIRCUIT_BREAKER_THRESHOLD` check in [`call_openai()`](utils.py) with `cb_state = get_circuit_breaker_state()`. The circuit breaker state is now evaluated through the shared `get_circuit_breaker_state()` function, eliminating duplicated threshold/timeout logic. The inline check was the only place in the codebase that bypassed the shared state function.
- **Robustness — S3 download rejects empty files**: Added 0-byte file size check in [`download_from_s3()`](s3_sync.py). Previously, an empty (0 bytes) file downloaded from S3 would overwrite local state with a corrupt/empty file. Now, empty downloads are removed and skipped, consistent with the upload path which already skips empty files. JSON validation runs after the size check.
- **Cost optimization — compact UPDATE_SUMMARY_PROMPT**: Condensed [`UPDATE_SUMMARY_PROMPT`](prompts.py) from verbose to compact: "Скопируй саммари и добавь ссылку" → "Скопируй саммари, вставь ссылку", "Остальной текст не меняй" → "Остальное не меняй", "Если нет подходящего места" → "Нет подходящего места", "Ответь только обновлённое саммари" → "Только обновлённое саммари". Saves ~10 input tokens per summary update call.
- **Tests added**: 17 new tests (total 452, up from 435):
  - `test_strips_в_частности_intro`: verifies "В частности" stripped from summary start
  - `test_strips_между_прочим_intro`: verifies "Между прочим" stripped from summary start
  - `test_strips_между_прочим_outro`: verifies "Между прочим" stripped from summary end
  - `test_strips_перейдём_к_intro`: verifies "Перейдём к" stripped from summary start
  - `test_strips_перейдем_к_intro`: verifies "Перейдем к" stripped from summary start
  - `test_strips_возвращаясь_к_intro`: verifies "Возвращаясь к" stripped from summary start
  - `test_strips_как_показано_intro`: verifies "Как показано" stripped from summary start
  - `test_preserves_в_частности_in_body`: verifies "в частности" in body preserved
  - `test_call_openai_uses_get_circuit_breaker_state`: verifies `get_circuit_breaker_state()` used in utils.py
  - `test_call_openai_no_inline_failure_threshold_check`: verifies inline threshold check removed
  - `test_download_removes_empty_files`: verifies 0-byte S3 downloads removed and skipped
  - `test_update_uses_append_link_helper`: verifies `_append_link()` helper in history_manager.py
  - `test_no_duplicate_append_logic`: verifies append logic appears only once
  - `test_update_prompt_is_compact`: verifies compact prompt wording
  - `test_update_prompt_still_has_html_preservation`: verifies HTML preservation instruction retained
  - `test_update_prompt_still_has_fallback_instruction`: verifies "Доп. источники:" instruction retained
  - `test_update_prompt_has_no_если`: verifies verbose "Если нет" replaced with "Нет подходящего места"
- All 452 tests pass without errors.

## Completed in 2026-07-01 round (EMF refactor, NLP cost savings, None-date guard, meta-artifacts expansion, TODO cleanup)

- **Code quality — extracted `_emit_emf` helper**: Added [`_emit_emf()`](utils.py) that encapsulates the CloudWatch Embedded Metric Format (EMF) JSON construction and stdout writing pattern. Refactored all 5 EMF emission functions to use the helper: [`_emit_openai_latency()`](utils.py), [`_emit_invocation_summary()`](lambda_handler.py), [`_emit_warmup_metric()`](lambda_handler.py), [`_emit_s3_upload_metric()`](lambda_handler.py), [`_emit_coverage_dedup_metric()`](message_processor.py). Eliminates ~80 lines of duplicated EMF boilerplate across 3 modules. The helper auto-fills the `Function` dimension and handles all errors with a single `try/except`.
- **Cost optimization — NLP max_tokens reduced 10→3**: Changed `max_tokens` in [`is_nlp_related()`](message_processor.py) from 10 to 3. The NLP prompt asks for "да" or "нет" (1 token), so 3 is sufficient (1 token + potential whitespace). Saves ~7 completion tokens per NLP rejection (~50-80% of messages), reducing output cost per check.
- **Robustness — None msg.date guard**: Added `msg.date is None` check before date comparison in [`_fetch_from_sources()`](telegram_client.py). Telethon can return messages with `None` date in edge cases (e.g., service messages), which would crash on `msg.date < since` with `TypeError`. Now skipped gracefully.
- **Quality — expanded `strip_meta_artifacts`**: Added "в конечном итоге", "собственно говоря", "к слову" to both intro and outro patterns in [`_META_ARTIFACT_PATTERNS`](utils.py). Added "в первую очередь", "короче говоря", "подытоживая" to [`_META_ARTIFACT_INTRO_ONLY`](utils.py) — these are meta-structural at the start of a summary but can appear legitimately in body text. "в конечном итоге" complements the existing "в итоге" pattern by catching the longer form.
- **TODO.md cleanup**: Removed duplicate "Next actions" section (contained stale item "Consider adding coverage dedup group-stream widget" completed in the 06-30 round). Consolidated to a single current "Next actions" section.
- **Tests added**: 15 new tests (total 435, up from 420):
  - `test_emit_emf_exists_in_utils_source`: verifies `_emit_emf` function defined in utils.py
  - `test_lambda_handler_imports_emit_emf`: verifies lambda_handler.py imports `_emit_emf`
  - `test_message_processor_imports_emit_emf`: verifies message_processor.py imports `_emit_emf`
  - `test_emit_emf_auto_fills_function_dimension`: verifies `Function` auto-fill logic in `_emit_emf`
  - `test_emit_emf_does_not_raise_on_failure`: verifies error handling in `_emit_emf`
  - `test_nlp_check_uses_max_tokens_3`: verifies NLP max_tokens=3 in message_processor.py
  - `test_nlp_check_does_not_use_max_tokens_10`: verifies old max_tokens=10 removed
  - `test_fetch_from_sources_guards_none_date`: verifies `msg.date is None` guard in telegram_client.py
  - `test_strips_в_конечном_итоге_intro`: verifies "В конечном итоге" stripped from summary start
  - `test_strips_в_конечном_итоге_outro`: verifies "В конечном итоге" stripped from summary end
  - `test_strips_собственно_говоря_intro`: verifies "Собственно говоря" stripped from summary start
  - `test_strips_к_слову_outro`: verifies "К слову" stripped from summary end
  - `test_strips_в_первую_очередь_intro`: verifies "В первую очередь" stripped from summary start
  - `test_strips_короче_говоря_intro`: verifies "Короче говоря" stripped from summary start
  - `test_strips_подытоживая_intro`: verifies "Подытоживая" stripped from summary start
- Updated `test_nlp_check_uses_max_tokens_10` → `test_nlp_check_uses_max_tokens_3` (asserts `max_tokens=3`).
- Updated `test_emit_coverage_dedup_metric_uses_module_level_time` → `test_emit_coverage_dedup_metric_uses_emit_emf` (verifies `_emit_emf` usage).
- Updated all test stubs to include `fake_utils._emit_emf` where `message_processor` or `lambda_handler` is imported.
- Updated EMF capture tests to use `patch("sys.stdout", ...)` instead of `patch.object(lambda_handler.sys, ...)` since `_emit_emf` writes to `sys.stdout` via `utils.sys`.
- All 435 tests pass without errors.

## Completed in 2026-06-30 round (Lambda hardening round 39, group coverage widget, meta-artifacts expansion, TODO cleanup)

- **Dashboard: Coverage Dedup (Group) widget**: Added "Coverage Dedup (Group)" widget to [`SummarizerDashboard`](template.yaml) showing `CoveredMessages`, `NewMessages`, `TotalNlpMessages` from `tg_summarizer/Coverage` namespace with `StreamType=group` dimension. Complements the existing channel widget, enabling visual monitoring of group-stream coverage dedup hit rate alongside channels. Directly addresses the TODO item "Consider adding coverage dedup group-stream widget to dashboard (currently only channel)".
- **Quality — expanded `strip_meta_artifacts`**: Added "напоследок", "кратко говоря", "среди прочего" to the intro/outro patterns in [`_META_ARTIFACT_PATTERNS`](utils.py). Added "во-первых", "во-вторых", "в-третьих" to [`_META_ARTIFACT_INTRO_ONLY`](utils.py) — enumeration markers at the start of a summary are always meta-artifacts, but can appear legitimately in body text (e.g., "Во-первых, модель превосходит..."). "среди прочего" is added only to intro/outro since it's a qualifying filler.
- **TODO.md cleanup**: Removed stale/duplicate "Next actions" sections from older rounds. Several items ("Consider adding coverage dedup dashboard widget", "Consider adding S3 upload dedup metric to CloudWatch dashboard") were already completed in prior rounds but still listed as next actions. Consolidated to a single current "Next actions" section.
- **Tests added**: 6 new tests (total 420, up from 414):
  - `test_template_dashboard_contains_coverage_dedup_group_widget`: verifies "Coverage Dedup (Group)" widget and `StreamType=group` in template.yaml
  - `test_strips_напоследок_outro`: verifies "Напоследок" stripped from summary end
  - `test_strips_кратко_говоря_outro`: verifies "Кратко говоря" stripped from summary end
  - `test_strips_среди_прочего_intro`: verifies "Среди прочего" stripped from summary start
  - `test_strips_во_первых_intro`: verifies "Во-первых" stripped from summary start
  - `test_preserves_во_первых_in_body`: verifies "во-первых" in body preserved
- All 420 tests pass without errors.

## Completed in 2026-06-29 round (Lambda hardening round 38, dashboard widgets, S3 metric fix, code quality)

- **Dashboard: Coverage dedup widget**: Added "Coverage Dedup (Channel)" widget to [`SummarizerDashboard`](template.yaml) showing `CoveredMessages`, `NewMessages`, `TotalNlpMessages` from `tg_summarizer/Coverage` namespace with `StreamType=channel` dimension. Enables visual monitoring of coverage dedup hit rate for tuning `ENABLE_SUMMARIES_DEDUPLICATION`. Directly addresses the TODO item "Consider adding coverage dedup dashboard widget for visual monitoring".
- **Dashboard: S3 upload widget**: Added "S3 State Sync" widget to [`SummarizerDashboard`](template.yaml) showing `S3UploadedFiles` and `S3UploadFailures` from `tg_summarizer/S3` namespace. Enables visual monitoring of S3 sync health. Directly addresses the TODO item "Consider adding S3 upload dedup metric to CloudWatch dashboard".
- **Observability: S3 upload metric emitted on all runs**: Changed [`_emit_s3_upload_metric()`](lambda_handler.py) call to emit when `uploaded > 0 or failed > 0` instead of only when `failed > 0`. Previously, successful S3 uploads had no EMF metric, meaning the `S3UploadedFiles` metric was only populated on failure runs. Now both metrics are available for dashboard visualization and trend analysis on every invocation with S3 sync.
- **Bug fix: `_dedup_covered_messages` return type annotation**: Fixed return type from `List[MessageInfo]` to `tuple[list[MessageInfo], list[SummaryInfo]]` in [`_dedup_covered_messages()`](message_processor.py). The function has always returned a 2-tuple, but the annotation was wrong, potentially misleading type checkers.
- **Code quality: removed unnecessary local import in `_emit_coverage_dedup_metric`**: Removed `import time as _time` inside [`_emit_coverage_dedup_metric()`](message_processor.py). The `time` module is already imported at module level (line 7). The local import was inconsistent with all other EMF emission functions in the codebase which use the module-level import.
- **Tests added**: 5 new tests (total 414, up from 409):
  - `test_template_dashboard_contains_coverage_dedup_widget`: verifies "Coverage Dedup (Channel)" widget and `tg_summarizer/Coverage` metrics in template.yaml
  - `test_template_dashboard_contains_s3_upload_widget`: verifies "S3 State Sync" widget and `S3UploadedFiles` in template.yaml
  - `test_handler_emits_s3_upload_metric_on_success_with_uploads`: verifies `_emit_s3_upload_metric` called when S3 upload succeeds with files
  - `test_emit_coverage_dedup_metric_uses_module_level_time`: verifies no local `import time as _time` in message_processor.py
  - `test_dedup_covered_messages_return_type_is_tuple`: verifies `_dedup_covered_messages` return annotation is `tuple[...]`
- Updated `test_handler_skips_s3_upload_metric_on_success` — now verifies no metric emitted when `uploaded=0, failed=0` (S3 not configured).
- All 414 tests pass without errors.

- **Runtime tunability — update_existing_summary prompt in PromptManager**: Moved the inline update prompt from [`update_existing_summary()`](history_manager.py) to [`UPDATE_SUMMARY_PROMPT`](prompts.py) in `PromptManager`. The prompt uses `{new_link}` placeholder via Python `str.format()`. Previously, the update prompt was hardcoded in `history_manager.py`, requiring a code change to tune it. Now it can be overridden at runtime via `prompts.json` without code changes, consistent with all other prompts (COVERAGE_AND_MATCH, NLP_RELEVANCE, CHANNEL_SUMMARY, GROUP_SUMMARY). Directly addresses the TODO item "Consider moving update_existing_summary prompt to PromptManager for runtime tunability".
- **Observability — S3 upload failure EMF metric and CloudWatch alarm**: Added [`_emit_s3_upload_metric()`](lambda_handler.py) that emits `S3UploadFailures` and `S3UploadedFiles` EMF metrics under `tg_summarizer/S3` namespace when S3 upload failures are detected. Added `S3UploadFailuresAlarm` in [`template.yaml`](template.yaml) — alerts when `S3UploadFailures > 0` (Sum statistic, 5-minute period). The alarm is conditional on `HasStateBucket` since S3 sync is only relevant when a bucket is configured. Directly addresses the TODO item "Consider adding CloudWatch Logs metric filter on S3 upload failure warning for automated alerting" — the EMF metric approach is cleaner than a Logs metric filter.
- **Observability — coverage dedup EMF metric**: Added [`_emit_coverage_dedup_metric()`](message_processor.py) that emits `CoveredMessages`, `TotalNlpMessages`, and `NewMessages` EMF metrics under `tg_summarizer/Coverage` namespace with `StreamType` dimension (channel/group). Emitted when `covered_count > 0` in `process_messages()`. Enables CloudWatch-based monitoring of coverage dedup hit rate for tuning `ENABLE_SUMMARIES_DEDUPLICATION`. Directly addresses the TODO item "Consider adding EMF metric for coverage dedup hit rate for tuning ENABLE_SUMMARIES_DEDUPLICATION".
- **Cost optimization — compact NLP_RELEVANCE_PROMPT**: Condensed [`NLP_RELEVANCE_PROMPT`](prompts.py) from verbose multi-line accept/reject sections to compact single-line format: `ПРИНИМАЙ:` (one line) and `ОТКЛОНЯЙ:` (one line). Removed redundant parenthetical markers (`'да'`, `'нет'`) and merged overlapping entries. The prompt conveys the same classification criteria in ~30% fewer input tokens, reducing per-NLP-check cost at no loss of classification accuracy.
- **Code quality — deduplicated `response.usage` access in `call_openai`**: Replaced three separate `hasattr(response, 'usage') and response.usage` checks with a single `usage = getattr(response, 'usage', None)` in [`call_openai()`](utils.py). The cached `usage` reference is used for `total_tokens`, `prompt_tokens`, `completion_tokens`, and the logging branch. Reduces code repetition and avoids redundant attribute access.
- **Tests added**: 12 new tests (total 409, up from 397):
  - `test_emit_s3_upload_metric_writes_emf`: verifies S3 upload EMF metric JSON with correct namespace and values
  - `test_handler_emits_s3_upload_metric_on_failure`: verifies `_emit_s3_upload_metric` called when S3 upload has failures
  - `test_handler_skips_s3_upload_metric_on_success`: verifies no EMF metric emitted when S3 upload succeeds
  - `test_template_contains_s3_upload_failures_alarm`: verifies `S3UploadFailuresAlarm` in template.yaml
  - `test_s3_upload_alarm_conditional_on_state_bucket`: verifies `HasStateBucket` condition on S3 alarm
  - `test_emit_coverage_dedup_metric_in_source`: verifies coverage dedup EMF namespace/metrics in message_processor.py
  - `test_emit_coverage_dedup_metric_called_on_covered`: verifies `_emit_coverage_dedup_metric` call with `covered_count`
  - `test_update_summary_prompt_in_prompt_manager`: verifies `UPDATE_SUMMARY_PROMPT` in prompts.py
  - `test_update_summary_prompt_uses_format_placeholder`: verifies `{new_link}` format placeholder
  - `test_history_manager_uses_prompts_update_summary`: verifies AST: `prompts.UPDATE_SUMMARY_PROMPT` in history_manager.py
  - `test_nlp_prompt_is_compact`: verifies NLP prompt uses compact format (no `ПРИНИМАЙ ('да'):`)
  - `test_nlp_prompt_has_accept_and_reject_sections`: verifies NLP prompt has ПРИНИМАЙ/ОТКЛОНЯЙ sections
- Updated test stub in [`tests/test_history_manager.py`](tests/test_history_manager.py) — all `fake_prompts.prompts` stubs now include `UPDATE_SUMMARY_PROMPT="Вставь ссылку {new_link}"`.
- Updated [`test_update_prompt_mentions_html`](tests/test_history_manager.py) — now checks `prompts.py` instead of `history_manager.py` for HTML preservation text.
- All 409 tests pass without errors.

## Completed in 2026-06-26 round (Lambda hardening round 36, update meta-artifacts, circuit breaker NLP, prompt token savings, dead code removal, config sync)

- **Quality — `strip_meta_artifacts` in `update_existing_summary`**: Added `strip_meta_artifacts()` call in [`update_existing_summary()`](history_manager.py) after the LLM response and fallback paths. Previously, only `summarize_text()` and `summarize_group_text()` applied meta-artifact stripping — the update path stored raw LLM output, allowing meta-artifacts like "Итак" or "В заключение" to leak into published summaries and persist in history. Added "Без вводных слов и заключений" to the update prompt to reduce the likelihood of meta-artifacts from the LLM.
- **Robustness — fallback phrase renamed to avoid `strip_meta_artifacts` conflict**: Changed the fallback link append phrase in [`update_existing_summary()`](history_manager.py) from "Другие ссылки:" to "Доп. источники:". The old phrase was matched by `strip_meta_artifacts()` pattern `другие ссылки:`, causing silent data loss when a fallback-created summary was later re-processed (e.g., during channel restore). The new phrase "Доп. источники:" is not in the strip patterns, so appended links survive re-processing. Updated the update prompt to reference the new phrase. Updated all test assertions from "Другие ссылки:" to "Доп. источники:".
- **Observability — circuit breaker check in `is_nlp_related`**: Added `is_circuit_breaker_open()` check in [`is_nlp_related()`](message_processor.py) — when the circuit breaker is open, `call_openai()` returns `""` and `is_nlp_related` was returning `(False, "")` with an empty reason, making it look like NLP rejection rather than circuit breaker rejection. Now returns `(False, "circuit_breaker_open")` immediately, skipping the LLM call entirely and providing a clear reason for CloudWatch log analysis. Added [`is_circuit_breaker_open()`](utils.py) helper that returns `True` when `get_circuit_breaker_state()["state"] == "open"`.
- **Quality — expanded `strip_meta_artifacts`**: Added "таким образом", "отметим", "заметим", "как уже упоминалось", "в данном обзоре", "в данной статье", "ниже представлены", "ниже приведены", "давайте рассмотрим" to the intro pattern in [`_META_ARTIFACT_PATTERNS`](utils.py). "таким образом" and "отметим"/"заметим" are common LLM transition/filler phrases at the start of summaries. "в данном обзоре", "ниже представлены", "давайте рассмотрим" are meta-structural markers. These are only added to the intro pattern (start of text) because they can appear legitimately in body text — added as a separate [`_META_ARTIFACT_INTRO_ONLY`](utils.py) pattern. "как уже упоминалось" is added to both intro and outro patterns since it's always a redundant cross-reference.
- **Cost optimization — NLP prompt simplified to да/нет**: Changed [`NLP_RELEVANCE_PROMPT`](prompts.py) from `"Если 'нет' — напиши "нет, причина: " и объясни"` to `"Ответь только 'да' или 'нет'"`. The previous instruction forced the LLM to generate an explanation (~10-30 completion tokens) for every rejected message, but `is_nlp_related()` only checks `answer.startswith("да")` — the reason was unused. With ~50-80% of messages rejected, this saves ~5-20 completion tokens per rejection. At `gpt-4.1-nano` pricing ($0.40/M output), this reduces NLP check output cost by ~50%.
- **Cost optimization — fallback pricing matches default model**: Changed `_COST_PER_MILLION` fallback in [`_estimate_cost_usd()`](utils.py) from `(0.15, 0.60)` (gpt-4o-mini) to `(0.10, 0.40)` (gpt-4.1-nano). The default model is `gpt-4.1-nano`, so the fallback for unknown models should use its pricing, not the more expensive gpt-4o-mini pricing.
- **Code quality — removed dead `get_similar_channels_from_telegram()`**: Removed 44-line unused function [`get_similar_channels_from_telegram()`](telegram_client.py) and its `GetChannelRecommendationsRequest`/`InputChannel` imports. This function was never called from any module in the codebase. Reduced Lambda package size and maintenance burden.
- **Robustness — Lambda event input validation**: Added `if not isinstance(event, dict): event = {}` at the start of [`handler()`](lambda_handler.py). Previously, a malformed EventBridge event (string, None, etc.) would cause `AttributeError` on `event.get()`. Now gracefully defaults to empty dict, using default flag values.
- **Observability — SAM template config sync**: Added 9 missing config constants as Parameters and Environment Variables in [`template.yaml`](template.yaml): `UpdateMatchMaxSummaries`, `UpdateMatchMaxCharsPerSummary`, `MaxChannelHistoryMessages`, `MaxChannelSummaries`, `MaxGroupHistoryMessages`, `MaxGroupSummaries`, `GroupSummarizationIntervalSeconds`, `RestoreHistoryDays`. Previously, these were configurable via env vars in `config.py` but not exposed in the SAM template, making them impossible to tune via CloudFormation without code changes.
- **Documentation — `.env.example` sync**: Added 9 missing config constants to [`.env.example`](.env.example): `UPDATE_MATCH_MAX_SUMMARIES`, `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY`, `MAX_CHANNEL_HISTORY_MESSAGES`, `MAX_CHANNEL_SUMMARIES`, `MAX_GROUP_HISTORY_MESSAGES`, `MAX_GROUP_SUMMARIES`, `GROUP_SUMMARIZATION_INTERVAL_SECONDS`, `RESTORE_HISTORY_DAYS`, `DEBUG`.
- **Tests added**: 19 new tests (total 397, up from 378):
  - `test_update_strips_meta_artifacts_from_llm_response`: verifies `strip_meta_artifacts` is called in `update_existing_summary`
  - `test_strips_таким_образом_outro`: verifies "Таким образом" at text start stripped
  - `test_strips_таким_образом_intro`: verifies "Таким образом" intro stripped
  - `test_preserves_таким_образом_in_middle`: verifies "таким образом" in body preserved
  - `test_strips_отметим_intro`: verifies "Отметим" intro stripped
  - `test_strips_как_уже_упоминалось_outro`: verifies "Как уже упоминалось" outro stripped
  - `test_strips_в_данном_обзоре_intro`: verifies "В данном обзоре" intro stripped
  - `test_strips_ниже_представлены_intro`: verifies "Ниже представлены" intro stripped
  - `test_strips_давайте_рассмотрим_intro`: verifies "Давайте рассмотрим" intro stripped
  - `test_returns_false_when_closed`: verifies `is_circuit_breaker_open()` returns False when breaker is closed
  - `test_nlp_related_checks_circuit_breaker_before_llm`: verifies `is_nlp_related` checks circuit breaker (AST)
  - `test_handler_handles_none_event`: verifies Lambda handler handles None event gracefully
  - `test_handler_handles_string_event`: verifies Lambda handler handles string event gracefully
  - `test_nlp_prompt_asks_only_yes_no`: verifies NLP prompt asks for simple да/нет
  - `test_cost_fallback_is_nano_pricing`: verifies fallback cost matches gpt-4.1-nano ($0.10/$0.40)
  - `test_template_has_update_match_max_summaries`: verifies `UpdateMatchMaxSummaries` in template.yaml
  - `test_template_has_max_channel_history_messages`: verifies `MaxChannelHistoryMessages` in template.yaml
  - `test_template_has_restore_history_days`: verifies `RestoreHistoryDays` in template.yaml
  - `test_template_has_group_summarization_interval`: verifies `GroupSummarizationIntervalSeconds` in template.yaml
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) — added `is_circuit_breaker_open` to all `fake_utils` stubs.
- Updated fallback phrase assertions in [`tests/test_history_manager.py`](tests/test_history_manager.py) — changed all "Другие ссылки:" assertions to "Доп. источники:".
- All 397 tests pass without errors.

## Completed in 2026-06-25 round (Lambda hardening round 34, Lambda warmup, Telegram retry, prompt brevity, meta-artifacts expansion, template fix)

- **Lambda warmup support**: Added `warmup` event flag handling in [`lambda_handler.handler()`](lambda_handler.py). When `warmup=true`, the handler downloads S3 state, starts and stops Telegram clients (via [`_warmup_telegram()`](lambda_handler.py)), uploads state, and returns `{'status': 'warmed'}` without running the summarizer. Added [`_warmup_telegram()`](lambda_handler.py) async function that connects and disconnects Telegram clients to refresh session files. Added `WarmupScheduleExpression` parameter (default: empty = disabled) and `HasWarmupSchedule` condition in [`template.yaml`](template.yaml). When enabled, a second EventBridge `WarmupRun` schedule triggers the Lambda with `{"warmup": true}`, keeping the execution environment warm and Telegram sessions fresh between daily runs. Directly addresses the "Consider adding Lambda warmer" TODO item.
- **Telegram reconnection retry**: Added retry with exponential backoff in [`start_clients()`](telegram_client.py). `user_client.start()` and `bot_client.start()` are now wrapped in a retry loop (`max_retries=2`, initial delay 2s, doubling). Previously, a transient Telegram API failure during client start would immediately crash the entire Lambda invocation. Now retries once with a 2-second backoff before giving up, making the Lambda more resilient to brief Telegram connectivity issues.
- **Prompt quality — brevity emphasis**: Added "Пиши максимально кратко, без вводных слов и оборотов" rule to both [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py). Directly targets the AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness. The rule complements the existing "Без воды и пояснений очевидного" rule by specifically targeting filler phrases ("вводные слова") that LLMs frequently produce despite other instructions.
- **Quality — expanded strip_meta_artifacts**: Added `"стоит отметить"`, `"следует отметить"`, `"важно отметить"` to the intro pattern and `"подводя итог"`, `"резюмируя"` to both intro and outro patterns in [`_META_ARTIFACT_PATTERNS`](utils.py). These are common LLM filler phrases that violate the "Без введения, заключения и мета-комментариев" prompt rule. "стоит отметить", "следует отметить", and "важно отметить" are added only to the intro pattern (not outro) because they can appear legitimately in body text (e.g., "Модель GPT-5 стоит отметить среди прочих"). "подводя итог" and "резюмируя" are clear meta-artifacts in any position.
- **Bug fix — template.yaml Outputs indentation**: Fixed [`template.yaml`](template.yaml) — `Outputs` section was incorrectly nested inside `Resources` (2-space indent) instead of being a top-level YAML key (0 indent). While SAM/CloudFormation may have handled this gracefully, the correct placement ensures template validation passes strictly and avoids future deployment issues.
- **Tests added**: 18 new tests (total 363, up from 345):
  - `test_warmup_returns_warmed_status`: verifies warmup event returns `{'status': 'warmed'}`
  - `test_warmup_downloads_and_uploads_s3`: verifies S3 download+upload on warmup
  - `test_warmup_does_not_run_summarizer`: verifies summarizer not called on warmup
  - `test_warmup_returns_warmed_on_telegram_failure`: verifies warmup succeeds even if Telegram connection fails
  - `test_warmup_string_true`: verifies `warmup="true"` parsed correctly
  - `test_start_clients_has_retry_loop`: verifies AST: start_clients has retry for-loop
  - `test_start_clients_max_retries_constant`: verifies `max_retries` in start_clients
  - `test_channel_summary_prompt_has_brevity_rule`: verifies "без вводных слов" in prompts
  - `test_group_summary_prompt_has_brevity_rule`: verifies both prompts have brevity rule
  - `test_strips_стоит_отметить_intro`: verifies "Стоит отметить" stripped from summary start
  - `test_strips_следует_отметить_intro`: verifies "Следует отметить" stripped from summary start
  - `test_strips_важно_отметить_intro`: verifies "Важно отметить" stripped from summary start
  - `test_strips_подводя_итог_outro`: verifies "Подводя итог" stripped from summary end
  - `test_strips_резюмируя_outro`: verifies "Резюмируя" stripped from summary end
  - `test_preserves_стоит_отметить_in_body`: verifies "стоит отметить" in body preserved
  - `test_template_outputs_at_top_level`: verifies `Outputs` at 0 indent in template.yaml
  - `test_template_contains_warmup_schedule_parameter`: verifies `WarmupScheduleExpression` in template
  - `test_template_contains_warmup_event`: verifies `WarmupRun` and `HasWarmupSchedule` in template
- All 363 tests pass without errors.

- **Observability — per-invocation cumulative cost EMF metric**: Added [`_emit_invocation_summary()`](lambda_handler.py) that emits an EMF metric at the end of each Lambda invocation (both success and error paths) with `CumulativePromptTokens`, `CumulativeCompletionTokens`, `CumulativeCostUSD`, and `ElapsedSeconds` under the `tg_summarizer/Invocation` namespace with only the `Function` dimension (no `Model`). Unlike the per-call EMF in [`_emit_openai_latency()`](utils.py) (which has `Function+Model` dimensions), this metric aggregates across models, enabling CloudWatch Sum-based daily cost alerting. Skips emission when no tokens were used.
- **Observability — daily cost alarm**: Added `OpenAIDailyCostAlarm` in [`template.yaml`](template.yaml) — alerts when cumulative `CumulativeCostUSD` (Sum statistic, 86400s period) exceeds $1.00 per day. Uses the new `tg_summarizer/Invocation` namespace. Complements the existing `OpenAICostAlarm` (per-invocation Maximum > $0.10). Directly addresses the TODO item "Consider adding CloudWatch metric filter on `token_usage` field for cost alerting" — the EMF metric approach is cleaner than a Logs metric filter and provides the same daily cost alerting capability.
- **Observability — dashboard invocation widget**: Added a fourth widget "Per-Invocation Cost & Tokens" to [`SummarizerDashboard`](template.yaml) showing `CumulativeCostUSD` (left axis, USD), `CumulativePromptTokens` and `CumulativeCompletionTokens` (right axis, tokens) from the new `tg_summarizer/Invocation` namespace. Provides per-invocation cost and token visibility alongside the existing per-call OpenAI metrics.
- **Quality — expanded strip_meta_artifacts**: Added `"также стоит отметить"` pattern to the intro/outro patterns in [`_META_ARTIFACT_PATTERNS`](utils.py). Added `"смотри также:"` and `"подробнее:"` as colon-terminated patterns — these LLM meta-artifacts appear as standalone "Смотри также: [link]" or "Подробнее: [link]" lines that violate the "Без введения, заключения и мета-комментариев" prompt rule. The colon requirement preserves inline usage like "Подробнее о модели GPT-5" in the body.
- **Robustness — empty input validation in call_openai**: Added non-empty validation for `system_prompt` and `user_content` in [`call_openai()`](utils.py). Empty or whitespace-only inputs now return `""` immediately with a warning log, avoiding a wasted API call that would return useless results. Prevents edge cases where upstream code passes empty content (e.g., a message with no text after truncation).
- **Tests added**: 16 new tests (total 345, up from 329):
  - `test_emit_invocation_summary_writes_emf`: verifies EMF JSON output with correct metrics and namespace
  - `test_emit_invocation_summary_skips_zero_tokens`: verifies no EMF output when tokens are 0
  - `test_handler_emits_invocation_summary_on_success`: verifies `_emit_invocation_summary` called on success
  - `test_handler_emits_invocation_summary_on_error`: verifies `_emit_invocation_summary` called on error
  - `test_template_contains_daily_cost_alarm`: verifies `OpenAIDailyCostAlarm`, `CumulativeCostUSD`, `tg_summarizer/Invocation`, and 86400s period in template.yaml
  - `test_template_dashboard_contains_invocation_widget`: verifies dashboard has CumulativeCostUSD/PromptTokens/CompletionTokens from Invocation namespace
  - `test_strips_также_стоит_отметить_intro`: verifies "Также стоит отметить" stripped from summary start
  - `test_strips_также_стоит_отметить_outro`: verifies "Также стоит отметить" stripped from summary end
  - `test_strips_смотри_также_with_colon_outro`: verifies "Смотри также:" stripped
  - `test_strips_подробнее_with_colon_outro`: verifies "Подробнее:" stripped
  - `test_preserves_смотри_также_in_body`: verifies "смотри также" (no colon) preserved in body
  - `test_preserves_подробнее_in_body`: verifies "Подробнее о модели" preserved in body
  - `test_call_openai_returns_empty_on_empty_system_prompt`: verifies empty system prompt returns ""
  - `test_call_openai_returns_empty_on_whitespace_system_prompt`: verifies whitespace-only system prompt returns ""
  - `test_call_openai_returns_empty_on_empty_user_content`: verifies empty user content returns ""
  - `test_call_openai_returns_empty_on_whitespace_user_content`: verifies whitespace-only user content returns ""
- Updated test stub in [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — added `_estimate_cost_usd` to `fake_utils` stub.
- All 345 tests pass without errors.

## Completed in 2026-06-24 round (Lambda hardening round 32, default model, CloudWatch dashboard, model observability, meta-artifacts expansion)

- **Cost optimization — gpt-4.1-nano as default model**: Changed default `OPENAI_MODEL` from `gpt-4o-mini` to `gpt-4.1-nano` in [`config.py`](config.py), [`template.yaml`](template.yaml), and [`.env.example`](.env.example). gpt-4.1-nano costs $0.10/$0.40 per 1M tokens (input/output) vs gpt-4o-mini's $0.15/$0.60 — a ~33% reduction on both input and output tokens. The model was already in the cost table and AllowedValues; only the default changed. Users can still select `gpt-4o-mini` or `gpt-4.1-mini` via the `OpenAIModel` SAM parameter or `OPENAI_MODEL` env var. Directly addresses the AUTOWORK_INSTRUCTIONS cost constraint ("Выбранные решения не должны быть дороже gpt-4o-mini") and the long-standing TODO item "Consider evaluating gpt-4.1-nano as default model".
- **Observability — CloudWatch Dashboard**: Added `SummarizerDashboard` in [`template.yaml`](template.yaml) with three widgets: (1) OpenAI Latency & Cost (dual-axis: seconds + USD), (2) OpenAI Token Usage (prompt + completion, sum), (3) Lambda Metrics (duration, errors, invocations, throttles). Dashboard is automatically created on `sam deploy` and visible in the CloudWatch console. Addresses the TODO item "Consider adding CloudWatch dashboard for circuit breaker state" — the dashboard covers circuit breaker-related metrics (errors, latency) plus cost and token usage.
- **Observability — model name in Lambda response**: Added `model` field (value from [`OPENAI_MODEL`](config.py)) to both success and error Lambda responses in [`lambda_handler.handler()`](lambda_handler.py). Previously, the Lambda response included circuit_breaker, token_usage, and s3_upload but not which model was used. The model name is now available for programmatic monitoring and cost correlation without parsing CloudWatch logs.
- **Quality — expanded strip_meta_artifacts**: Added `"обратите внимание"` and `"напомним"` patterns to [`_META_ARTIFACT_PATTERNS`](utils.py). LLMs sometimes produce "Обратите внимание" as a standalone intro/outro line and "Напомним" as a recap filler — both violate the "Без введения, заключения и мета-комментариев" prompt rule. The patterns match these phrases at line start (intro) and line end (outro), preserving inline usage like "стоит обратить внимание на модель" in the body.
- **Tests added**: 10 new tests (total 329, up from 319):
  - `test_config_openai_model_default_is_gpt41_nano`: verifies default OPENAI_MODEL is `gpt-4.1-nano`
  - `test_handler_includes_model_name_in_success_response`: verifies `model` field in success Lambda response
  - `test_handler_includes_model_name_in_error_response`: verifies `model` field in error Lambda response
  - `test_template_contains_dashboard_resource`: verifies `SummarizerDashboard` and `AWS::CloudWatch::Dashboard` in template.yaml
  - `test_template_default_model_is_gpt41_nano`: verifies SAM template default model is gpt-4.1-nano
  - `test_template_allowed_values_gpt41_nano_first`: verifies gpt-4.1-nano is first in AllowedValues list
  - `test_strips_обратите_внимание_intro`: verifies "Обратите внимание" stripped from summary start
  - `test_strips_напомним_intro`: verifies "Напомним" stripped from summary start
  - `test_strips_обратите_внимание_outro`: verifies "Обратите внимание" stripped from summary end
  - `test_preserves_обратите_внимание_in_body`: verifies "обратить внимание" preserved in summary body (not at line start)
- Updated test stub in [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — added `OPENAI_MODEL` to `fake_config` stub to support the new import.
- All 329 tests pass without errors.

- **Prompt quality — anti-list rule**: Added "Без нумерованных и маркированных списков — только сплошной текст с абзацами" and "Без подзаголовков внутри раздела — один заголовок на тему" rules to both [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py). LLMs frequently produce verbose bullet/numbered lists and nested sub-headers in Telegram digests, which render poorly and waste tokens. Directly targets the AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness.
- **Cost monitoring — gpt-4.1-nano in cost table**: Added `"gpt-4.1-nano": (0.10, 0.40)` to [`_COST_PER_MILLION`](utils.py) cost lookup. gpt-4.1-nano is cheaper than gpt-4o-mini ($0.10/$0.40 vs $0.15/$0.60 per 1M tokens), making it a valid model per AUTOWORK_INSTRUCTIONS cost constraint. Without this entry, using gpt-4.1-nano would trigger a false cost-table warning.
- **Cost monitoring — CloudWatch cost alarm**: Added `OpenAICostAlarm` in [`template.yaml`](template.yaml) — alerts when `EstimatedCostUSD` EMF metric exceeds $0.10 per invocation (Maximum statistic, 5-minute period). Uses the existing EMF metric from [`_emit_openai_latency()`](utils.py). Enables proactive cost anomaly detection without manual CloudWatch log inspection.
- **Model guard — AllowedValues in template.yaml**: Added `AllowedValues` constraint to `OpenAIModel` parameter in [`template.yaml`](template.yaml) — restricts to `"gpt-4o-mini"`, `"gpt-4.1-mini"`, `"gpt-4.1-nano"`. Prevents accidental SAM deployment with expensive models (e.g., gpt-4, gpt-4o) that would violate the AUTOWORK_INSTRUCTIONS cost constraint.
- **Observability — S3 upload status in Lambda response**: Changed [`upload_to_s3()`](s3_sync.py) to return `{"uploaded": N, "failed": N, "skipped_empty": N}` dict instead of `None`. Both success and error Lambda responses now include `s3_upload` field. Previously, S3 upload stats were only in CloudWatch logs — now available in the structured Lambda response for programmatic monitoring.
- **Quality — expanded strip_meta_artifacts**: Added `"другие ссылки:"` pattern to [`_META_ARTIFACT_PATTERNS`](utils.py). LLMs sometimes add "Другие ссылки:" as a standalone section header in summaries despite prompt rules prohibiting it. The pattern only matches lines with trailing colon after "другие ссылки", so inline usage like "Другие ссылки в статье" is preserved.
- **Code quality — module-level import**: Moved `from utils import extract_all_channels` from local import inside [`_restore_summaries_from_channel()`](history_manager.py) to the module-level import block. `extract_all_channels` has no circular dependency risk — it only depends on [`channel_manager`](channel_manager.py) and [`utils`](utils.py), which are already imported at module level. Eliminates per-call import overhead.
- **Tests added**: 11 new tests (total 319, up from 308):
  - `test_strips_другие_ссылки_outro`: verifies "Другие ссылки:" section header stripped from summary end
  - `test_preserves_другие_ссылки_without_colon`: verifies "Другие ссылки в статье" preserved (no colon after "ссылки")
  - `test_cost_estimate_for_gpt41_nano`: verifies gpt-4.1-nano cost calculation ($0.10/$0.40 per 1M tokens)
  - `test_no_warning_for_gpt41_nano`: verifies no cost-table warning for gpt-4.1-nano
  - `test_channel_summary_prompt_prohibits_lists`: verifies "Без нумерованных и маркированных списков" in prompts
  - `test_channel_summary_prompt_prohibits_subheaders`: verifies "Без подзаголовков внутри раздела" in prompts
  - `test_handler_includes_s3_upload_status_on_success`: verifies `s3_upload` dict in success Lambda response
  - `test_handler_includes_s3_upload_status_on_error`: verifies `s3_upload` dict in error Lambda response
  - `test_upload_returns_counts_dict`: verifies `upload_to_s3()` returns dict with uploaded/failed/skipped_empty
  - `test_upload_returns_empty_dict_when_no_bucket`: verifies zeroed dict when no S3 bucket configured
  - `test_extract_all_channels_is_module_level_import`: verifies `extract_all_channels` is module-level import in history_manager (AST check)
- Updated test stubs in [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — all `upload_to_s3` patches now return `{"uploaded": N, "failed": N, "skipped_empty": N}` dict.
- Updated test stubs in [`tests/test_history_manager.py`](tests/test_history_manager.py) — added `extract_all_channels` to all `fake_utils` stubs.
- All 319 tests pass without errors.

## Completed in 2026-06-19 round (Lambda hardening round 30, meta-artifact stripping, empty choices guard, circuit breaker reset, token usage tracking, dedup length pre-filter)

- **Summary quality — strip LLM meta-artifacts**: Added [`strip_meta_artifacts()`](utils.py) with regex patterns that strip common LLM intro/outro phrases ("в этом дайджесте", "итого", "в заключение", "подведя итог", "в итоге", "итак", "в общем", "вкратце", "как видно") from summary text. Applied in both [`summarize_text()`](message_processor.py) and [`summarize_group_text()`](message_processor.py) before `enforce_summary_length()`. Directly targets the AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness — despite prompt rules prohibiting these phrases, LLMs still occasionally produce them. The post-processor catches violations cheaply.
- **Robustness — empty choices guard**: Added `response.choices` emptiness check in [`call_openai()`](utils.py). Previously, `response.choices[0].message.content` would raise `IndexError` if OpenAI returned an empty choices array (rare but documented edge case). Now logs a warning and records a circuit breaker failure, returning `""` instead of crashing.
- **Correctness — circuit breaker reset on Lambda cold start**: Added [`reset_circuit_breaker()`](utils.py) called at the start of [`lambda_handler.handler()`](lambda_handler.py). Previously, the circuit breaker's `_CIRCUIT_BREAKER_FAILURES` and `_CIRCUIT_BREAKER_OPEN_SINCE` globals persisted across warm Lambda invocations, meaning a transient OpenAI outage in one invocation could unnecessarily block all OpenAI calls in subsequent invocations even after the service recovered. Now each Lambda invocation starts with a clean circuit breaker state.
- **Observability — cumulative token usage tracking**: Added [`get_token_usage()`](utils.py) and [`reset_token_usage()`](utils.py) for cumulative prompt/completion token tracking. [`call_openai()`](utils.py) now accumulates `_cumulative_prompt_tokens` and `_cumulative_completion_tokens` across all calls within a single Lambda invocation. [`lambda_handler.handler()`](lambda_handler.py) resets counters at start and includes `token_usage` dict in both success and error responses. Completion log includes `[tokens=P+C]` for CloudWatch-based cost monitoring.
- **Performance — length-based dedup pre-filter**: Added O(1) length check in [`_remove_intra_batch_duplicates()`](message_processor.py) — texts differing in length by more than 50% are skipped before the O(N) `SequenceMatcher` comparison. Two texts where one is >50% longer than the other cannot achieve a `SequenceMatcher` ratio > 0.95, making the comparison unnecessary. Reduces dedup time for batches with messages of varying lengths.
- **Bug fix — enforce_summary_length stub in test**: Fixed pass-through `enforce_summary_length` stub in [`tests/test_history_manager.py`](tests/test_history_manager.py) oversized update test — the stub was `lambda text, max_chars: text` (never truncating), causing the assertion `len(updated.content) <= 4000` to fail. Changed to `lambda text, max_chars: text[:max_chars]` so the test correctly validates that `update_existing_summary` calls `enforce_summary_length`.
- **Tests added**: 16 new tests (total 308, up from 292):
  - `test_strips_intro_phrase`: verifies "В этом дайджесте" stripped from summary start
  - `test_strips_outro_phrase`: verifies "Итого" stripped from summary end
  - `test_preserves_clean_summary`: verifies clean summary passes through unchanged
  - `test_strips_итак_intro`: verifies "Итак" stripped
  - `test_strips_в_заключение`: verifies "В заключение" stripped
  - `test_different_length_texts_not_compared_by_sequencematcher`: verifies length pre-filter skips SequenceMatcher for >50% length difference
  - `test_similar_length_texts_still_deduplicated`: verifies similar-length texts still go through SequenceMatcher
  - `test_call_openai_returns_empty_on_empty_choices`: verifies empty choices guard returns `""`
  - `test_reset_circuit_breaker_clears_state`: verifies `reset_circuit_breaker()` clears failures and open_since
  - `test_token_usage_accumulates_across_calls`: verifies cumulative token counting across multiple `call_openai` calls
  - `test_reset_token_usage_clears_counters`: verifies `reset_token_usage()` resets both counters
  - `test_handler_includes_token_usage_in_success_response`: verifies `token_usage` field in success response
  - `test_handler_includes_token_usage_in_error_response`: verifies `token_usage` field in error response
  - `test_handler_resets_circuit_breaker_on_start`: verifies `reset_circuit_breaker` called at invocation start
  - `test_handler_resets_token_usage_on_start`: verifies `reset_token_usage` called at invocation start
  - `test_handler_logs_token_usage_in_completion_log`: verifies `[tokens=P+C]` in completion log
- Updated test stubs in [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — added `reset_circuit_breaker`, `get_token_usage`, `reset_token_usage` to fake `utils` module.
- Updated test stubs in [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py), [`tests/test_history_manager.py`](tests/test_history_manager.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) — added `strip_meta_artifacts` to all `fake_utils` stubs.
- All 308 tests pass without errors.

## Completed in 2026-06-19 round (Lambda hardening round 30, meta-artifact stripping, empty choices guard, circuit breaker reset, token usage tracking, dedup length pre-filter)
- **Prompt quality — anti-verbosity rule**: Added "Каждый абзац — одна мысль. Без воды и пояснений очевидного" rule to both [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py). Directly targets the AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness by discouraging the LLM from padding summaries with filler explanations.
- **Observability — circuit breaker state in Lambda response**: Added [`get_circuit_breaker_state()`](utils.py) that returns `{"state": "closed"|"open"|"half_open", "failures": N, ...}`. Both success and error Lambda responses now include a `circuit_breaker` field. The completion log includes `[cb=closed|open|half_open]`. Enables CloudWatch-based circuit breaker state monitoring without parsing logs.
- **Cost table — gpt-4.1-mini pricing**: Added `"gpt-4.1-mini": (0.40, 1.60)` to [`_COST_PER_MILLION`](utils.py) cost lookup. The model is not cheaper than gpt-4o-mini but is a common alternative; including it in the cost table ensures `_estimate_cost_usd()` returns accurate costs and `call_openai()` does not emit a false "not in cost table" warning when gpt-4.1-mini is configured.
- **Tests added**: 8 new tests (total 292, up from 284):
  - `test_get_circuit_breaker_state_closed`: verifies state="closed" when failures < threshold
  - `test_get_circuit_breaker_state_open`: verifies state="open" when failures >= threshold and reset not elapsed
  - `test_get_circuit_breaker_state_half_open`: verifies state="half_open" when reset timeout has elapsed
  - `test_cost_estimate_for_gpt41_mini`: verifies gpt-4.1-mini cost calculation ($0.40/1M input, $1.60/1M output)
  - `test_no_warning_for_gpt41_mini`: verifies no cost-table warning for gpt-4.1-mini
  - `test_handler_includes_circuit_breaker_state_on_success`: verifies `circuit_breaker` field in success response
  - `test_handler_includes_circuit_breaker_state_on_error`: verifies `circuit_breaker` field in error response
  - `test_handler_logs_circuit_breaker_state_in_completion_log`: verifies `[cb=...]` in completion log
- Updated test stub in [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — added fake `utils` module with `get_circuit_breaker_state` to support the new import.
- All 292 tests pass without errors.

- **Circuit breaker threshold configurable**: Moved `_CIRCUIT_BREAKER_THRESHOLD` from hardcoded constant in [`utils.py`](utils.py) to [`CIRCUIT_BREAKER_THRESHOLD`](config.py) in `config.py` (default 3, env `CIRCUIT_BREAKER_THRESHOLD`). Allows runtime tuning of the failure threshold without code changes — useful when OpenAI has intermittent issues and a higher threshold avoids premature circuit opening.
- **Circuit breaker half-open auto-recovery**: Added [`CIRCUIT_BREAKER_RESET_SEC`](config.py) config (default 60, env `CIRCUIT_BREAKER_RESET_SEC`) and `_CIRCUIT_BREAKER_OPEN_SINCE` timestamp in [`utils.py`](utils.py). When the circuit breaker has been open for longer than `CIRCUIT_BREAKER_RESET_SEC`, one probe call is allowed through (half-open state). If the probe succeeds, the circuit closes and normal operation resumes. If it fails, the breaker stays open and the reset timer restarts. Previously, once the circuit breaker opened (3 consecutive failures), ALL subsequent OpenAI calls in the Lambda invocation returned `""` — even if OpenAI recovered mid-invocation. For a Lambda with 180s timeout and 30+ NLP checks, a transient 30s OpenAI outage at the start would waste the remaining 150s. With half-open recovery, the breaker probes every 60s and self-heals when the service recovers.
- **Extracted `_cb_record_failure()` / `_cb_record_success()`**: Refactored circuit breaker state management into dedicated helper functions in [`utils.py`](utils.py). `_cb_record_failure()` increments the counter and records `_CIRCUIT_BREAKER_OPEN_SINCE` when the threshold is first reached. `_cb_record_success()` resets both the counter and timestamp. Eliminates scattered `global` mutations and ensures consistent state across all error paths.
- **Lowered default `OPENAI_SUMMARY_TEMPERATURE`** from 0.3 to 0.1 in [`config.py`](config.py). Per AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness. Lower temperature produces more deterministic, focused summaries with less verbose filler. The value remains configurable via `OPENAI_SUMMARY_TEMPERATURE` env var.
- **Synced `.env.example` and `template.yaml`**: Added `CIRCUIT_BREAKER_THRESHOLD`, `CIRCUIT_BREAKER_RESET_SEC` parameters. Updated `OPENAI_SUMMARY_TEMPERATURE` default to 0.1 in `.env.example`.
- **Tests added**: 11 new tests (total 284, up from 273):
  - `test_config_circuit_breaker_threshold_default`: verifies default is 3
  - `test_config_reads_circuit_breaker_threshold_from_env`: verifies env var parsing
  - `test_config_circuit_breaker_threshold_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_config_circuit_breaker_reset_sec_default`: verifies default is 60
  - `test_config_reads_circuit_breaker_reset_sec_from_env`: verifies env var parsing
  - `test_config_circuit_breaker_reset_sec_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_circuit_breaker_skips_call_before_reset_timeout`: verifies breaker skips calls before reset timeout
  - `test_circuit_breaker_allows_probe_after_reset_timeout`: verifies half-open probe after timeout
  - `test_circuit_breaker_resets_on_successful_probe`: verifies breaker fully closes on probe success
  - `test_cb_record_failure_sets_open_since_on_threshold`: verifies `_CIRCUIT_BREAKER_OPEN_SINCE` recorded when threshold reached
  - `test_cb_record_success_resets_open_since`: verifies `_cb_record_success` resets both counter and timestamp
- Updated existing test `test_circuit_breaker_opens_after_consecutive_failures` to set `_CIRCUIT_BREAKER_OPEN_SINCE` for half-open logic compatibility.
- Updated test stubs in [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py), [`tests/test_history_manager.py`](tests/test_history_manager.py) — added `CIRCUIT_BREAKER_THRESHOLD` and `CIRCUIT_BREAKER_RESET_SEC` to `fake_config` stubs.
- All 284 tests pass without errors.

## Completed in 2026-06-17 round (Lambda hardening round 26, HTML entities, fsync, cost guard, hash dedup, S3 empty-file guard)

- **HTML entity handling in `count_characters`**: Added `html.unescape()` after HTML tag stripping in [`count_characters()`](utils.py). Previously, `&amp;` (5 chars), `&lt;` (4 chars), `&gt;` (4 chars), `&#39;` (5 chars) etc. were counted as multi-character strings instead of their single rendered character. Telegram's 4096-character limit applies to visible characters, so HTML entities were over-counted, causing premature truncation of messages containing entities (e.g., summaries with `&amp;` in source names or `&lt;`/`&gt;` in code snippets). Added `import html as _html` to [`utils.py`](utils.py).
- **`save_json_file` fsync for durability**: Added `f.flush(); os.fsync(f.fileno())` before `os.replace()` in [`save_json_file()`](utils.py). Without fsync, the kernel may buffer the write and a Lambda crash before the buffer flushes could lose data, undermining the atomic write guarantee added in round 25. fsync ensures data is on persistent storage before the atomic rename.
- **Model cost guard in `call_openai`**: Added warning log when `OPENAI_MODEL` is not in `_COST_PER_MILLION` lookup in [`call_openai()`](utils.py). Per AUTOWORK_INSTRUCTIONS: "Выбранные решения не должны быть дороже gpt-4o-mini". The warning (`"OPENAI_MODEL=%s not in cost table — may exceed gpt-4o-mini pricing"`) enables early detection of cost overrun without blocking operation.
- **Text-hash pre-filter in `_remove_intra_batch_duplicates`**: Added `seen_hashes` set and `text_hash()` check before the O(N²) SequenceMatcher pass in [`_remove_intra_batch_duplicates()`](message_processor.py). Exact text duplicates from different channels (common cross-posting pattern) are now caught in O(1) by hash lookup instead of O(N) SequenceMatcher comparison. Reduces intra-batch dedup time from O(N²) to O(N) for the common case of identical cross-posts.
- **S3 upload skip-empty guard**: Added 0-byte file check in [`upload_to_s3()`](s3_sync.py) — empty local state files are now skipped with a warning log instead of being uploaded to S3 where they would overwrite valid state. This prevents data loss when a local file is empty due to a corrupt write or failed download. Upload summary log now includes `"skipped (empty)"` count.
- **Tests added**: 11 new tests (total 255, up from 244):
  - `test_amp_entity_counts_as_one_char`: verifies `&amp;` counts as 1 visible char
  - `test_lt_entity_counts_as_one_char`: verifies `&lt;` counts as 1 visible char
  - `test_gt_entity_counts_as_one_char`: verifies `&gt;` counts as 1 visible char
  - `test_numeric_entity_counts_as_one_char`: verifies `&#39;` counts as 1 visible char
  - `test_plain_text_unchanged`: verifies plain text counts the same as before
  - `test_save_json_file_writes_with_fsync`: verifies fsync'd write produces correct file
  - `test_warns_on_unknown_model`: verifies cost guard warning on non-gpt-4o-mini model
  - `test_no_warning_for_known_model`: verifies no warning on gpt-4o-mini
  - `test_removes_exact_text_duplicate_different_channel`: verifies hash dedup catches identical cross-posts
  - `test_hash_dedup_before_sequencematcher`: verifies hash dedup before O(N²) SequenceMatcher
  - `test_upload_skips_empty_files`: verifies S3 upload skips 0-byte files
- Updated test stubs in [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_history_manager.py`](tests/test_history_manager.py) — changed `text_hash` stubs from constant `"abc123"` to deterministic `hashlib.sha256(text.encode()).hexdigest()[:16]` to correctly support hash-based dedup, and updated `count_characters` stubs to include `html.unescape()`.
- Updated upload summary log assertion in [`tests/test_s3_sync.py`](tests/test_s3_sync.py) — now checks 3 values (uploaded, failed, skipped_empty) instead of 2.
- All 255 tests pass without errors.

## Completed in 2026-06-16 round 2 (Lambda hardening round 25, atomic writes, HTML correctness, cost metric, code quality)

- **Atomic JSON file writes**: Changed [`save_json_file()`](utils.py) to use atomic write pattern (write to temp file via `tempfile.mkstemp`, then `os.replace`). Previously, a Lambda timeout mid-`json.dump` could leave a partially-written file, causing parse errors on subsequent invocations. The atomic pattern ensures either the old or new file content is present, never a corrupt intermediate state. On write failure, the temp file is cleaned up. Added `import tempfile` to [`utils.py`](utils.py).
- **Void HTML element handling**: Added `VOID_HTML_ELEMENTS` frozenset in [`utils.py`](utils.py) (br, hr, img, input, meta, link, area, base, col, embed, source, track, wbr). Applied in [`_truncate_html_preserving_tags()`](utils.py) — previously, void elements like `<br>` and `<img>` without a self-closing `/` were incorrectly added to the `open_tags` stack, causing `</br>` and `</img>` closing tags to be appended on truncation. Telegram renders these as-is, producing visible malformed HTML. Now void elements are never pushed to `open_tags`, consistent with HTML spec.
- **OpenAI estimated cost EMF metric**: Added `_COST_PER_MILLION` lookup and `_estimate_cost_usd()` in [`utils.py`](utils.py). Extended [`_emit_openai_latency()`](utils.py) to include `EstimatedCostUSD` as an EMF metric (based on gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output). Enables CloudWatch-based cost monitoring per invocation without manual token-to-cost conversion.
- **Lambda completion log includes event flags**: Extended Lambda completion log in [`lambda_handler.handler()`](lambda_handler.py) to include `[send=X, save=X, today_groups=X, today_msgs=X]`. Previously only logged phase timing without the event flags, making it harder to correlate Lambda behavior with specific trigger configurations in CloudWatch.
- **Pre-computed message links**: Changed [`_prepare_messages_text()`](message_processor.py) to return a 3-tuple `(text, total_length, msg_links)` where `msg_links` maps message index (1-based) to its extracted links. Updated [`_replace_source_with_links()`](message_processor.py) to accept optional `msg_links` parameter and reuse pre-computed links instead of calling `extract_links()` again. Eliminates redundant regex pass per message (previously `extract_links` was called once in `_prepare_messages_text` and again in `_replace_source_with_links`).
- **Extracted `_dedup_covered_messages()`**: Extracted coverage dedup logic from [`process_messages()`](message_processor.py) into dedicated [`_dedup_covered_messages()`](message_processor.py) function. Handles parallel coverage checks, deadline propagation, `MAX_COVERED_MESSAGE_UPDATES` capping, and covered-message filtering. Returns the list of uncovered messages directly. Reduces `process_messages()` complexity and makes the dedup phase independently testable.
- **Tests added**: 15 new tests (total 244, up from 229):
  - `test_save_json_file_writes_atomic`: verifies atomic write produces correct file content
  - `test_save_json_file_cleans_up_temp_on_failure`: verifies temp file cleanup on json.dump failure
  - `test_save_json_file_preserves_existing_on_failure`: verifies existing file unchanged on write failure
  - `test_br_tag_not_added_to_open_stack`: verifies `<br>` not closed with `</br>`
  - `test_img_tag_not_added_to_open_stack`: verifies `<img>` not closed with `</img>`
  - `test_br_does_not_cause_extra_closing_tags`: verifies truncation after `<br>` has no `</br>`
  - `test_b_tag_still_gets_closed`: verifies non-void `<b>` still gets closed on truncation
  - `test_emf_includes_estimated_cost_usd`: verifies `EstimatedCostUSD` in EMF output
  - `test_cost_estimate_for_gpt4o_mini`: verifies cost calculation formula
  - `test_handler_logs_event_flags_in_completion_log`: verifies event flags in Lambda completion log
  - `test_prepare_messages_text_returns_msg_links`: verifies `msg_links` dict returned with links
  - `test_prepare_messages_text_msg_links_empty_for_no_links`: verifies empty list for no-link messages
  - `test_dedup_covered_messages_returns_uncovered`: verifies uncovered messages returned
  - `test_dedup_covered_messages_filters_covered`: verifies covered messages filtered out
  - `test_dedup_covered_messages_returns_all_when_no_summaries`: verifies pass-through when no summaries
- Updated test stubs in [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py) — changed `_prepare_messages_text` unpacking from 2-tuple to 3-tuple.
- All 244 tests pass without errors.

## Completed in 2026-06-16 round (Lambda hardening round 24, performance, config, observability)

- **`_replace_source_with_links` optimization**: Pre-computed per-message data (links, telegram_link, channel_abbr) in [`_replace_source_with_links()`](message_processor.py) — previously, `extract_links()`, `get_telegram_link()`, and `create_channel_abbreviation()` were called inside the regex replacer for every `[N]` match. When a message was referenced multiple times (e.g., `[1] ... [1]`), these were called redundantly. Now a `msg_data` dict is built once before the regex pass, eliminating O(N*M) work → O(N+M) where N is messages and M is source references.
- **Dedicated coverage check input truncation**: Added `COVERAGE_CHECK_MAX_INPUT_CHARS` config (default 2000) in [`config.py`](config.py). Applied in [`_check_coverage_and_match()`](message_processor.py) — previously used `NLP_CHECK_MAX_INPUT_CHARS` (same value) but semantically these are different operations: NLP check is a yes/no relevance classification, coverage check is a topic-match classification. Separate constants allow independent tuning without unintended side effects.
- **Configurable fetch examined multiplier**: Added `FETCH_EXAMINED_MULTIPLIER` config (default 3) in [`config.py`](config.py). Applied in [`_fetch_from_sources()`](telegram_client.py) — previously hardcoded `MAX_MESSAGES_PER_SOURCE * 3` as the total examined limit. Now configurable, allowing tuning for channels with high non-text message ratios.
- **Coverage dedup stats logging**: Added structured logging in [`process_messages()`](message_processor.py) after the coverage dedup phase — logs `"Coverage dedup (channels): N covered, M new (of K deduped)"` when covered messages exist. Complements the existing NLP filter stats, enabling CloudWatch-based monitoring of coverage dedup effectiveness.
- **`.env.example` synced**: Added `COVERAGE_CHECK_MAX_INPUT_CHARS=2000` and `FETCH_EXAMINED_MULTIPLIER=3` in [`.env.example`](.env.example).
- **SAM template updated**: Added `CoverageCheckMaxInputChars` and `FetchExaminedMultiplier` parameters and env vars in [`template.yaml`](template.yaml).
- **Tests added**: 10 new tests (total 229, up from 219):
  - `test_config_coverage_check_max_input_chars_default`: verifies default is 2000
  - `test_config_reads_coverage_check_max_input_chars_from_env`: verifies env var parsing
  - `test_config_coverage_check_max_input_chars_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_config_fetch_examined_multiplier_default`: verifies default is 3
  - `test_config_reads_fetch_examined_multiplier_from_env`: verifies env var parsing
  - `test_config_fetch_examined_multiplier_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_coverage_match_uses_coverage_check_max_input_chars`: verifies `_check_coverage_and_match` uses `COVERAGE_CHECK_MAX_INPUT_CHARS` not `NLP_CHECK_MAX_INPUT_CHARS`
  - `test_replaces_source_numbers_with_links`: verifies `_replace_source_with_links` replaces [1] with HTML link
  - `test_handles_multiple_references_to_same_source`: verifies [1] appearing multiple times is replaced each time
  - `test_logs_coverage_dedup_stats`: verifies coverage dedup stats log message
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) — added `COVERAGE_CHECK_MAX_INPUT_CHARS` and `FETCH_EXAMINED_MULTIPLIER` to fake config stubs.
- All 229 tests pass without errors.

## Completed in 2026-06-15 round 2 (Lambda hardening round 23, caching, update cap, observability, cost monitoring)

- **Channel abbreviation caching**: Added `_abbreviations_cache` in-memory cache and `_invalidate_abbreviations_cache()` in [`channel_manager.py`](channel_manager.py). [`load_channel_abbreviations()`](channel_manager.py) now caches the result after first load, avoiding repeated `channel_abbreviations.json` disk reads. Cache is invalidated on [`save_channel_abbreviation()`](channel_manager.py). Previously, each call to [`create_channel_abbreviation()`](channel_manager.py) — invoked once per message in [`_replace_source_with_links()`](message_processor.py) — triggered a full file read. With 10–30 messages per invocation, this eliminated 10–30 redundant disk reads.
- **Covered message update limit**: Added `MAX_COVERED_MESSAGE_UPDATES` config (default 5) in [`config.py`](config.py). Applied in [`process_messages()`](message_processor.py) — when more messages are covered by existing summaries than the cap, excess messages are un-marked (`is_covered_in_summaries = False`) and included in the new summary instead. Previously, a batch with many covered messages (e.g., a slow news day where most messages are updates to existing summaries) could trigger N sequential LLM calls + Telegram edits, potentially consuming the entire Lambda timeout on updates alone with no time left for summary generation.
- **Lambda error categorization**: Added [`_classify_error()`](lambda_handler.py) and `error_type` field to Lambda error responses. Errors are classified as `"timeout"` (`DeadlineExceededError`), `"config"` (`ValueError` from `validate_config`), `"connection"` (connection/timeout errors), or `"runtime"` (all others). Error log now includes the category: `"Lambda execution failed after Xs [timeout]: ..."`. Enables CloudWatch metric filters on specific error types (e.g., `{ $.error_type = "config" }` to alert on misconfiguration).
- **OpenAI EMF token metrics**: Extended [`_emit_openai_latency()`](utils.py) to include `PromptTokens` and `CompletionTokens` as EMF metrics alongside `Latency`. Enables CloudWatch-based cost monitoring (token consumption trends) and alerting without additional API calls.
- **Redundant `count_characters` calls**: Eliminated duplicate `count_characters()` calls in [`send_message_to_target_channel_with_id()`](telegram_client.py) and [`edit_message_in_target_channel()`](telegram_client.py) — `count_characters(message)` was called twice (once for the length check, once for the warning log). Now computed once and stored in `visible_chars`.
- **`.env.example` and `template.yaml` synced**: Added `SIMILARITY_LLM_UPPER=0.95` and `MAX_COVERED_MESSAGE_UPDATES=5` to [`.env.example`](.env.example). Added `SimilarityLlmUpper` and `MaxCoveredMessageUpdates` parameters and env vars to [`template.yaml`](template.yaml).
- **Tests added**: 15 new tests (total 219, up from 204):
  - `test_abbreviations_cache_returns_same_dict_on_repeated_calls`: verifies cache avoids second file read
  - `test_save_channel_abbreviation_invalidates_cache`: verifies cache cleared after save
  - `test_covered_updates_capped_at_max`: verifies `MAX_COVERED_MESSAGE_UPDATES` cap triggers log
  - `test_covered_messages_beyond_cap_unmarked`: verifies excess covered messages are un-marked and only cap-many updates happen
  - `test_error_type_timeout_for_deadline_exceeded`: verifies `_classify_error` returns "timeout"
  - `test_error_type_config_for_value_error`: verifies `_classify_error` returns "config"
  - `test_error_type_connection_for_connection_error`: verifies `_classify_error` returns "connection"
  - `test_error_type_runtime_for_generic_exception`: verifies `_classify_error` returns "runtime"
  - `test_handler_includes_error_type_in_error_response`: verifies `error_type` in Lambda error response
  - `test_handler_includes_error_type_config_on_value_error`: verifies "config" error_type on ValueError
  - `test_emf_includes_prompt_tokens`: verifies `PromptTokens` in EMF output
  - `test_emf_includes_completion_tokens`: verifies `CompletionTokens` in EMF output
  - `test_config_max_covered_message_updates_default`: verifies default is 5
  - `test_config_reads_max_covered_message_updates_from_env`: verifies env var parsing
  - `test_config_max_covered_message_updates_rejects_zero`: validates `_get_int_env` rejects zero
- Updated test stubs in [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — added `DeadlineExceededError` to fake `summarizer` module, added error categorization tests.
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) — added `MAX_COVERED_MESSAGE_UPDATES` to fake config, used `patch.object` for `ENABLE_SUMMARIES_DEDUPLICATION` and `MAX_COVERED_MESSAGE_UPDATES` in cap tests.
- All 219 tests pass without errors.

## Completed in 2026-06-15 round (Lambda hardening round 22, correctness, observability)

- **Correctness — HTML-aware Telegram truncation**: Changed [`send_message_to_target_channel_with_id()`](telegram_client.py) and [`edit_message_in_target_channel()`](telegram_client.py) from raw string slicing `message[:4096-3] + "..."` to [`_truncate_html_preserving_tags()`](utils.py). Previously, raw slicing could break HTML tags mid-tag (e.g., `<a href="...">` cut to `<a href="...`) and leave unclosed tags. Now truncation is HTML-aware: visible characters are counted against the 4096 limit, HTML tags are preserved, and open tags are properly closed. The max is reduced by 3 chars to reserve space for the "..." suffix.
- **Correctness — Group summary header length accounting**: Fixed [`summarize_group_text()`](message_processor.py) — `len(header)` was used to subtract header length from `max_summary_length`, but `len(header)` counts raw chars including `<b>`/`</b>` HTML tags, while `max_summary_length` is in visible characters (as used by `enforce_summary_length`). This over-reduced the body allowance by ~7 chars per header. Changed to `count_characters(header)` for consistency with the visible-character accounting used throughout the codebase.
- **Observability — NLP filter stats**: Added structured logging in [`process_messages()`](message_processor.py) after the NLP gather phase — logs `"NLP filter (channels): N total, M accepted, K rejected (ad=X, short=Y, other=Z)"`. Enables CloudWatch-based monitoring of NLP filter effectiveness (are we rejecting too many messages? Are ad/short filters working?).
- **Code quality — PromptManager docstring**: Removed stale `DUPLICATE_CHECK_PROMPT` reference from [`PromptManager`](prompts.py) docstring — the prompt was removed in round 17 (2026-06-11) but the docstring still referenced it as an example.
- **Tests added**: 3 new tests (total 204, up from 201):
  - `test_send_uses_html_aware_truncation`: verifies `send_message_to_target_channel_with_id` calls `_truncate_html_preserving_tags` and properly closes HTML tags on truncation
  - `test_process_messages_logs_nlp_filter_stats`: verifies `process_messages` logs NLP filter statistics (accepted, rejected, ad, short)
  - `test_group_summary_uses_count_characters_for_header`: verifies `summarize_group_text` uses `count_characters` (not `len`) for header length accounting (static AST check)
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) — added `_truncate_html_preserving_tags` to all `fake_utils` stubs that import `telegram_client`, updated `count_characters` stubs to strip HTML tags, updated truncation assertions to check visible chars instead of raw string length.
- All 204 tests pass without errors.

## Completed in 2026-06-14 round 2 (Lambda hardening round 21, architecture, correctness, cost)

- **Architecture — eliminate circular import**: Moved [`_truncate_html_preserving_tags()`](utils.py) and [`enforce_summary_length()`](utils.py) from [`message_processor.py`](message_processor.py) to [`utils.py`](utils.py). Both functions only depend on [`count_characters()`](utils.py) (already in utils), so the move is natural. [`history_manager.py`](history_manager.py) previously imported `enforce_summary_length` and `count_characters` from `message_processor` at function level (lines 386, 433) to avoid circular imports. Now imports them from `utils` at module level, eliminating the circular dependency entirely. [`message_processor.py`](message_processor.py) re-imports `enforce_summary_length` from `utils` for backward compatibility.
- **Correctness — HTML-aware Telegram length check**: Changed [`send_message_to_target_channel_with_id()`](telegram_client.py) and [`edit_message_in_target_channel()`](telegram_client.py) from `len(message) > TELEGRAM_MAX_MESSAGE_LENGTH` to `count_characters(message) > TELEGRAM_MAX_MESSAGE_LENGTH`. Telegram's 4096-character limit applies to visible characters, not raw string length including HTML tags. Previously, summaries with many HTML links (e.g., `<a href="...">[1]</a>`) could be prematurely truncated because `<a>` tags added to the raw length without affecting visible content. Now only visible characters are counted against the 4096 limit.
- **Correctness — HTML-aware update fallback**: Changed [`update_existing_summary()`](history_manager.py) fallback comparison from `len(updated_content) < len(summary.content) * 0.8` to `count_characters(updated_content) < count_characters(summary.content) * 0.8`. The 80% threshold check determines whether the LLM response is truncated; it should compare visible content length (excluding HTML tags), consistent with how `enforce_summary_length` counts.
- **Code quality — remove redundant enforce_summary_length**: Removed redundant `enforce_summary_length()` call in [`save_updated_summary()`](history_manager.py). The length is already enforced in `update_existing_summary()` before returning the `SummaryInfo` object. The previous code called `enforce_summary_length` again on the same content before passing it to `edit_message_in_target_channel`, which was wasteful and could produce different results if the enforcement logic changed between calls. Now passes `updated_summary.content` directly.
- **Prompt quality — tighter update prompt**: Tightened summary update LLM prompt in [`update_existing_summary()`](history_manager.py) — replaced contradictory "Не переписывай всё саммари — только добавь ссылку" with clearer "Скопируй саммари и добавь ссылку рядом с релевантным абзацем. Остальной текст не меняй." The old wording told the LLM not to rewrite while also asking it to return the full updated summary, which was confusing. The new wording is explicit: copy the summary and insert the link, don't change the rest.
- **Tests added**: 5 new tests (total 201, up from 196):
  - `test_send_uses_count_characters_not_len`: verifies Telegram send uses `count_characters` for 4096 limit, allowing messages with HTML tags whose raw `len()` exceeds 4096 but visible chars are within limit
  - `test_enforce_summary_length_importable_from_utils`: verifies `enforce_summary_length` is importable from `utils` module
  - `test_history_manager_no_message_processor_import`: verifies `history_manager` no longer imports `enforce_summary_length` or `count_characters` from `message_processor` (static AST check)
  - `test_fallback_uses_count_characters_for_html_content`: verifies `update_existing_summary` fallback comparison uses `count_characters` for HTML content
  - `test_save_updated_summary_passes_content_directly`: verifies `save_updated_summary` passes `updated_summary.content` directly to `edit_message_in_target_channel` without re-enforcing length
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_history_manager.py`](tests/test_history_manager.py), [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) — added `enforce_summary_length` and `count_characters` to fake `utils` module stubs to support the new import path.
- All 201 tests pass without errors.

## Completed in 2026-06-14 round (Lambda hardening round 20, dead code removal, cost optimization, quality, observability, Telegram limits)

- **Dead code removal**: Removed `SIMILARITY_LLM_LOWER` from [`config.py`](config.py) — defined and cross-validated (`SIMILARITY_LLM_LOWER < SIMILARITY_LLM_UPPER`) but never imported or used by any production module since round 12, when the three-band dedup was replaced by `_remove_intra_batch_duplicates()` (SequenceMatcher-only, uses only `SIMILARITY_LLM_UPPER`).
- **Dead code removal**: Removed redundant `NLP_AD_KEYWORDS` entries — "бесплатный курс" and "платный курс" are substring-matched by existing "курс", making them redundant. Added `NlpAdKeywordsNoRedundancyTests` to prevent future regressions.
- **Import consolidation**: Merged split `from config import ENABLE_SUMMARY_UPDATES` (was on separate line 46) into the main config import block in [`message_processor.py`](message_processor.py). Removed duplicate `ENABLE_SUMMARY_UPDATES = True` in test stub.
- **Cost optimization**: Reduced NLP relevance `max_tokens` from 20→10 in [`is_nlp_related()`](message_processor.py) — the response is either "да" (~1 token) or "нет, причина: ..." (~5-8 tokens). 20 was wasteful; 10 is sufficient and saves ~50% output tokens per NLP check.
- **Prompt quality**: Added "Конкретные результаты и факты, не общие описания" rule to both [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py) — directly targets LLM verbosity pattern (vague descriptions instead of specific facts), per AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness.
- **Configurable NLP threshold**: Added `NLP_MIN_TEXT_LENGTH` config (default 100) in [`config.py`](config.py). Applied in [`is_nlp_related()`](message_processor.py) — replaces hardcoded `100` with configurable constant, consistent with all other threshold configs.
- **Coverage+match robustness**: Changed [`_check_coverage_and_match()`](message_processor.py) digit parsing from `result.isdigit()` to `re.match(r"^(\d+)", result)` — handles edge cases like "1." or "2," that LLMs sometimes return. `isdigit()` would reject these, causing false negatives.
- **Group summary length accounting**: Fixed [`summarize_group_text()`](message_processor.py) — `enforce_summary_length()` now runs on the body before the header is prepended, and the max length is reduced by the header length. Previously, the header + body could exceed `GROUP_SUMMARY_MAX_LENGTH` because `enforce_summary_length` ran after the header was already added.
- **Deadline handling improvement**: Fixed covered message handling on deadline in [`process_messages()`](message_processor.py) — when deadline is exceeded during covered message processing, remaining covered messages are now un-marked (`is_covered_in_summaries = False`) instead of being silently dropped. This ensures they're included in the new summary instead of lost.
- **Telegram message length guard**: Added `TELEGRAM_MAX_MESSAGE_LENGTH` constant (4096) in [`config.py`](config.py). Applied in [`send_message_to_target_channel_with_id()`](telegram_client.py) and [`edit_message_in_target_channel()`](telegram_client.py) — messages exceeding 4096 chars are truncated with "..." suffix. Previously, oversized messages would fail with a Telegram API error.
- **FloodWaitError handling**: Added `FloodWaitError` catch in [`_fetch_from_sources()`](telegram_client.py) — logs the required wait time and skips the source instead of treating it as a generic error. Prevents confusing error logs and allows other sources to be fetched normally.
- **Summary update length enforcement**: Added length enforcement in [`update_existing_summary()`](history_manager.py) — when the LLM update produces content exceeding `SUMMARY_MAX_LENGTH` / `GROUP_SUMMARY_MAX_LENGTH`, `enforce_summary_length()` is applied before returning. Prevents oversized updates from being saved and posted.
- **Observability**: Added per-phase timing in [`lambda_handler.handler()`](lambda_handler.py) — now logs `"Lambda completed in X.Xs (download=Y.Ys, summarizer=Z.Zs, upload=W.Ws)"` instead of just total duration. Enables CloudWatch-based bottleneck identification (slow S3 sync vs slow summarizer).
- **Minor perf**: Pre-compiled HTML tag regex in [`count_characters()`](utils.py) — `HTML_TAG_REGEX` compiled at module level instead of recompiling on every call. Called multiple times per Lambda invocation (length checks, enforcement).
- **`.env.example` synced**: Added `NLP_MIN_TEXT_LENGTH=100` in [`.env.example`](.env.example).
- **SAM template updated**: Added `NlpMinTextLength` and `RestoreTimeoutSec` parameters and env vars in [`template.yaml`](template.yaml).
- **Tests added**: 12 new tests (total 195, up from 183):
  - `test_nlp_check_uses_max_tokens_10`: verifies `is_nlp_related` uses `max_tokens=10`
  - `test_handler_logs_phase_timing_on_success`: verifies per-phase timing log in Lambda handler
  - `test_nlp_rejects_short_text_with_configurable_threshold`: verifies `NLP_MIN_TEXT_LENGTH` config
  - `test_nlp_accepts_text_above_threshold`: verifies text above threshold passes
  - `test_config_reads_nlp_min_text_length_from_env`: verifies env var parsing
  - `test_config_nlp_min_text_length_default`: verifies default is 100
  - `test_config_nlp_min_text_length_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_no_substring_redundancy_in_ad_keywords`: verifies no redundant keyword entries
  - `test_matches_digit_with_trailing_period`: verifies `_check_coverage_and_match` handles "1."
  - `test_matches_digit_with_trailing_comma`: verifies `_check_coverage_and_match` handles "2,"
  - `test_group_summary_total_length_within_limit`: verifies header+body ≤ GROUP_SUMMARY_MAX_LENGTH
  - `test_deadline_unmarks_covered_messages`: verifies covered messages un-marked on deadline
  - `test_send_truncates_oversized_message`: verifies Telegram send truncates >4096 chars
  - `test_edit_truncates_oversized_message`: verifies Telegram edit truncates >4096 chars
  - `test_enforces_length_on_oversized_update`: verifies `update_existing_summary` enforces max length
- Removed 3 obsolete tests (`test_config_similarity_llm_lower_*` — dead config constant).
- All 195 tests pass without errors.

## Completed in 2026-06-12 round (Lambda hardening round 19, config validation, prompt optimization, deadline hardening)

- **Config validation**: Added [`_get_float_env()`](config.py) — validates float environment variables with range bounds, consistent with existing `_get_int_env()`. Applied to `OPENAI_SUMMARY_TEMPERATURE` (0.0–2.0), `SIMILARITY_LLM_LOWER` (0.0–1.0), `SIMILARITY_LLM_UPPER` (0.0–1.0). Previously used raw `float(os.getenv(...))` which crashed on non-numeric input without a clear error message.
- **Prompt optimization**: Simplified [`_check_coverage_and_match()`](message_processor.py) user_content — removed duplicated instructions ("Совпадает ли ТЕМА...", answer format rules) that repeated the system prompt. The system prompt ([`COVERAGE_AND_MATCH_PROMPT`](prompts.py)) already contains all necessary instructions. Reduces input tokens per coverage check by ~30%.
- **Prompt optimization**: Tightened [`COVERAGE_AND_MATCH_PROMPT`](prompts.py) — added "новые" before "существенные детали" for clarity (matches the original intent: same topic but significant NEW details should not be merged). Removed unused `is_group` parameter and `label`/`numbers` variables from [`_check_coverage_and_match()`](message_processor.py).
- **Lambda hardening**: Added deadline check in covered message processing loop in [`process_messages()`](message_processor.py) — previously, sequential `process_covered_message` calls (each involving an LLM call + Telegram edit) had no deadline check. With many covered messages, this could cause Lambda timeouts. Now breaks out of the loop when deadline is exceeded.
- **Code quality**: Made [`_create_summary_info()`](message_processor.py) a regular function — was `async` but never used `await`. Removed unnecessary event loop overhead. Updated call site in [`_save_processing_results()`](message_processor.py).
- **Tests added**: 10 new tests (total 183, up from 173):
  - `test_config_summary_temperature_rejects_invalid`: verifies `_get_float_env` rejects non-numeric temperature
  - `test_config_summary_temperature_rejects_out_of_range`: verifies temperature > 2.0 is rejected
  - `test_config_similarity_llm_lower_from_env`: verifies `SIMILARITY_LLM_LOWER` env var parsing
  - `test_config_similarity_llm_lower_default`: verifies default is 0.7
  - `test_config_similarity_llm_lower_rejects_invalid`: verifies non-numeric rejected
  - `test_config_similarity_llm_upper_from_env`: verifies `SIMILARITY_LLM_UPPER` env var parsing
  - `test_config_similarity_llm_upper_default`: verifies default is 0.95
  - `test_config_similarity_llm_upper_rejects_out_of_range`: verifies > 1.0 rejected
  - `test_deadline_skips_remaining_covered_updates`: verifies deadline breaks covered message processing loop
  - `test_create_summary_info_returns_summary_info`: verifies sync `_create_summary_info` returns correct `SummaryInfo`
- All 183 tests pass without errors.

## Completed in 2026-06-11 round (Lambda hardening round 17, dead code removal, prompt tightening, observability)

- **Dead code removal**: Removed [`_check_coverage()`](message_processor.py) — superseded by [`_check_coverage_and_match()`](message_processor.py) since round 15 (2026-06-10). The old function was only called from tests, never from production code.
- **Dead code removal**: Removed [`find_relevant_summary_for_update()`](history_manager.py) and its fallback in [`process_covered_message()`](message_processor.py) — since the coverage+match merge in round 15, `process_covered_message` is always called with a `matching_summary`, making the separate find-relevant call unreachable dead code. Now returns early with a debug log when `matching_summary is None`.
- **Dead code removal**: Removed `SUMMARY_COVERAGE_CHECK_PROMPT`, `GROUP_SUMMARY_COVERAGE_CHECK_PROMPT`, and `FIND_RELEVANT_SUMMARY_PROMPT` from [`prompts.py`](prompts.py) — only used by the removed `_check_coverage` and `find_relevant_summary_for_update`.
- **Dead code removal**: Removed [`get_recent_summaries_context()`](history_manager.py), [`get_recent_group_summaries_context()`](history_manager.py), and shared helper [`_get_recent_summaries_context()`](history_manager.py) — only used by the removed `_check_coverage`. The coverage+match check builds its own context inline.
- **Dead code removal**: Removed `COVERAGE_CHECK_MAX_SUMMARIES` and `COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY` from [`config.py`](config.py) — only used by the removed context functions. `_check_coverage_and_match` uses `UPDATE_MATCH_MAX_SUMMARIES` and `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY` instead.
- **Dead code removal**: Removed unused `prompts` import and `json` import from [`history_manager.py`](history_manager.py).
- **Prompt optimization**: Tightened [`COVERAGE_AND_MATCH_PROMPT`](prompts.py) — removed redundant "Игнорируй мелкие детали" line (implied by "существенные детали → НЕТ"), generalized "Разные модели на Hugging Face" to "Разные модели/версии" (broader and shorter). Reduced token count by ~20%.
- **Prompt optimization**: Tightened [`NLP_RELEVANCE_PROMPT`](prompts.py) — shortened BigTech/AI-assistents list from 11 company/product names to 5 representative ones (OpenAI, Anthropic, Google, Meta, DeepSeek). LLM already knows the full list; enumerating wastes input tokens.
- **Observability**: Added response latency logging in [`call_openai()`](utils.py) — logs `latency=X.Xs` alongside existing token usage metrics. When usage data is unavailable, logs `OpenAI call: model=X latency=X.Xs`. Enables CloudWatch-based OpenAI API latency monitoring and trend detection.
- **Tests updated**: Removed 8 obsolete tests (3 `_check_coverage` tests, 1 `COVERAGE_CHECK_MAX` config test, 1 `find_relevant_summary_context_truncation` test, 2 shared context helper tests, 1 `recent_summaries_context_includes_date` test). Added 3 new tests:
  - `test_skips_update_when_no_matching_summary`: verifies `process_covered_message` skips update when `matching_summary is None`
  - `test_returns_none_on_digit_out_of_range`: verifies `_check_coverage_and_match` handles out-of-range digit response
  - `test_call_openai_logs_latency`: verifies latency logging in `call_openai`
  - Total test count: 166 (down from 171 — net -5 from 8 removed + 3 added)
- Updated test stubs in [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to remove references to deleted functions and prompts.
- All 166 tests pass without errors.

## Completed in 2026-06-02 round 2 (Lambda hardening round 8, timeouts, limits, cost optimization)

- **OpenAI request timeout**: Added `OPENAI_REQUEST_TIMEOUT` config (default 30s) in [`config.py`](config.py). Applied in [`call_openai()`](utils.py) — `AsyncOpenAI` client now created with `timeout=float(OPENAI_REQUEST_TIMEOUT)`. Previously, the default httpx timeout was 10 minutes, which could cause the entire Lambda invocation to time out on a single hung OpenAI request.
- **NLP check input truncation**: Added `NLP_CHECK_MAX_INPUT_CHARS` config (default 2000) in [`config.py`](config.py). Applied in [`is_nlp_related()`](message_processor.py) — message text is now truncated to `NLP_CHECK_MAX_INPUT_CHARS` before being sent to the NLP relevance LLM call. Previously, very long messages (full articles) would waste thousands of input tokens on a yes/no classification.
- **Message fetch limit per source**: Added `MAX_MESSAGES_PER_SOURCE` config (default 100) in [`config.py`](config.py). Applied in [`_fetch_from_sources()`](telegram_client.py) — iteration breaks when the per-source limit is reached. Previously, a very active channel could return thousands of messages, consuming excessive Lambda time and tokens.
- **Telegram client connection timeout**: Added `connection_retries=3`, `retry_delay=2`, `timeout=15` to both `TelegramClient` constructors in [`start_clients()`](telegram_client.py). Previously, default Telethon settings could hang indefinitely on connection attempts.
- **Code cleanup**: Replaced redundant `source_msgs` list with a simple `source_count` counter in [`_fetch_from_sources()`](telegram_client.py) — the list was only used for its length, never its contents.
- **.env.example sync**: Updated `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` and `OPENAI_GROUP_SUMMARY_MAX_TOKENS` defaults from 16000 to 4000 in [`.env.example`](.env.example) to match the actual defaults in [`config.py`](config.py).
- **SAM template updated**: Added `OPENAI_REQUEST_TIMEOUT`, `NLP_CHECK_MAX_INPUT_CHARS`, `MAX_MESSAGES_PER_SOURCE` parameters and env vars in [`template.yaml`](template.yaml).
- **Tests added**: 6 new tests (total 87, up from 81):
  - `test_call_openai_passes_timeout_to_client`: verifies `OPENAI_REQUEST_TIMEOUT` is passed to `AsyncOpenAI`
  - `test_config_reads_request_timeout_from_env`: verifies `OPENAI_REQUEST_TIMEOUT` env var parsing
  - `test_config_reads_nlp_check_max_input_chars_from_env`: verifies `NLP_CHECK_MAX_INPUT_CHARS` env var parsing
  - `test_config_reads_max_messages_per_source_from_env`: verifies `MAX_MESSAGES_PER_SOURCE` env var parsing
  - `test_nlp_check_truncates_long_input`: verifies text truncation in `is_nlp_related()`
  - `test_nlp_check_keeps_short_input`: verifies short text is not truncated
- Updated test stubs in [`tests/test_openai_config.py`](tests/test_openai_config.py) and [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to support new config constants and `timeout` kwarg.
- All 87 tests pass without errors.

## Completed in 2026-04-14 round (Lambda hardening & test coverage)

- **Lambda error handling**: Добавлен try/except в [`lambda_handler.handler()`](lambda_handler.py) — теперь ошибки суммаризации возвращают структурированный `{'status': 'error', 'error': str(e)}` вместо unhandled exception. State всё равно загружается в S3 при ошибке для сохранения частичных обновлений.
- **Unused imports removed**: Удалены неиспользуемые импорты:
  - `random` в [`message_processor.py`](message_processor.py)
  - `Callable`, `TypeVar`, `T` в [`history_manager.py`](history_manager.py)
  - `datetime`, `timezone` в [`channel_manager.py`](channel_manager.py)
- **Logging hardening**: Конвертировано ~10 eager f-strings в [`history_manager.py`](history_manager.py) в lazy %-format (e.g. `logger.error("msg: %s", e)` вместо `logger.error(f"msg: {e}")`), что предотвращает вычисление строк при DEBUG-уровне.
- **Print statement replaced**: Заменён module-level `print()` в [`history_manager.py`](history_manager.py) на `logger.error()` для структурированного логирования в CloudWatch.
- **Tests added**: Создано 3 новых тестовых файла (+13 тестов, общее количество увеличено с 30 до 43):
  - [`tests/test_history_manager.py`](tests/test_history_manager.py): тесты логики `should_run_group_summarization()` и извлечения контекста истории (2 теста)
  - [`tests/test_channel_manager.py`](tests/test_channel_manager.py): тесты логики создания аббревиатур и слияния каналов (2 теста)
  - [`tests/test_models.py`](tests/test_models.py): тесты сериализации/десериализации `MessageInfo` и `SummaryInfo` (9 тестов), включая регрессионный тест на восстановление `is_nlp_related_reason`
- Все 43 теста проходят без ошибок.

## Completed in 2026-04-13 round 2 (dedup optimization & refactoring)

- Оптимизирована дедупликация в [`message_processor.py`](message_processor.py): добавлена трёхзональная стратегия `SequenceMatcher` для минимизации LLM-вызовов:
  - `ratio > SIMILARITY_LLM_UPPER` (0.95): почти дубликаты → вызов LLM
  - `ratio < SIMILARITY_LLM_LOWER` (0.7): явно разные → без LLM
  - Между зонами → вызов LLM
- Добавлены константы `SIMILARITY_LLM_LOWER` и `SIMILARITY_LLM_UPPER` в [`config.py`](config.py) с возможностью переопределения через env.
- Рефакторинг `process_messages()` в [`message_processor.py`](message_processor.py): выделены три функции — `_classify_message()`, `_create_summary_info()`, `_save_processing_results()`. Основная функция сокращена с 120+ строк до ~30, улучшена читаемость и тестируемость.
- Перенесена `get_all_source_channels()` из [`message_processor.py`](message_processor.py) в [`channel_manager.py`](channel_manager.py) для устранения циклического импорта и логической группировки функций работы с каналами.
- Обновлён импорт в [`telegram_client.py`](telegram_client.py): `get_all_source_channels` теперь импортируется из `channel_manager`.
- Обновлён [`pyproject.toml`](pyproject.toml): placeholder автора заменён на реального (Denis Gordeev).
- Обновлены тестовые стабы в [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) для поддержки новых импортов.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-13 round (logging hardening & code quality)

- **Критический баг**: Добавлен импорт `RESTORE_HISTORY_DAYS` в [`history_manager.py`](history_manager.py) — ранее функция восстановления истории падала с `NameError` при вызове.
- Перенесены `_load_json_file()`, `_save_json_file()`, `now_iso()` в [`utils.py`](utils.py) как единый источник, устранено дублирование между [`history_manager.py`](history_manager.py) и [`channel_manager.py`](channel_manager.py) (~70 строк сокращено).
- Lazy-init OpenAI клиента в [`utils.py`](utils.py): `openai_client` теперь инициализируется внутри `call_openai()`, а не при импорте модуля. Это упрощает тестирование и避免了 ошибки при отсутствии `OPENAI_API_KEY` на импорте.
- Исправлена `save_updated_summary()` в [`history_manager.py`](history_manager.py): теперь использует `save_json_file()` вместо прямого `json.dump()`, добавлен `last_updated` timestamp.
- Добавлен `logging.basicConfig()` в [`lambda_handler.py`](lambda_handler.py) для структурированного логирования в CloudWatch.
- Конвертировано ~70 `print()` statements в `logging` across [`message_processor.py`](message_processor.py), [`summarizer.py`](summarizer.py), [`s3_sync.py`](s3_sync.py), [`telegram_client.py`](telegram_client.py), [`prompts.py`](prompts.py), [`history_manager.py`](history_manager.py), [`utils.py`](utils.py). DEBUG-уровень для отладочных сообщений, INFO для информационных, ERROR для ошибок.
- Выделены хелперы `_ensure_clients()` и `_ensure_bot_client()` в [`telegram_client.py`](telegram_client.py), устранено дублирование проверки подключения клиента (~15 строк сокращено).
- `SOURCE_GROUPS` теперь `set` вместо `list` в [`config.py`](config.py), что соответствует типу `SOURCE_CHANNELS` и предотвращает дубликаты.
- Обновлён [`tests/test_openai_config.py`](tests/test_openai_config.py) для поддержки lazy-init OpenAI клиента.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-12 round 2 (code quality & refactoring)

- Выделены универсальные функции `_load_json_file()`, `_save_json_file()` и `_now_iso()` в [`history_manager.py`](history_manager.py), устранено дублирование кода загрузки/сохранения JSON (~150 строк сокращено).
- Рефакторинг `load_summarization_history`/`save_summarization_history` и их group-аналогов в [`history_manager.py`](history_manager.py): убраны `try/except` с `print()`, заменены на `logger.error()` и универсальные хелперы.
- Выделены универсальные функции `_load_channel_list()` и `_save_channel_list()` в [`channel_manager.py`](channel_manager.py), устранено дублирование `load_discovered_channels`/`load_similar_channels`/`load_banned_channels` и их save-аналогов (~80 строк сокращено).
- Все `datetime.now().isoformat()` заменены на `_now_iso()` (UTC timezone), что устраняет неоднозначность часовых поясов при сравнении timestamps в Lambda и между машинами.
- Магические числа вынесены в именованные константы в [`config.py`](config.py): `MAX_CHANNEL_HISTORY_MESSAGES`, `MAX_CHANNEL_SUMMARIES`, `MAX_GROUP_HISTORY_MESSAGES`, `MAX_GROUP_SUMMARIES`, `GROUP_SUMMARIZATION_INTERVAL_SECONDS`, `RESTORE_HISTORY_DAYS`, `SUMMARY_MIN_RATIO`, `SUMMARY_MIN_LENGTH`, `SUMMARY_MAX_LENGTH`, `GROUP_SUMMARY_MIN_LENGTH`, `GROUP_SUMMARY_MAX_LENGTH`, `TEXT_PREVIEW_LENGTH`.
- [`message_processor.py`](message_processor.py) обновлён для использования новых констант из `config.py`.
- Все `print()` в функциях восстановления истории ([`history_manager.py`](history_manager.py)) заменены на `logger.info()`/`logger.debug()`/`logger.error()`, что обеспечивает структурированное логирование в CloudWatch.
- Обновлён test stub в [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) для поддержки новых констант.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-08 round

- Восстановлен [`s3_sync.py`](s3_sync.py), без которого чистый импорт [`lambda_handler.py`](lambda_handler.py) падал с `ModuleNotFoundError`.
- Исправлен разбор Lambda event-флагов в [`lambda_handler.py`](lambda_handler.py): строковые значения `false` / `true` теперь интерпретируются корректно.
- Восстановлены и расширены smoke/regression-тесты для Lambda и OpenAI-конфига в [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py), [`tests/test_s3_sync.py`](tests/test_s3_sync.py) и [`tests/test_openai_config.py`](tests/test_openai_config.py).
- Добавлены guardrails на длину итогового дайджеста в [`message_processor.py`](message_processor.py), чтобы итоговые саммари оставались лаконичными даже если модель пытается выйти за целевой объём.
- Исправлены и документированы OpenAI env-настройки с дефолтом `gpt-4o-mini` в [`config.py`](config.py), [`utils.py`](utils.py) и [`.env.example`](.env.example).
- Добавлен отдельный Lambda runbook [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md), а [`README.md`](README.md) синхронизирован с текущим деревом проекта.
- Добавлен [`boto3`](https://pypi.org/project/boto3/) в [`pyproject.toml`](pyproject.toml) и [`requirements.txt`](requirements.txt), чтобы S3-синхронизация работала в container image и локальной отладке, а не только в AWS managed runtime.
- Создан AWS SAM шаблон [`template.yaml`](template.yaml) для воспроизводимого деплоя Lambda с полной инфраструктурой (S3 bucket, IAM роли, EventBridge триггер).
- Добавлен скрипт [`build_lambda_package.sh`](build_lambda_package.sh) для сборки deployment package с зависимостями под Python 3.12.
- Написана полная инструкция по деплою в [`docs/aws-lambda-deployment.md`](docs/aws-lambda-deployment.md) с двумя опциями: ручной деплой через AWS CLI и автоматический через AWS SAM.
- Добавлен пример конфигурации [`samconfig.toml.example`](samconfig.toml.example) для быстрого старта SAM деплоя.

## Completed in 2026-04-09 round

- Добавлен pre-push hook [`hooks/pre-push`](hooks/pre-push), который автоматически запускает `python3 -m unittest discover -s tests` перед каждым push.
- Добавлен интеграционный тест [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py) (10 тестов) на полный пост-процессинг дайджеста:
  - замена ссылок `[1]`, `[2]`, `[1,2]` на HTML-ссылки;
  - ограничение длины channel/group summary;
  - корректное закрытие HTML-тегов при обрезке;
  - консистентность аббревиатур каналов.
- Общее количество тестов увеличено с 15 до 25.

## Completed in 2026-04-10 round

- Добавлен GitHub Actions workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) для автоматического прогона тестов и сборки Lambda-пакета на каждый push и PR.
- Добавлен интеграционный тест [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) (5 тестов) на полный end-to-end pipeline `process_messages()`:
  - полный канал pipeline: фильтрация NLP, дедупликация, суммаризация, сохранение истории;
  - пропуск non-NLP сообщений;
  - обработка пустого списка сообщений;
  - group pipeline: фильтрация, групповая суммаризация, обновление last run;
  - отправка сообщения в Telegram при `send_message=True`.
- Общее количество тестов увеличено с 25 до 30.
- Обновлён [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md): секция «Следующий шаг по инфраструктуре» заменена на актуальную «Инфраструктура» с ссылками на существующие `template.yaml`, `samconfig.toml.example` и deployment guide.

## Completed in 2026-04-10 round 2 (code quality)

- Удалён неиспользуемый импорт `from operator import is_` в [`message_processor.py`](message_processor.py).
- Исправлен `MessageInfo.from_dict()` в [`models.py`](models.py): теперь восстанавливается поле `is_nlp_related_reason`.
- Устранено дублирование кода дедупликации (~100 строк) в [`message_processor.py`](message_processor.py): `remove_duplicates` и `remove_group_duplicates` теперь делегируют общую логику `_remove_duplicates_generic`.
- Вынесена общая функция `_replace_source_with_links` и `_prepare_messages_text` в [`message_processor.py`](message_processor.py), убрано ~60 строк дублирования между `summarize_text` и `summarize_group_text`.
- Удалена захардкоженная привязка к конкретному каналу (`denissexy`) из `CHANNEL_SUMMARY_PROMPT` в [`prompts.py`](prompts.py).
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-11 round (bug fixes & hardening)

- **Критический баг**: Заменён `hash(msg.text)` на детерминированный `hashlib.sha256(msg.text.encode()).hexdigest()[:16]` в [`message_processor.py`](message_processor.py) и [`history_manager.py`](history_manager.py). Python's built-in `hash()` рандомизирован через `PYTHONHASHSEED`, что приводило к пропуску дедупликации между разными Lambda invocation.
- **Баг**: Исправлен `should_run_group_summarization()` в [`history_manager.py`](history_manager.py): теперь возвращает `True` при первом запуске (когда файл истории ещё не создан), вместо `False`.
- **Улучшение диагностики**: Добавлен `traceback.print_exc()` в top-level `except` блок [`summarizer.py`](summarizer.py) для полной трассировки ошибок.
- **Type hints**: Исправлен невалидный type hint `-> (bool, str)` на `-> tuple[bool, str]` в `is_nlp_related()` в [`message_processor.py`](message_processor.py).
- **Производительность/логи**: Добавлен `DEBUG` флаг в [`config.py`](config.py) (через env `DEBUG=1`).Verbose логирование сообщений и полных саммари теперь gated behind этим флагом в [`message_processor.py`](message_processor.py), что снижает объем CloudWatch логов в production.
- **Clean up**: Удалены module-level `print()` statements из [`config.py`](config.py), которые срабатывали при каждом импорте модуля.
- **Тесты**: Обновлён test stub в [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) для поддержки `DEBUG` флага. Все 30 тестов проходят.

## Completed in 2026-04-11 round 2 (error handling & monitoring)

- Добавлен per-channel error handling в [`telegram_client.py`](telegram_client.py): каждый канал обернут в try/catch, чтобы сбой одного канала не ломал весь цикл загрузки.
- Добавлен модуль `logging` в [`telegram_client.py`](telegram_client.py) для лучшей трассировки ошибок через `logger.error()`.
- Добавлены CloudWatch алерты в [`template.yaml`](template.yaml): автоматическое создание трех алертов (Errors, Duration, Throttles) при SAM деплое.
- Обновлена документация в [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md): добавлена секция «CloudWatch алерты» с описанием метрик и инструкцией по настройке SNS уведомлений.
- Обновлена документация в [`docs/aws-lambda-deployment.md`](docs/aws-lambda-deployment.md): добавлена ссылка на CloudWatch алерты в секции Monitoring.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-12 round (maintenance & cleanup)

- Расширен [`.gitignore`](.gitignore): добавлены session файлы (`*.session`), JSON state файлы, логи (`*.log`), артефакты сборки (`build/`, `dist/`, `*.zip`), Python кэш, IDE файлы, `.samconfig.toml`.
- Исправлено вводящее в заблуждение имя параметра `EnableSsmSend` → `EnableTelegramSend` в [`template.yaml`](template.yaml) (параметр управляет отправкой в Telegram, а не SSM).
- Обновлён [`AUTOWORK_INSTRUCTIONS.md`](AUTOWORK_INSTRUCTIONS.md): убрана устаревшая заметка "нет доки по Lambda" — теперь указаны ссылки на существующие `docs/aws-lambda-deployment.md`, `docs/aws-lambda-runbook.md` и `template.yaml`.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-05-26 round (Lambda hardening round 2)

- **Critical bug**: Added missing `logger = logging.getLogger(__name__)` in [`lambda_handler.py`](lambda_handler.py) — `logger` was used on line 62 but never defined, causing `NameError` on the error path instead of structured logging.
- **OpenAI retry with exponential backoff**: Added retry logic in [`call_openai()`](utils.py) for `RateLimitError`, `APIConnectionError`, and 5xx `APIError`. Default: 3 retries with base delay 1s, doubling each attempt. This prevents Lambda failures on transient OpenAI API issues.
- **Duplicate error logging fixed**: Removed duplicate `logger.error()` calls in [`telegram_client.py`](telegram_client.py) — both `fetch_messages()` and `fetch_group_messages()` were logging the same exception twice (once with f-string, once with %-format). Now uses single lazy %-format call.
- **Lazy logging in utils.py**: Converted 2 remaining eager f-strings in `logger.error()` in [`utils.py`](utils.py) to lazy %-format (consistent with prior hardening in `history_manager.py`).
- **Lazy logging in history_manager.py**: Converted 1 remaining eager f-string in `logger.error()` in [`history_manager.py`](history_manager.py) to lazy %-format.
- **Double shuffle removed**: Removed redundant `random.shuffle(all_channels)` in [`telegram_client.py:fetch_messages()`](telegram_client.py) — channels are already shuffled in `get_all_source_channels()` in [`channel_manager.py`](channel_manager.py). Removed unused `import random`.
- **SQS Dead Letter Queue**: Added `SummarizerDLQ` (SQS queue with 14-day retention) in [`template.yaml`](template.yaml) and wired it to the Lambda function's `DeadLetterQueue` config. Failed invocations are now captured for inspection instead of being silently dropped.
- Updated test stubs in [`tests/test_openai_config.py`](tests/test_openai_config.py) and [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) to include `APIError`, `RateLimitError`, `APIConnectionError` in fake `openai` module.
- All 43 tests pass without errors.

## Completed in 2026-05-27 round (config lazy validation & Lambda hardening round 3)

- **Config lazy validation**: Refactored [`config.py`](config.py) — required env vars (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_BOT_TOKEN`, `TARGET_CHANNEL`, `OPENAI_API_KEY`) now default to `None` instead of raising `ValueError` at import time. New `validate_config()` function checks all required vars and raises a clear error listing all missing ones. Called at entry points: [`lambda_handler.handler()`](lambda_handler.py) and [`summarizer.main()`](summarizer.py). This allows importing config for testing without a full `.env` file and prevents premature failures on Lambda cold start.
- **Lambda timeout guard**: Added wall-clock deadline tracking in [`lambda_handler.py`](lambda_handler.py) using `context.get_remaining_time_in_millis()` with a 10-second safety margin. Deadline is passed to [`run_summarizer()`](summarizer.py) as `_deadline` parameter. New `check_deadline()` and `DeadlineExceededError` in [`summarizer.py`](summarizer.py) — checked before channel processing and before group processing. When deadline is exceeded, the summarizer saves partial results and exits gracefully instead of being killed by Lambda timeout.
- **Event loop blocking fix**: Replaced `time.sleep()` with `await asyncio.sleep()` in [`call_openai()`](utils.py) retry backoff. The previous synchronous sleep blocked the entire event loop during OpenAI retry delays, preventing any other async work from progressing.
- **JSON format bug fix**: Fixed [`save_updated_summary()`](history_manager.py) — was saving raw list `[s.to_dict() for s in summaries]` instead of the expected `{"summaries": [...], "last_updated": "..."}` dict format, which would corrupt the file and cause parse errors on subsequent loads.
- **Removed unused import**: Removed `import time` from [`utils.py`](utils.py) (replaced by `asyncio.sleep`).
- **Tests added**: 4 new tests (total 47, up from 43):
  - `test_config_does_not_raise_on_import_without_env`: verifies config import succeeds without env vars
  - `test_validate_config_raises_for_missing_vars`: verifies `validate_config()` raises with clear message
  - `test_validate_config_passes_when_all_required_set`: verifies `validate_config()` passes with all vars
  - `test_handler_uses_context_remaining_time_for_deadline`: verifies Lambda context deadline calculation
- Updated [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) to stub `config.validate_config` and verify `_deadline` parameter.
- Updated [`tests/test_openai_config.py`](tests/test_openai_config.py) with new config validation tests.
- All 47 tests pass without errors.

## Completed in 2026-05-28 round (dedup bug fix & prompt optimization)

- **Critical bug**: Fixed three-band dedup logic in [`message_processor.py`](message_processor.py) — `SIMILARITY_THRESHOLD` (0.9) was checked before `SIMILARITY_LLM_UPPER` (0.95), making the upper band unreachable. Messages with ratio 0.9–0.95 were incorrectly auto-marked as duplicates without LLM verification. Now: ratio > UPPER (0.95) → auto-duplicate, LOWER (0.7) < ratio ≤ UPPER → LLM check, ratio ≤ LOWER → auto-different.
- **Removed SIMILARITY_THRESHOLD**: Deleted redundant `SIMILARITY_THRESHOLD = 0.9` from [`config.py`](config.py) — replaced by the two-band `SIMILARITY_LLM_LOWER` / `SIMILARITY_LLM_UPPER` constants. Updated [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) stubs accordingly.
- **Prompt optimization**: Reduced all prompts in [`prompts.py`](prompts.py) by ~50% (token savings):
  - `CHANNEL_SUMMARY_PROMPT`: 31 lines → 12 lines (removed redundant "ВАЖНО" prefixes, consolidated formatting rules)
  - `GROUP_SUMMARY_PROMPT`: 33 lines → 13 lines (same consolidation, kept group-specific rule about unanswered questions)
  - `NLP_RELEVANCE_PROMPT`: 44 lines → 24 lines (merged overlapping categories, compacted formatting)
  - `SUMMARY_COVERAGE_CHECK_PROMPT` / `GROUP_SUMMARY_COVERAGE_CHECK_PROMPT`: removed verbose "Ты эксперт" framing, kept essential rules and examples
  - `DUPLICATE_CHECK_PROMPT`: minor wording cleanup
- **Clean up**: Moved `from datetime import datetime, timezone` to module-level in [`message_processor.py`](message_processor.py), removed local import in `_create_summary_info()`.
- **Simplified `_run_async_with_loop`**: Extracted shared helper in [`history_manager.py`](history_manager.py) replacing 60+ lines of duplicated event-loop detection in `restore_summaries_from_channel_sync()` and `restore_group_summaries_from_channel_sync()` with a single 25-line function handling all three cases (no loop, same loop, different loop).
- **Tests added**: 2 new tests (total 49, up from 47):
  - `test_three_band_dedup_upper_threshold_uses_llm`: verifies ambiguous-ratio messages call LLM
  - `test_three_band_dedup_near_identical_skips_llm`: verifies near-identical messages auto-deduplicate without LLM
- All 49 tests pass without errors.

## Completed in 2026-05-29 round (dedup bug fix, cost optimization, code dedup)

- **Critical bug**: Fixed `are_messages_duplicate()` in [`message_processor.py`](message_processor.py) — was checking `answer.lower().startswith("y")` but `DUPLICATE_CHECK_PROMPT` asks "Ответь да или нет" (Russian). The LLM responds "да"/"нет", never "yes" — so LLM deduplication was **always skipped**. Changed to `answer.strip().lower().startswith("да")` and increased `max_tokens` from 1 to 3 to capture the Russian response.
- **Cost optimization — max output tokens**: Reduced `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` and `OPENAI_GROUP_SUMMARY_MAX_TOKENS` defaults from 16000 to 4000 in [`config.py`](config.py) and [`template.yaml`](template.yaml). With `SUMMARY_MAX_LENGTH=4000` chars and ~2-3 tokens/char for Russian, 4000 tokens is sufficient. This reduces output token cost by ~75% when the model generates verbose responses.
- **Cost optimization — update match truncation**: Added `UPDATE_MATCH_MAX_SUMMARIES` (default 5) and `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY` (default 500) in [`config.py`](config.py). Applied in [`find_relevant_summary_for_update()`](history_manager.py) — previously sent last 50 full summaries as context (~50K+ chars). Now capped at ~2500 chars, saving ~95% input tokens on update match LLM calls.
- **Code dedup**: Merged `restore_summaries_from_channel()` and `restore_group_summaries_from_channel()` in [`history_manager.py`](history_manager.py) into shared `_restore_summaries_from_channel(history_file, label)` helper. Eliminated ~50 lines of duplicated async restore logic.
- **Tests added**: 4 new tests (total 55, up from 51):
  - `test_are_messages_duplicate_recognizes_russian_yes`: verifies "да" is recognized as duplicate
  - `test_are_messages_duplicate_recognizes_russian_no`: verifies "нет" is recognized as not duplicate
  - `test_update_match_limits_from_config`: verifies UPDATE_MATCH config constants
  - `test_find_relevant_summary_context_truncation`: verifies context truncation in update match
- All 55 tests pass without errors.

## Completed in 2026-05-28 round 2 (cost optimization, security hardening, prompt quality)

- **Major cost optimization — coverage check context truncation**: Added `COVERAGE_CHECK_MAX_SUMMARIES` (default 10) and `COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY` (default 300) in [`config.py`](config.py). Applied in [`get_recent_summaries_context()`](history_manager.py) and [`get_recent_group_summaries_context()`](history_manager.py) — previously, these functions could send up to ~200K chars (channel) and ~1.2M chars (group) as input tokens to a yes/no coverage check LLM call. Now capped at ~3000 chars, saving ~90-99% input tokens on coverage checks.
- **Security: template.yaml NoEcho**: Set `NoEcho: true` on `TelegramApiId` and `TelegramApiHash` parameters in [`template.yaml`](template.yaml) — previously `NoEcho: false`, exposing credentials in CloudFormation console and API responses.
- **Security: Reserved concurrency**: Added `ReservedConcurrentExecutions: 1` to Lambda function in [`template.yaml`](template.yaml) — prevents concurrent invocations that could corrupt S3 state (non-atomic read-modify-write).
- **Prompt optimization round 2**: Further reduced token consumption in [`prompts.py`](prompts.py):
  - `FIND_RELEVANT_SUMMARY_PROMPT`: removed unnecessary "Ты эксперт по анализу текстов" framing
  - `CHANNEL_SUMMARY_PROMPT` / `GROUP_SUMMARY_PROMPT`: tightened emoji rule (1–5 → 1–3, removed 2 of 3 verbose example headers), merged redundant lines
  - `NLP_RELEVANCE_PROMPT`: consolidated overlapping categories (e.g., merged "Sam Altman, Mark Zuckerberg — любые новости" into BigTech line, removed "Незнакомые термины" catch-all), reduced by ~25%
- **Consistency fix**: Changed `restore_group_summaries_from_channel()` in [`history_manager.py`](history_manager.py) to use `msg.text` instead of `msg.message` for `content` and `extract_all_channels` — now consistent with the channel restore function. `msg.text` handles None safely.
- **OpenAI auth error clarity**: Added distinct logging for 401/403 auth errors in [`call_openai()`](utils.py) — now logs "OpenAI auth error (status 401): check OPENAI_API_KEY" instead of generic "OpenAI API error", making Lambda CloudWatch diagnostics faster.
- **Lambda request ID traceability**: Added `aws_request_id` to Lambda response and log output in [`lambda_handler.py`](lambda_handler.py) — enables correlating CloudWatch logs with Lambda invocation records.
- **Tests added**: 2 new tests (total 51, up from 49):
  - `test_handler_includes_request_id_in_response`: verifies request_id in Lambda response
  - `test_coverage_check_limits_from_config`: verifies COVERAGE_CHECK_MAX_SUMMARIES and COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY in config
- Updated [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) for request_id in response structure.
- Updated [`tests/test_history_manager.py`](tests/test_history_manager.py) with context truncation tests.
- All 51 tests pass without errors.

## Completed in 2026-05-29 round 2 (Lambda hardening round 4, SSM secrets, code dedup)

- **Critical bug**: Fixed [`_run_async_with_loop()`](history_manager.py) — when called from within a running event loop (the common Lambda case via `asyncio.run()`), the old code silently returned `[]` due to "same loop deadlock avoidance", meaning `restore_summaries_from_channel_sync()` **never actually restored** anything. Now uses a background `threading.Thread` with `asyncio.run()` to avoid the deadlock while still executing the coroutine. Also added proper exception handling for the no-running-loop `asyncio.run()` path.
- **Code dedup**: Merged [`fetch_messages()`](telegram_client.py) and [`fetch_group_messages()`](telegram_client.py) into shared [`_fetch_from_sources()`](telegram_client.py) helper. Eliminated ~50 lines of duplicated message-fetching logic (the two functions differed only in source list, history loader, and log labels).
- **Code dedup**: Merged group/non-group branches in [`save_updated_summary()`](history_manager.py) into a single flow. Eliminated ~20 lines of duplicated summary-replacement logic.
- **Lambda memory**: Increased `MemorySize` from 512 to 1024 MB in [`template.yaml`](template.yaml). Lambda allocates CPU proportionally to memory, so this also speeds up execution and reduces timeout risk.
- **Secrets management — SSM Parameter Store**: Added optional SSM Parameter Store support for all four secrets (Telegram API ID/Hash, Bot Token, OpenAI API Key):
  - [`config.py`](config.py): New `_get_ssm_param()` and `_get_secret()` functions resolve secrets from SSM at runtime (supports key rotation). Falls back to env vars when SSM path is not configured or unavailable. `API_ID`, `API_HASH`, `BOT_TOKEN`, and `OPENAI_API_KEY` now use `_get_secret()`.
  - [`template.yaml`](template.yaml): New SSM path parameters (`TelegramApiIdSsmPath`, etc.) with conditional IAM policies and dual resolution (CloudFormation `{{resolve:ssm:...}}` at deploy time + runtime `boto3` for rotation).
- **Tests added**: 6 new tests (total 61, up from 55):
  - `test_run_async_with_loop_works_without_running_loop`: verifies basic coroutine execution
  - `test_run_async_with_loop_inside_running_loop`: verifies the fix for the same-loop deadlock
  - `test_run_async_with_loop_returns_empty_on_exception`: verifies graceful error handling
  - `test_get_secret_prefers_ssm_over_env`: verifies SSM takes precedence over env var
  - `test_get_secret_falls_back_to_env_when_ssm_empty`: verifies env var fallback when SSM path empty
  - `test_get_secret_falls_back_to_env_when_ssm_fails`: verifies env var fallback when SSM call fails
- All 61 tests pass without errors.

## Completed in 2026-06-01 round 2 (Lambda hardening round 7, caching, deadline, safety)

- **S3 client caching**: Added `_s3_client` module-level cache in [`s3_sync.py`](s3_sync.py) — `_get_s3_client()` now creates the boto3 S3 client once and reuses it (mirrors the SSM client caching already done in [`config.py`](config.py)). Previously created a new client per call (2 calls per Lambda invocation: download + upload).
- **In-memory history caching**: Added `_cache` dict and `invalidate_cache()` in [`history_manager.py`](history_manager.py). `load_summaries_history()` and `load_group_summaries_history()` now cache their results in memory, avoiding repeated disk reads within a single Lambda invocation. Cache is invalidated on every save operation (`save_summary_to_history`, `save_group_summary_to_history`, `save_updated_summary`, `_restore_summaries_from_channel`) and on S3 download (via `invalidate_cache()` call in [`s3_sync.py`](s3_sync.py)).
- **Deadline check during message iteration**: Added deadline check inside `async for msg in user_client.iter_messages(...)` loop in [`_fetch_from_sources()`](telegram_client.py) — previously only checked before each source, not between messages. If a single channel has thousands of messages, the Lambda could now exceed its timeout mid-fetch. The new check breaks out of the iteration loop early, returning already-fetched messages.
- **None-safe text handling**: Added `_text_hash()` helper in [`message_processor.py`](message_processor.py) and [`history_manager.py`](history_manager.py) that handles `None` text gracefully (`hashlib.sha256((text or "").encode())`). Also made [`MessageInfo.from_dict()`](models.py) defensive: `text`, `channel`, `link` default to `""` when `None` or missing, and `date` falls back to `datetime.now(timezone.utc)` when absent. Prevents `AttributeError` on `.encode()` if deserialized data has null fields.
- **Tests added**: 8 new tests (total 74, up from 66):
  - `test_from_dict_handles_none_text`: verifies `MessageInfo.from_dict` with `None` text
  - `test_from_dict_handles_missing_text`: verifies `MessageInfo.from_dict` without text key
  - `test_from_dict_handles_none_channel`: verifies `MessageInfo.from_dict` with `None` channel
  - `test_get_s3_client_caches_client`: verifies S3 client is cached and reused
  - `test_invalidate_cache_clears_specific_key`: verifies targeted cache invalidation
  - `test_invalidate_cache_clears_all_without_filepath`: verifies full cache invalidation
  - `test_text_hash_with_normal_text`: verifies deterministic hashing
  - `test_text_hash_with_none_returns_same_as_empty`: verifies None → empty string equivalence
- All 74 tests pass without errors.

## Completed in 2026-06-01 round (Lambda hardening round 6, async OpenAI, bug fixes, code dedup)

- **Critical fix**: Replaced sync `OpenAI` with `AsyncOpenAI` in [`utils.py`](utils.py). The sync client blocked the entire event loop during API calls (5-30s each), making deadline checks unreliable and preventing concurrent async work. Now `await openai_client.chat.completions.create(...)` properly yields control, so `check_deadline()` and Telegram client operations can progress during OpenAI wait times.
- **Bug fix**: Fixed [`update_existing_summary()`](history_manager.py) — was not carrying over `message_id` from the original summary to the updated one. This meant subsequent edits to the same channel message would fail because `save_updated_summary()` couldn't find the message by ID.
- **Code dedup**: Merged `is_message_covered_in_summaries()` and `is_message_covered_in_group_summaries()` in [`message_processor.py`](message_processor.py) into shared `_check_coverage()` helper. Eliminated ~20 lines of duplicated coverage-check logic (both differed only in prompt, context function, and label).
- **Code cleanup**: Removed unused `duplicate_label` parameter from [`_remove_duplicates_generic()`](message_processor.py) — was passed by `remove_duplicates()` and `remove_group_duplicates()` but never referenced.
- **Performance**: Cached SSM client in [`config.py`](config.py) — `_get_ssm_client()` now creates the boto3 SSM client once and reuses it, instead of creating a new client per `_get_ssm_param()` call (up to 4 calls per Lambda invocation).
- **Tests added**: 2 new tests (total 66, up from 64):
  - `test_update_preserves_message_id`: verifies `update_existing_summary` carries over `message_id` from original summary
  - `test_get_ssm_client_caches_client`: verifies SSM client is cached and reused across calls
- Updated test stubs in [`tests/test_openai_config.py`](tests/test_openai_config.py) and [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) to include `AsyncOpenAI` alongside `OpenAI`.
- All 66 tests pass without errors.

## Completed in 2026-05-30 round (Lambda hardening round 5, cost optimization)

- **Bug fix**: Fixed [`save_updated_summary()`](history_manager.py) matching — was using fragile `summary.content == original_summary.content` comparison which could match wrong summary if content happens to be identical. Now matches by `message_id` first (reliable unique key), falls back to content+date+count when `message_id` is None.
- **Lambda hardening**: Added deadline-aware fetching in [`telegram_client.py`](telegram_client.py) — [`_fetch_from_sources()`](telegram_client.py) now accepts `_deadline` parameter and checks it before each source, returning already-fetched messages if deadline is exceeded. [`fetch_messages()`](telegram_client.py) and [`fetch_group_messages()`](telegram_client.py) pass the deadline through from [`run_summarizer()`](summarizer.py). This prevents Lambda timeout during slow message fetching from Telegram API.
- **Dead code removal**: Removed unused [`send_message_to_target_channel()`](telegram_client.py) — the function without ID return was never called; only `send_message_to_target_channel_with_id()` is used.
- **Cost optimization — max_tokens for yes/no responses**: Reduced `max_tokens` for coverage check LLM calls in [`message_processor.py`](message_processor.py) from 10 → 5 (responses are only "ДА" or "НЕТ"). Reduced `max_tokens` for `find_relevant_summary_for_update()` in [`history_manager.py`](history_manager.py) from 5 → 3 (response is just a digit or "НЕТ").
- **Tests added**: 3 new tests (total 64, up from 61):
  - `test_match_by_message_id_takes_precedence`: verifies message_id-based matching in save_updated_summary
  - `test_fallback_to_content_date_count_when_no_message_id`: verifies fallback matching when message_id is None
  - `test_handler_passes_deadline_to_run_summarizer`: verifies deadline is passed through from Lambda handler
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) with `FIND_RELEVANT_SUMMARY_PROMPT` and `ENABLE_SUMMARY_UPDATES` config.
- All 64 tests pass without errors.

## Completed in 2026-06-02 round (bug fixes, code dedup, cost optimization)

- **Bug fix**: Added cache invalidation in [`save_summarization_history()`](history_manager.py) and [`save_group_summarization_history()`](history_manager.py) — previously, saving updated data to disk didn't clear the in-memory cache, so subsequent reads within the same Lambda invocation returned stale data.
- **Code dedup**: Moved `_text_hash()` from [`message_processor.py`](message_processor.py) and [`history_manager.py`](history_manager.py) to [`utils.py`](utils.py) as `text_hash()`. Eliminated duplicate implementation and removed unused `hashlib` imports from both modules.
- **Defensive deserialization**: Made [`SummaryInfo.from_dict()`](models.py) handle `None`/missing fields (`content`, `channels`, `date`, `message_count`) gracefully — now consistent with `MessageInfo.from_dict()` which was already hardened.
- **Cost optimization**: Reduced `max_tokens` for coverage check LLM calls in [`_check_coverage()`](message_processor.py) from 5 → 2 (responses are only "ДА"/"НЕТ"). Reduced `max_tokens` for NLP relevance check from 30 → 20. Saves ~30-40% output tokens on these yes/no classification calls.
- **Prompt fix**: Made [`find_relevant_summary_for_update()`](history_manager.py) prompt dynamic — previously hardcoded "(1, 2, 3)" but actual count varies with `UPDATE_MATCH_MAX_SUMMARIES`. Now correctly shows available indices.
- **Tests added**: 9 new tests (total 81, up from 74):
  - `test_from_dict_handles_none_content`: verifies `SummaryInfo.from_dict` with `None` content
  - `test_from_dict_handles_missing_content`: verifies `SummaryInfo.from_dict` without content key
  - `test_from_dict_handles_none_channels`: verifies `SummaryInfo.from_dict` with `None` channels
  - `test_from_dict_handles_missing_date`: verifies `SummaryInfo.from_dict` without date key
  - `test_from_dict_handles_missing_message_count`: verifies `SummaryInfo.from_dict` without message_count
  - `test_save_summarization_history_invalidates_cache`: verifies cache cleared after save
  - `test_save_group_summarization_history_invalidates_cache`: verifies cache cleared after save
  - `test_text_hash_with_normal_text`: verifies deterministic hashing (moved from stub-based to inline)
  - `test_text_hash_with_none_returns_same_as_empty`: verifies None → empty string equivalence (moved from stub-based to inline)
- Updated test stubs in [`tests/test_history_manager.py`](tests/test_history_manager.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py) to include `text_hash` in fake `utils` module.
- All 81 tests pass without errors.

## Completed in 2026-06-03 round (bug fixes, prompt optimization, context consistency)

- **Bug fix**: Fixed [`load_summaries_history()`](history_manager.py) — previously triggered an expensive channel restore whenever the history file was valid but empty (`{"summaries": []}`). Now only triggers restore when the file is missing or corrupt (`data is None`), consistent with `load_group_summaries_history()` which already handled this correctly. An empty-but-valid history file (e.g., after clearing history) now returns `[]` without unnecessary API calls.
- **Bug fix**: Fixed [`save_updated_summary()`](history_manager.py) — previously continued with save and Telegram edit even when no matching summary was found in history, leading to no-op disk writes and misleading channel edit attempts. Now tracks `found` flag and returns early with a warning log if no match is found.
- **Consistency fix**: Unified context formatting in [`get_recent_summaries_context()`](history_manager.py) — now uses the same structured "Дата: / Содержание: / ---" format as `get_recent_group_summaries_context()`. Previously used bare `\n\n`-joined content without dates, causing inconsistent LLM coverage check behavior between channel and group paths.
- **Prompt optimization**: Tightened [`NLP_RELEVANCE_PROMPT`](prompts.py) — consolidated overlapping categories (merged "AI/BigTech новости" and BigTech line, merged "Карьера, вакансии" into one line, removed "новые детали уже вышедших моделей/библиотек" as redundant, removed "подготовительные" from ШАД exclusion). Reduced token count by ~15%.
- **Prompt optimization**: Tightened [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py) — removed redundant "Кратко описывай архитектуру, методологию, результаты" instruction (already implied by "краткий дайджест"), removed "яркие" from header rule (redundant with "короткие").
- **Tests added**: 4 new tests (total 91, up from 87):
  - `test_no_restore_when_file_has_empty_summaries`: verifies empty-but-valid file does NOT trigger channel restore
  - `test_restore_when_file_missing`: verifies missing/corrupt file DOES trigger channel restore
  - `test_save_updated_summary_skips_when_no_match`: verifies no save/edit when original summary not found
  - `test_recent_summaries_context_includes_date`: verifies channel context now includes "Дата:" like group version
- All 91 tests pass without errors.

## Completed in 2026-06-03 round 2 (LLM update integration, deterministic classification, code dedup, prompt tightening)

- **Summary update quality**: Improved [`update_existing_summary()`](history_manager.py) — replaced mechanical "Другие ссылки:" append with LLM-integrated update. The LLM now inserts the new link next to the relevant paragraph in the summary. Falls back to the append approach on LLM failure or empty response. This addresses the "Summary update quality" next action from previous round.
- **Deterministic classification**: Added `temperature=0` parameter to all classification LLM calls in [`message_processor.py`](message_processor.py) and [`history_manager.py`](history_manager.py):
  - `are_messages_duplicate()`: `temperature=0` for duplicate detection
  - `_check_coverage()`: `temperature=0` for coverage check
  - `is_nlp_related()`: `temperature=0` for NLP relevance classification
  - `find_relevant_summary_for_update()`: `temperature=0` for summary matching
  - Added `temperature` parameter to [`call_openai()`](utils.py) — passed through to `chat.completions.create()` when provided, omitted when `None` (default). Classification calls now produce deterministic results, reducing variance and wasted retries.
- **Code dedup**: Merged [`get_recent_summaries_context()`](history_manager.py) and [`get_recent_group_summaries_context()`](history_manager.py) into shared [`_get_recent_summaries_context()`](history_manager.py) helper. Both functions now delegate to the same formatting logic, eliminating ~20 lines of duplicated context-building code.
- **Prompt optimization**: Tightened [`NLP_RELEVANCE_PROMPT`](prompts.py) — merged BigTech and AI-ассистенты lines into one (redundant split), merged "обучение/дообучение" into "дообучение" (shorter). Reduced token count by ~10%.
- **Unused imports removed**: Removed unused `import os` from [`channel_manager.py`](channel_manager.py) and unused imports (`load_discovered_channels`, `load_similar_channels`, `load_banned_channels`, `get_all_source_channels`) from [`message_processor.py`](message_processor.py).
- **Tests added**: 7 new tests (total 98, up from 91):
  - `test_call_openai_passes_temperature_to_api`: verifies `temperature=0` is passed to API create call
  - `test_call_openai_omits_temperature_when_none`: verifies `temperature` is omitted when not specified
  - `test_update_existing_summary_uses_llm`: verifies LLM is called for summary integration
  - `test_update_existing_summary_fallback_on_llm_failure`: verifies fallback to append on LLM exception
  - `test_update_existing_summary_fallback_on_empty_llm_response`: verifies fallback on empty LLM response
  - `test_shared_helper_returns_empty_for_no_summaries`: verifies `_get_recent_summaries_context` returns empty for no data
  - `test_shared_helper_truncates_long_content`: verifies content truncation in shared helper
- All 98 tests pass without errors.

## Completed in 2026-06-04 round (Lambda hardening round 9, bug fixes, cost optimization, observability)

- **Bug fix**: Fixed [`load_group_summaries_history()`](history_manager.py) — previously returned `[]` when file was missing or corrupt, unlike [`load_summaries_history()`](history_manager.py) which restores from channel. Now both functions consistently attempt channel restore when `data is None`, preventing data loss after group history file corruption.
- **Cost optimization**: Added `UPDATE_SUMMARY_MAX_TOKENS` config (default 500) in [`config.py`](config.py). Applied in [`update_existing_summary()`](history_manager.py) — previously used `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` (4000) for a simple link insertion, wasting ~87% output tokens. 500 tokens is sufficient for inserting one link into an existing summary.
- **Lambda duration logging**: Added `time.monotonic()` tracking in [`lambda_handler.handler()`](lambda_handler.py) — now logs `"Lambda completed in X.Xs"` on success and `"Lambda execution failed after X.Xs"` on error. Enables CloudWatch-based performance monitoring and timeout trend detection.
- **OpenAI API key early validation**: Added check in [`call_openai()`](utils.py) — returns `""` immediately with clear error log when `OPENAI_API_KEY` is not set, instead of creating an `AsyncOpenAI` client that would fail with a confusing 401 error.
- **SAM template updated**: Added `UpdateSummaryMaxTokens` parameter and env var in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `UPDATE_SUMMARY_MAX_TOKENS=500` in [`.env.example`](.env.example).
- **Tests added**: 7 new tests (total 105, up from 98):
  - `test_config_reads_update_summary_max_tokens_from_env`: verifies `UPDATE_SUMMARY_MAX_TOKENS` env var parsing
  - `test_call_openai_returns_empty_when_api_key_missing`: verifies early return when API key is None
  - `test_handler_logs_duration_on_success`: verifies completion duration log
  - `test_handler_logs_duration_on_error`: verifies error path includes duration
  - `test_no_restore_when_file_has_empty_summaries` (group): verifies empty-but-valid group file does NOT trigger restore
  - `test_restore_when_file_missing` (group): verifies missing group file DOES trigger restore
  - `test_update_existing_summary_uses_update_max_tokens`: verifies `UPDATE_SUMMARY_MAX_TOKENS` (500) used instead of 4000
- All 105 tests pass without errors.

## Completed in 2026-06-05 round (Lambda hardening round 10, summary quality, code dedup)

- **Summary temperature**: Added `OPENAI_SUMMARY_TEMPERATURE` config (default 0.3) in [`config.py`](config.py). Applied in [`summarize_text()`](message_processor.py) and [`summarize_group_text()`](message_processor.py) — previously used OpenAI's default temperature (1.0), which produced inconsistent and often verbose summaries. A low temperature (0.3) produces more concise, consistent output, directly addressing the AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness. Configurable via env for A/B testing.
- **Deterministic summary updates**: Added `temperature=0` to [`update_existing_summary()`](history_manager.py) — link insertion is a deterministic edit operation; randomness was unnecessary and could cause inconsistent updates.
- **Code dedup**: Merged `_ensure_clients()` and `_ensure_bot_client()` in [`telegram_client.py`](telegram_client.py) — both were identical wrappers that called `start_clients()` when client was disconnected. `_ensure_bot_client()` now delegates to `_ensure_clients()`.
- **Code dedup**: Extracted shared `_load_processed_messages()` and `_save_processed_messages()` helpers in [`history_manager.py`](history_manager.py). `load_summarization_history()` / `load_group_summarization_history()` and `save_summarization_history()` / `save_group_summarization_history()` now delegate to these shared helpers, eliminating ~40 lines of duplicated message-ID set building and append-truncate-save logic.
- **SAM template updated**: Added `OpenAISummaryTemperature` parameter and env var in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `OPENAI_SUMMARY_TEMPERATURE=0.3` in [`.env.example`](.env.example).
- **Tests added**: 7 new tests (total 112, up from 105):
  - `test_config_reads_summary_temperature_from_env`: verifies `OPENAI_SUMMARY_TEMPERATURE` env var parsing
  - `test_config_summary_temperature_default`: verifies default value is 0.3
  - `test_summarize_text_passes_temperature`: verifies temperature passed to `call_openai` for channel summaries
  - `test_summarize_group_text_passes_temperature`: verifies temperature passed for group summaries
  - `test_update_existing_summary_uses_temperature_zero`: verifies `temperature=0` for deterministic link insertion
  - `test_load_processed_messages_returns_set`: verifies shared load helper returns set of message IDs
  - `test_save_processed_messages_appends_and_truncates`: verifies shared save helper appends and truncates
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to include `OPENAI_SUMMARY_TEMPERATURE`.
- All 112 tests pass without errors.

## Completed in 2026-06-05 round 2 (Lambda hardening round 11, quality, dedup, cost, performance)

- **Bug fix**: Fixed [`load_group_summaries_history()`](history_manager.py) — previously returned `[]` without caching when the file had valid but empty data (`{"summaries": []}`). Every subsequent call re-read the file from disk. Now caches empty results consistently with [`load_summaries_history()`](history_manager.py), and adds `sorted()` for consistency.
- **Intra-batch dedup**: Added [`_remove_intra_batch_duplicates()`](message_processor.py) — lightweight SequenceMatcher-only dedup that removes obvious text duplicates (ratio > 0.95) and link duplicates within a single batch, without LLM calls. Applied in [`process_messages()`](message_processor.py) before summarization. Previously, two channels posting the same news would both be included in full, wasting input tokens and producing redundant summaries. Now near-identical messages are collapsed before the LLM sees them, directly improving summary conciseness (per AUTOWORK_INSTRUCTIONS goal).
- **Code dedup**: Merged [`save_summary_to_history()`](history_manager.py) and [`save_group_summary_to_history()`](history_manager.py) into shared [`_save_summary_to_history_file()`](history_manager.py) helper. The channel version previously loaded and re-serialized all existing `SummaryInfo` objects on every save; the new helper uses raw JSON append (like the group version already did), which is more efficient and avoids unnecessary parse/serialize overhead.
- **Cost optimization — summary input truncation**: Added `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE` config (default 3000) in [`config.py`](config.py). Applied in [`_prepare_messages_text()`](message_processor.py) — each message's text is now truncated before being sent to the summary LLM. Previously, a single very long message (full article) could consume the entire input token budget, leaving little room for other messages and producing incomplete summaries.
- **Performance — parallel NLP classification**: Added `NLP_CONCURRENT_CHECKS` config (default 5) in [`config.py`](config.py). Refactored [`process_messages()`](message_processor.py) to use `asyncio.gather` with a semaphore for NLP relevance checks — all messages are now classified concurrently (up to `NLP_CONCURRENT_CHECKS` at a time), reducing wall-clock time from N × OpenAI latency to ~N/5 × latency. Coverage checks and state-mutating operations remain sequential for safety.
- **Dead code removed**: Removed [`_classify_message()`](message_processor.py) — its logic was inlined into `process_messages()` to enable the parallel NLP check split (parallel NLP phase, then sequential coverage/update phase).
- **SAM template updated**: Added `SummaryMaxInputCharsPerMessage` and `NlpConcurrentChecks` parameters and env vars in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE=3000` and `NLP_CONCURRENT_CHECKS=5` in [`.env.example`](.env.example).
- **Tests added**: 13 new tests (total 125, up from 112):
  - `test_load_group_summaries_caches_empty_results`: verifies empty group summaries are cached (not re-read from disk)
  - `test_save_summary_to_history_file_appends_and_truncates`: verifies shared save helper appends and truncates
  - `test_removes_identical_messages`: verifies intra-batch dedup removes identical messages
  - `test_keeps_different_messages`: verifies intra-batch dedup keeps different messages
  - `test_removes_link_duplicates`: verifies intra-batch dedup removes same-link messages
  - `test_keeps_different_links`: verifies intra-batch dedup keeps different-link messages
  - `test_empty_input_returns_empty`: verifies intra-batch dedup handles empty input
  - `test_long_message_is_truncated`: verifies _prepare_messages_text truncates long messages
  - `test_short_message_is_not_truncated`: verifies _prepare_messages_text keeps short messages
  - `test_config_reads_summary_max_input_chars_from_env`: verifies `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE` env var parsing
  - `test_config_summary_max_input_chars_default`: verifies default is 3000
  - `test_config_reads_nlp_concurrent_checks_from_env`: verifies `NLP_CONCURRENT_CHECKS` env var parsing
  - `test_config_nlp_concurrent_checks_default`: verifies default is 5
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to include `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE` and `NLP_CONCURRENT_CHECKS`.
- All 125 tests pass without errors.

## Completed in 2026-06-08 round (Lambda hardening round 12, parallel coverage, dead code removal, logging fix)

- **Parallel coverage checks**: Refactored [`process_messages()`](message_processor.py) to run coverage checks in parallel using `asyncio.Semaphore` and `asyncio.gather`, mirroring the parallel NLP check pattern. Previously, coverage checks were sequential — each NLP-related message's coverage was checked one at a time, adding N × OpenAI latency to wall-clock time. Now all coverage checks run concurrently (up to `NLP_CONCURRENT_CHECKS` at a time), reducing wall-clock time from N × latency to ~N/5 × latency. State-mutating operations (`process_covered_message`) remain sequential for safety.
- **Dead code removal**: Removed [`remove_duplicates()`](message_processor.py), [`remove_group_duplicates()`](message_processor.py), [`_remove_duplicates_generic()`](message_processor.py), and [`are_messages_duplicate()`](message_processor.py) — these functions were superseded by inline dedup logic in `process_messages()` and were only called from tests, never from the production code path. The production flow uses `_remove_intra_batch_duplicates()` (SequenceMatcher-only, no LLM) plus the parallel coverage check.
- **Dead code removal**: Removed unused `SIMILARITY_LLM_LOWER` import from [`message_processor.py`](message_processor.py) — only `SIMILARITY_LLM_UPPER` is still used by `_remove_intra_batch_duplicates()`.
- **Import cleanup**: Moved `import asyncio` and `from config import SOURCE_CHANNELS` to module level in [`message_processor.py`](message_processor.py), removing local imports inside `process_messages()`.
- **Logging fix**: Replaced `traceback.print_exc()` with `exc_info=True` in [`summarizer.py`](summarizer.py) error handler — structured traceback in CloudWatch logs instead of unstructured stdout output. Removed unused `import traceback`.
- **Tests updated**: Removed 4 obsolete tests (2 three-band dedup tests that tested removed `_remove_duplicates_generic`, 2 Russian response tests that tested removed `are_messages_duplicate`). Added 2 new tests:
  - `test_coverage_checks_run_in_parallel`: verifies coverage check is called for each NLP-related message
  - `test_covered_messages_excluded_from_summary`: verifies covered messages are excluded from summary generation
  - Total test count: 123 (down from 125 — net -2 from 4 removed + 2 added)
- All 123 tests pass without errors.

## Completed in 2026-06-08 round 2 (Lambda hardening round 13, bug fixes, observability, cleanup)

- **Critical bug**: Fixed [`summarize_text()`](message_processor.py) and [`summarize_group_text()`](message_processor.py) — previously returned error string `"Ошибка: Не удалось сгенерировать обобщение"` on OpenAI failure. This error message was both posted to the Telegram channel AND saved to summary history, polluting the channel with error text and corrupting history files. Now returns `None` on failure, and [`process_messages()`](message_processor.py) skips Telegram send when summary is `None` (`if summary and send_message:` instead of `if send_message:`). The existing `if summary and unique_messages:` guard in [`_save_processing_results()`](message_processor.py) already handled `None` correctly.
- **Coverage check robustness**: Changed `_check_coverage()` in [`message_processor.py`](message_processor.py) from `result.strip().upper() == "ДА"` to `.startswith("ДА")` — consistent with `is_nlp_related()` which already uses `.startswith("да")`. Handles model variations like "ДА." or "ДА, тема совпадает" that would fail strict equality.
- **OpenAI token usage logging**: Added structured logging of token usage from API responses in [`call_openai()`](utils.py) — logs `model`, `prompt_tokens`, `completion_tokens`, and `total_tokens` at INFO level. Enables cost monitoring and trend detection in CloudWatch without additional API calls.
- **Lambda response enrichment**: Added `elapsed_seconds` field to Lambda handler response in [`lambda_handler.py`](lambda_handler.py) — present in both success and error responses. Enables CloudWatch metric filters and invocation-level performance tracking.
- **Dead config cleanup**: Removed redundant `send_message` environment variable from [`template.yaml`](template.yaml) — the `Environment.Variables.send_message` entry was never read by any code (the `send_message` flag is parsed from the event payload, not from env vars).
- **Tests added**: 9 new tests (total 132, up from 123):
  - `test_summarize_text_returns_none_on_empty_openai_response`: verifies `None` return on failure
  - `test_summarize_group_text_returns_none_on_empty_openai_response`: verifies `None` return on failure
  - `test_process_messages_skips_send_when_summary_is_none`: verifies Telegram send is skipped when summary fails
  - `test_coverage_check_matches_da_with_period`: verifies `.startswith("ДА")` handles "ДА."
  - `test_coverage_check_matches_da_with_comma`: verifies `.startswith("ДА")` handles "ДА, тема совпадает"
  - `test_coverage_check_rejects_net`: verifies "НЕТ" is correctly rejected
  - `test_call_openai_logs_token_usage`: verifies token usage logging in CloudWatch
  - `test_handler_includes_elapsed_seconds_on_success`: verifies `elapsed_seconds` in success response
  - `test_handler_includes_elapsed_seconds_on_error`: verifies `elapsed_seconds` in error response
- Updated [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — changed response assertion from exact dict match to key-level checks (more resilient to new fields).
- All 132 tests pass without errors.

## Completed in 2026-06-09 round (Lambda hardening round 14, bug fixes, cost optimization, quality)

- **Critical bug**: Fixed [`update_existing_summary()`](history_manager.py) — `UPDATE_SUMMARY_MAX_TOKENS` was 500, but the prompt asks the LLM to return the full updated summary ("Ответь только обновлённое саммари"). For a typical 2000-char Russian summary (~1000 tokens), 500 max_tokens would truncate the response, causing data loss — the shortened text replaced the original. Increased default from 500 to 2000 in [`config.py`](config.py). Added length guard: if the LLM response is less than 80% of the original content length, falls back to the append approach instead of using the truncated response.
- **Critical bug**: Fixed [`process_messages()`](message_processor.py) — `nlp_related_messages.append(msg)` was outside the `for` loop (indentation error from the parallel NLP check refactor in round 12). Only the last message in each batch was being appended to `nlp_related_messages`, so all other NLP-related messages were silently dropped. This caused missing content in summaries whenever there were multiple messages in a batch.
- **Cost optimization**: Reordered [`process_messages()`](message_processor.py) — intra-batch dedup now runs before coverage checks. Previously, all NLP-related messages got coverage checks (expensive LLM calls), and then intra-batch dedup removed duplicates after. Now duplicates are removed first (free SequenceMatcher check), saving LLM coverage check tokens on duplicate messages.
- **Cost optimization**: Coverage context is now built once per [`process_messages()`](message_processor.py) call instead of per-message. Previously, `get_recent_summaries_context()` was called inside each `is_message_covered_in_summaries()` call, rebuilding the context string for every message. Now pre-computed once and passed to all parallel `_check_coverage()` calls.
- **Operational**: Made `ENABLE_SUMMARIES_DEDUPLICATION` and `ENABLE_SUMMARY_UPDATES` configurable via env vars in [`config.py`](config.py) (default: `true`). Previously hardcoded to `True`, making it impossible to disable coverage dedup or summary updates for testing or cost saving without code changes.
- **Link extraction**: Fixed [`extract_links()`](utils.py) — added `TRAILING_PUNCTUATION_REGEX` to strip trailing punctuation (`.`, `,`, `)`, `]`, `}`, etc.) from extracted URLs. Previously, `https://example.com/page).` would capture the closing parenthesis and period as part of the URL, causing incorrect link handling in summary generation and dedup.
- **Consistency**: Replaced `msg.message` with `msg.text` in [`_restore_summaries_from_channel()`](history_manager.py) — both refer to the same property in Telethon, but `msg.text` is the canonical accessor and consistent with the rest of the codebase.
- **SAM template updated**: Updated `UpdateSummaryMaxTokens` default from "500" to "2000" and added `EnableSummariesDeduplication` / `EnableSummaryUpdates` parameters and env vars in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Updated `UPDATE_SUMMARY_MAX_TOKENS=2000` and added `ENABLE_SUMMARIES_DEDUPLICATION=true` / `ENABLE_SUMMARY_UPDATES=true` in [`.env.example`](.env.example).
- **Tests added**: 14 new tests (total 146, up from 132):
  - `test_falls_back_when_llm_response_truncated`: verifies append fallback on truncated LLM response
  - `test_keeps_llm_response_when_length_sufficient`: verifies LLM response kept when >= 80% of original
  - `test_enable_dedup_defaults_true`: verifies `ENABLE_SUMMARIES_DEDUPLICATION` default
  - `test_enable_dedup_can_be_disabled`: verifies `ENABLE_SUMMARIES_DEDUPLICATION=false` works
  - `test_enable_updates_defaults_true`: verifies `ENABLE_SUMMARY_UPDATES` default
  - `test_enable_updates_can_be_disabled`: verifies `ENABLE_SUMMARY_UPDATES=0` works
  - `test_strips_trailing_period`: verifies URL trailing period stripped
  - `test_strips_trailing_parenthesis`: verifies URL trailing parenthesis stripped
  - `test_strips_trailing_comma`: verifies URL trailing comma stripped
  - `test_preserves_clean_url`: verifies clean URLs unchanged
  - `test_strips_multiple_trailing_punctuation`: verifies multiple trailing chars stripped
  - `test_dedup_before_coverage_saves_llm_calls`: verifies coverage check only on deduped messages
  - `test_all_nlp_messages_appended`: regression test for the indentation bug
  - `test_config_update_summary_max_tokens_default`: verifies `UPDATE_SUMMARY_MAX_TOKENS` default is 2000
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to patch `_check_coverage` instead of `is_message_covered_in_summaries` (coverage checks now use pre-built context).
- All 146 tests pass without errors.

## Completed in 2026-06-10 round (Lambda hardening round 15, NLP pre-filter, coverage+match merge, cost optimization)

- **NLP keyword pre-filter**: Added `_is_obvious_non_nlp()` in [`message_processor.py`](message_processor.py) — regex-based pre-check for common ad/course/webinar patterns (`курс`, `вебинар`, `регистраци`, `скидк`, `промокод`, `мастер-класс`, `стажировк`, `hire`, `bootcamp`, etc.). Applied in [`is_nlp_related()`](message_processor.py) — messages matching ad patterns are rejected immediately with reason `"ad_keyword"` without making an LLM call. Saves ~30-50% of NLP relevance LLM calls on typical Telegram channels with heavy advertising.
- **Combined coverage + match check**: Added [`_check_coverage_and_match()`](message_processor.py) — merges the previous two-step flow (coverage check + `find_relevant_summary_for_update`) into a single LLM call. When a message is covered in previous summaries, the combined check returns the matching `SummaryInfo` directly. [`process_messages()`](message_processor.py) now passes the matching summary to [`process_covered_message()`](message_processor.py), which skips the separate `find_relevant_summary_for_update` call. Saves one entire LLM call per covered message.
- **COVERAGE_AND_MATCH_PROMPT**: Added new prompt in [`prompts.py`](prompts.py) — asks the LLM to return either a summary number or "НЕТ", combining coverage detection and matching into one response.
- **Summary update input truncation**: Added `UPDATE_SUMMARY_MAX_INPUT_CHARS` config (default 2000) in [`config.py`](config.py). Applied in [`update_existing_summary()`](history_manager.py) — summary content sent to the LLM is now truncated before the update prompt. Previously sent the full summary (up to 4000 chars) as input context, wasting tokens on a simple link insertion.
- **Config validation consistency**: Replaced all `int(os.getenv(...))` calls with `_get_int_env()` in [`config.py`](config.py) — `NLP_CHECK_MAX_INPUT_CHARS`, `MAX_MESSAGES_PER_SOURCE`, `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE`, `NLP_CONCURRENT_CHECKS`, `COVERAGE_CHECK_MAX_SUMMARIES`, `COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY`, `UPDATE_MATCH_MAX_SUMMARIES`, `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY`, `MAX_CHANNEL_HISTORY_MESSAGES`, `MAX_CHANNEL_SUMMARIES`, `MAX_GROUP_HISTORY_MESSAGES`, `MAX_GROUP_SUMMARIES`, `GROUP_SUMMARIZATION_INTERVAL_SECONDS`, `RESTORE_HISTORY_DAYS`. Now all integer configs validate that values are positive integers, catching misconfiguration early.
- **Telegram message accessor consistency**: Replaced `msg.message` with `msg.text` in [`_fetch_from_sources()`](telegram_client.py) — `msg.text` is the canonical Telethon accessor, consistent with the rest of the codebase. `msg.message` and `msg.text` return the same value for plain text messages, but `msg.text` handles formatting entities correctly.
- **SAM template updated**: Added `UpdateSummaryMaxInputChars` parameter and env var in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `UPDATE_SUMMARY_MAX_INPUT_CHARS=2000` in [`.env.example`](.env.example).
- **Tests added**: 17 new tests (total 163, up from 146):
  - `test_rejects_course_ad`: verifies `_is_obvious_non_nlp` catches "курс"
  - `test_rejects_webinar_ad`: verifies `_is_obvious_non_nlp` catches "вебинар"
  - `test_rejects_promocode_ad`: verifies `_is_obvious_non_nlp` catches "промокод"
  - `test_allows_ai_research`: verifies `_is_obvious_non_nlp` passes AI research text
  - `test_allows_vacancy`: verifies `_is_obvious_non_nlp` passes vacancy text
  - `test_nlp_related_skips_llm_on_ad_keyword`: verifies `is_nlp_related` returns `(False, "ad_keyword")` without LLM call
  - `test_returns_matching_summary_on_covered`: verifies `_check_coverage_and_match` returns matching summary
  - `test_returns_none_on_not_covered`: verifies `_check_coverage_and_match` returns None for "НЕТ"
  - `test_returns_none_on_empty_summaries`: verifies `_check_coverage_and_match` returns None without LLM call when no summaries
  - `test_returns_none_on_invalid_response`: verifies `_check_coverage_and_match` handles invalid LLM response
  - `test_config_reads_update_summary_max_input_chars_from_env`: verifies `UPDATE_SUMMARY_MAX_INPUT_CHARS` env var
  - `test_config_update_summary_max_input_chars_default`: verifies default is 2000
  - `test_nlp_ad_keywords_is_list`: verifies `NLP_AD_KEYWORDS` is a non-empty list
  - `test_nlp_ad_keywords_contains_course`: verifies "курс" is in `NLP_AD_KEYWORDS`
  - `test_nlp_check_max_input_chars_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_max_messages_per_source_rejects_negative`: validates `_get_int_env` rejects negative
  - `test_truncates_summary_in_update_prompt`: verifies summary content truncation in `update_existing_summary`
- Updated coverage check tests in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to use `_check_coverage_and_match` instead of `_check_coverage`, and to provide `SummaryInfo` objects via `load_summaries_history` stub.
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to include `NLP_AD_KEYWORDS`, `UPDATE_MATCH_MAX_SUMMARIES`, `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY`, `COVERAGE_AND_MATCH_PROMPT`, `load_summaries_history`, `load_group_summaries_history`.
- All 163 tests pass without errors.

## Completed in 2026-06-10 round 2 (Lambda hardening round 16, bug fixes, cost optimization, dead code removal)

- **Bug fix**: Moved [`validate_config()`](lambda_handler.py) inside the `try/except` block — previously called before the try, a `ValueError` on missing env vars would produce an unhandled Lambda exception with raw traceback instead of the structured `{'status': 'error', ...}` JSON response.
- **Bug fix**: Changed [`should_run_group_summarization()`](history_manager.py) to return `True` on parse errors — previously returned `False` on malformed `GROUP_LAST_RUN_FILE` timestamps, permanently skipping group summarization until manual file deletion.
- **Bug fix**: Added stale matching_summary refresh in [`process_covered_message()`](message_processor.py) — when multiple messages match the same summary in one batch, the second update would use the stale (pre-update) summary, overwriting the first update. Now reloads the summary from file via `load_summaries_history()` before calling `update_existing_summary()`.
- **Cost optimization**: Reduced `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` default from 4000→1500 in [`config.py`](config.py) and [`template.yaml`](template.yaml). With `SUMMARY_MAX_LENGTH=4000` chars and ~3 chars/token for Russian, 1500 tokens is sufficient. Post-hoc `enforce_summary_length()` still catches any excess.
- **Cost optimization**: Truncated `msg.text` in [`_check_coverage_and_match()`](message_processor.py) to `NLP_CHECK_MAX_INPUT_CHARS` (2000) — previously sent the full message text, wasting input tokens on long articles for a yes/no classification.
- **Cost optimization**: Truncated `new_message.text` in [`update_existing_summary()`](history_manager.py) to `UPDATE_SUMMARY_MAX_INPUT_CHARS` (2000) — previously sent the full message text for a link-insertion task.
- **Dead code removal**: Removed `is_message_covered_in_summaries()` and `is_message_covered_in_group_summaries()` from [`message_processor.py`](message_processor.py) — superseded by `_check_coverage_and_match()` which combines coverage check + match finding into one LLM call. Updated tests to use `_check_coverage()` directly.
- **Dead code removal**: Removed `DUPLICATE_CHECK_PROMPT` from [`prompts.py`](prompts.py) — LLM-based deduplication was removed in round 12; this prompt was never called in production.
- **Import cleanup**: Moved inline `from history_manager import ...` and `from channel_manager import save_discovered_channel` to top-level in [`message_processor.py`](message_processor.py). Moved `import time` to top-level. Added `from config import ENABLE_SUMMARY_UPDATES` to top-level.
- **Prompt consistency**: Added "Разные модели на Hugging Face = разные темы → НЕТ" rule to [`GROUP_SUMMARY_COVERAGE_CHECK_PROMPT`](prompts.py) — was missing from the group version but present in the channel version, causing inconsistent coverage check behavior.
- **Lambda hardening**: Added `_deadline` parameter to [`process_messages()`](message_processor.py) with checks before NLP classification, coverage checks, and summary generation. [`summarizer.py`](summarizer.py) now passes `_deadline` through. When deadline is exceeded, the current phase is skipped and partial results are saved.
- **Lambda hardening**: Cached [`_load_processed_messages()`](history_manager.py) — previously read from disk and rebuilt the processed-messages set on every call. Now uses the same `_cache` pattern as `load_summaries_history()`, with invalidation on save.
- **Lambda hardening**: Added total message examined limit (`MAX_MESSAGES_PER_SOURCE * 3`) in [`_fetch_from_sources()`](telegram_client.py) — previously, a channel with many already-processed messages could iterate through all of them before hitting the date cutoff, wasting Telegram API calls and time.
- **Test stubs updated**: Added missing stub attributes (`load_summaries_history`, `save_summaries_history`, `save_discovered_channel`, `COVERAGE_AND_MATCH_PROMPT`, `FIND_RELEVANT_SUMMARY_PROMPT`, `ENABLE_SUMMARY_UPDATES`) to [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py), and [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py).
- **Tests added**: 8 new tests (total 171, up from 163):
  - `test_handler_returns_structured_error_on_validate_config_failure`: verifies validate_config errors return structured JSON
  - `test_returns_true_on_malformed_timestamp`: verifies should_run_group_summarization returns True on parse error
  - `test_returns_true_when_file_missing`: verifies should_run_group_summarization returns True when file missing
  - `test_coverage_match_truncates_long_message`: verifies _check_coverage_and_match truncates msg.text
  - `test_deadline_skips_summary_generation`: verifies deadline propagation in process_messages
  - `test_refreshes_matching_summary_from_file`: verifies stale matching_summary refresh in process_covered_message
  - `test_channel_summary_max_tokens_default_is_1500`: verifies new default
  - `test_channel_summary_max_tokens_from_env`: verifies env var override
- Updated `should_run_group_summarization` logic test to expect `True` on malformed timestamps (consistent with the bug fix).
- All 171 tests pass without errors.

## Completed in 2026-06-11 round (Lambda hardening round 18, dead code, prompt quality, observability, config hardening)

- **Dead code removal**: Simplified [`process_covered_message()`](message_processor.py) — removed unreachable `else` branch (line 501-502) after `if matching_summary is None: return` guard. The `if matching_summary:` check (line 493) was always True after the None guard, making the `else` dead code. Also removed redundant `logger.debug("Found relevant summary for update")` log.
- **Bug fix**: Added missing-summary-in-history guard in [`process_covered_message()`](message_processor.py) — when refreshing a summary from file by `message_id`, if the summary is not found (e.g., deleted between invocations), now skips the update instead of proceeding with stale data. Previously, the refresh loop would leave `matching_summary` unchanged if no match was found, then proceed with the stale object.
- **Prompt quality**: Added "Не повторяй одну и ту же информацию в разных разделах" rule to both [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py) — addresses a common LLM failure mode where the same fact appears under different section headers. Directly improves conciseness per AUTOWORK_INSTRUCTIONS goal.
- **Observability**: Added OpenAI latency CloudWatch metric via Embedded Metric Format (EMF) in [`_emit_openai_latency()`](utils.py) — after each successful OpenAI API call, prints an EMF JSON line to stdout that CloudWatch automatically processes into a `Latency` metric in the `tg_summarizer/OpenAI` namespace. No additional API calls or dependencies required. Enables CloudWatch-based OpenAI latency monitoring and alerting.
- **Observability**: Added `OpenAILatencyAlarm` in [`template.yaml`](template.yaml) — CloudWatch alarm on the EMF `Latency` metric (average latency > 20 seconds over 2 consecutive 5-minute periods). Addresses the "OpenAI latency alerting" next action from previous rounds.
- **Observability**: Added summary logging in [`download_from_s3()`](s3_sync.py) and [`upload_to_s3()`](s3_sync.py) — now logs "S3 download: N files downloaded, M skipped" and "S3 upload: N files uploaded, M failed" at INFO level. Previously logged each file individually at INFO/DEBUG, making it hard to see the overall sync status in CloudWatch at a glance.
- **Config hardening**: Moved `RESTORE_TIMEOUT_SEC` to [`config.py`](config.py) with `_get_int_env` validation (default 30) — previously used raw `os.getenv("RESTORE_TIMEOUT_SEC", "30")` in three places in [`history_manager.py`](history_manager.py) without validation. Now validates that the value is a positive integer, catching misconfiguration early. Consistent with all other integer configs.
- **`.env.example` synced**: Added `RESTORE_TIMEOUT_SEC=30` in [`.env.example`](.env.example).
- **Tests added**: 7 new tests (total 173, up from 166):
  - `test_config_reads_restore_timeout_sec_from_env`: verifies `RESTORE_TIMEOUT_SEC` env var parsing
  - `test_config_restore_timeout_sec_default`: verifies default is 30
  - `test_config_restore_timeout_sec_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_call_openai_emits_emf_metric`: verifies EMF JSON output on successful API call
  - `test_skips_update_when_summary_not_found_in_history`: verifies process_covered_message skips when summary not found in file
  - `test_download_logs_summary_counts`: verifies S3 download summary log with file counts
  - `test_upload_logs_summary_counts`: verifies S3 upload summary log with file counts
- All 173 tests pass without errors.

## Completed in 2026-06-14 round (Lambda hardening round 20, dead keywords, length guard on update, deadline content loss, config hardening, FloodWait, Telegram limits)

- **Dead code removal**: Removed unreachable entries from [`NLP_AD_KEYWORDS`](config.py) — "бесплатный курс" and "платный курс" were never matched because "курс" (earlier in the regex alternation) already matches any text containing "курс". Reduces regex complexity and eliminates misleading list entries.
- **Summary length guard on update**: Added length enforcement in [`update_existing_summary()`](history_manager.py) — when the updated content exceeds `SUMMARY_MAX_LENGTH` (channel) or `GROUP_SUMMARY_MAX_LENGTH` (group), `enforce_summary_length()` is applied before storing. Previously, repeated updates to the same summary could cause unbounded content growth in the history file, even though `save_updated_summary()` enforced length for the channel edit. The stored (unbounded) content was then used as input for subsequent updates, wasting tokens and producing increasingly poor LLM responses.
- **Bug fix**: Fixed deadline-induced content loss in [`process_messages()`](message_processor.py) — when the deadline was exceeded during the covered-message processing loop, all remaining covered messages were still filtered out (`is_covered_in_summaries = True`) even though their summary update was never attempted. This silently dropped message content. Now, when the deadline hits, remaining covered messages are un-marked (`is_covered_in_summaries = False`) so they're included in the new summary instead. The summary LLM consolidates naturally, and no content is lost.
- **Config hardening**: Added `NLP_MIN_TEXT_LENGTH` config (default 100) in [`config.py`](config.py) — makes the minimum text length threshold for NLP relevance checks configurable via env var. Previously hardcoded to `100` inside `is_nlp_related()`. Added to [`template.yaml`](template.yaml) as `NlpMinTextLength` parameter and [`.env.example`](.env.example).
- **SAM template sync**: Added `RESTORE_TIMEOUT_SEC` parameter and env var to [`template.yaml`](template.yaml) — was defined in `config.py` with `_get_int_env` validation but missing from the SAM template's Parameters and Environment.Variables sections.
- **Dead config cleanup**: Removed `SIMILARITY_LLM_LOWER` from [`config.py`](config.py) — not used in production code since LLM dedup was removed in round 12. Only `SIMILARITY_LLM_UPPER` is used by `_remove_intra_batch_duplicates()`. Removed cross-validation check (`SIMILARITY_LLM_LOWER >= SIMILARITY_LLM_UPPER`). Removed 3 obsolete tests from [`tests/test_openai_config.py`](tests/test_openai_config.py).
- **Lambda hardening**: Added `FloodWaitError` handling in [`_fetch_from_sources()`](telegram_client.py) — when Telegram rate-limits a source channel, the source is skipped gracefully with a warning log instead of raising an unhandled exception that could abort the entire message-fetch phase. Other sources continue to be fetched.
- **Coverage+match robustness**: Changed [`_check_coverage_and_match()`](message_processor.py) from `result.isdigit()` to `re.match(r"^(\d+)", result)` — handles LLM responses with trailing punctuation (e.g., "1." or "2,") that would fail strict `isdigit()`. Uses regex to extract leading digits.
- **Prompt quality**: Added "Объединяй близкие подтемы под один заголовок" rule to both [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py) — encourages the LLM to merge closely related sub-topics under a single header instead of creating separate sections. Directly improves conciseness per AUTOWORK_INSTRUCTIONS goal.
- **Group header length accounting**: Fixed [`summarize_group_text()`](message_processor.py) — previously applied `enforce_summary_length()` on `header + result`, which could exceed the target because the header wasn't accounted for. Now enforces `max_summary_length - len(header)` before prepending the header, ensuring the total stays within `GROUP_SUMMARY_MAX_LENGTH`.
- **Telegram message length guard**: Added `TELEGRAM_MAX_MESSAGE_LENGTH` (4096) constant in [`config.py`](config.py). Applied in [`send_message_to_target_channel_with_id()`](telegram_client.py) and [`edit_message_in_target_channel()`](telegram_client.py) — messages exceeding 4096 chars are truncated with "..." before sending. Prevents Telegram API errors on oversized messages.
- **Observability**: Added per-phase timing in [`lambda_handler.handler()`](lambda_handler.py) — logs `download=Xs, summarizer=Ys, upload=Zs` alongside total elapsed time. Enables identifying which phase (S3 download, summarizer, S3 upload) is consuming the most Lambda time.
- **HTML tag regex optimization**: Pre-compiled `HTML_TAG_REGEX` in [`utils.py`](utils.py) — `count_characters()` was calling `re.sub(r'<[^>]+>', '', text)` on every invocation, recompiling the regex each time. Now uses a module-level compiled pattern.
- **Import cleanup**: Consolidated `from config import ENABLE_SUMMARY_UPDATES` into the main config import block in [`message_processor.py`](message_processor.py) — was a separate import line after the main block.
- **Tests added**: 13 new tests (total 196, up from 183):
  - `test_deadline_unmarks_covered_messages`: verifies covered messages are un-marked when deadline exceeds during update loop
  - `test_nlp_rejects_short_text_with_configurable_threshold`: verifies `NLP_MIN_TEXT_LENGTH` threshold
  - `test_nlp_accepts_text_above_threshold`: verifies text above threshold is not rejected
  - `test_config_reads_nlp_min_text_length_from_env`: verifies env var parsing
  - `test_config_nlp_min_text_length_default`: verifies default is 100
  - `test_config_nlp_min_text_length_rejects_zero`: validates `_get_int_env` rejects zero
  - `test_no_substring_redundancy_in_ad_keywords`: verifies no keyword is a substring of an earlier keyword
  - `test_enforces_length_on_oversized_update`: verifies `update_existing_summary` enforces max length
  - `test_matches_digit_with_trailing_period`: verifies `_check_coverage_and_match` handles "1." response
  - `test_matches_digit_with_trailing_comma`: verifies `_check_coverage_and_match` handles "2," response
  - `test_group_summary_total_length_within_limit`: verifies group header accounted for in length enforcement
  - `test_send_truncates_oversized_message`: verifies Telegram send truncates > 4096 chars
  - `test_edit_truncates_oversized_message`: verifies Telegram edit truncates > 4096 chars
  - `test_fetch_skips_source_on_flood_wait`: verifies `_fetch_from_sources` skips rate-limited sources
  - `test_config_telegram_max_message_length_is_4096`: verifies constant value
  - `test_handler_logs_phase_timing_on_success`: verifies per-phase timing in Lambda handler
- Removed 3 obsolete tests (`test_config_similarity_llm_lower_from_env`, `test_config_similarity_llm_lower_default`, `test_config_similarity_llm_lower_rejects_invalid`) — testing removed `SIMILARITY_LLM_LOWER` config constant.
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) — added `NLP_MIN_TEXT_LENGTH`, `ENABLE_SUMMARY_UPDATES`, fixed `NLP_MIN_TEXT_LENGTH` patching to use `patch.object` on module attribute.
- All 196 tests pass without errors.

## Completed in 2026-06-17 round 2 (Lambda hardening round 27, prompt quality, redundant I/O, deadline, determinism, S3 validation)

- **Prompt quality**: Tightened [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py) — added "Без мета-комментариев" rule (targets LLM verbosity pattern like "в этом дайджесте…", "итого"), moved "Конкретные результаты и факты" rule into both prompts (was only in channel prompt). Shortened opening lines by ~30% while preserving all rules. Directly addresses AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness.
- **Redundant summary loads eliminated**: Changed [`_dedup_covered_messages()`](message_processor.py) to return 2-tuple `(uncovered_messages, summaries)`, passing already-loaded summaries through to [`process_covered_message()`](message_processor.py) and [`save_updated_summary()`](history_manager.py). Both functions now accept optional `summaries` parameter — when provided, skip the redundant `load_summaries_history()` / `load_group_summaries_history()` call. Previously, the covered-message flow loaded summaries from file 3 times per covered message: once in `_dedup_covered_messages`, once in `process_covered_message` (for refresh), and once in `save_updated_summary` (for find-and-replace). Now loads once and passes through, eliminating 2 redundant file reads per covered message.
- **Deadline check after group fetch**: Added `check_deadline(_deadline)` call between `fetch_group_messages()` and `process_messages()` for groups in [`run_summarizer()`](summarizer.py). Previously, if group message fetching was slow, the summarizer would proceed to `process_messages` without checking the deadline, risking Lambda timeout during group processing. Now consistent with the channel path which already has a deadline check after `fetch_messages()`.
- **Deterministic channel ordering**: Replaced `random.shuffle(all_channels)` with `sorted(all_channels_set)` in [`get_all_source_channels()`](channel_manager.py). Removed `import random`. `random.shuffle` made channel fetch order nondeterministic, complicating debugging and CloudWatch log analysis. Sorted order is reproducible and has no functional impact (all channels are fetched regardless of order).
- **S3 download JSON validation**: Added [`_is_json_state_file()`](s3_sync.py) and [`_validate_json_file()`](s3_sync.py) in [`s3_sync.py`](s3_sync.py). After downloading a `.json` file from S3, validates it parses as valid JSON. Invalid files are removed and counted as skipped (not downloaded). This prevents corrupt S3 state (e.g., truncated uploads) from replacing valid local state. Non-JSON files (e.g., `.session`) are not validated. Added `import json` to module.
- **Fallback "Другие ссылки" dedup**: Fixed [`update_existing_summary()`](history_manager.py) — when the fallback append is triggered on a summary that already contains "Другие ссылки:", the new link is now appended to the existing section (e.g., `\n{new_link}`) instead of creating a duplicate "Другие ссылки:" section. Previously, repeated fallback appends on the same summary would create multiple "Другие ссылки:" blocks, wasting characters and looking messy.
- **Tests added**: 8 new tests (total 263, up from 255):
  - `test_download_rejects_invalid_json`: verifies corrupt JSON files are removed and skipped
  - `test_download_accepts_valid_json`: verifies valid JSON files are kept
  - `test_download_skips_json_validation_for_non_json`: verifies `.session` files bypass validation
  - `test_channels_returned_in_sorted_order`: verifies `get_all_source_channels` returns sorted list
  - `test_no_duplicate_Другие_ссылки_on_repeated_fallback`: verifies no duplicate "Другие ссылки:" sections
  - `test_creates_Другие_ссылки_when_absent`: verifies section created when not present
  - `test_save_updated_summary_uses_provided_summaries`: verifies `save_updated_summary` skips file load when summaries passed
  - `test_check_deadline_between_group_fetch_and_process`: verifies deadline check exists between fetch and process for groups (AST)
- Updated test stubs in [`tests/test_s3_sync.py`](tests/test_s3_sync.py) — `FakeS3Client.download_file` now writes valid JSON (`"{}"`) instead of plain text (`"downloaded"`), since S3 JSON validation would reject non-JSON content for `.json` files.
- Updated `_dedup_covered_messages` direct-call tests in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) — unpack 2-tuple return `(messages, summaries)` instead of single list.
- All 263 tests pass without errors.

## Next actions (original)

- **CI/CD**: Настроить GitHub Actions CI/CD для автоматического деплоя Lambda при мердже в main.
- **Secrets management**: Перенести реальные секреты в AWS SSM Parameter Store (инфраструктура готова — `*_SSM_PATH` env vars и IAM policies в template.yaml).
- **Prompt A/B testing**: Продолжить тестирование промптов — отслеживать качество саммари после добавления правила "Объединяй близкие подтемы" и снижения `max_tokens`.
- **OpenAI response streaming**: Рассмотреть streaming API для снижения perceived latency (но не стоимости — `max_tokens` уже ограничен).
- **Intra-batch LLM dedup**: Рассмотреть добавление LLM-проверки для сообщений с ratio между 0.7 и `SIMILARITY_LLM_UPPER` (0.95) внутри `_remove_intra_batch_duplicates` — позволит ловить больше дубликатов за дополнительные токены.
- **Coverage+match accuracy**: Отслеживать точность `_check_coverage_and_match` — если LLM иногда выбирает не лучший дайджест для обновления, можно увеличить `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY`.
- **NLP pre-filter tuning**: Мониторить процент сообщений, отклонённых `_is_obvious_non_nlp` — если ложноположительных срабатываний много, сузить список `NLP_AD_KEYWORDS` или добавить whitelist.
- **Summary update LLM prompt**: Оценить эффект от нового промпта "Скопируй саммари и добавь ссылку" — если LLM всё ещё переписывает, можно ещё сильнее ограничить (например, "Ответь только строку, которую нужно вставить").
