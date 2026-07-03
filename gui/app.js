const NOTICE_INFO_MS = 6000;
const NOTICE_WARNING_MS = 7000;
const PROVIDER_LABELS = {
  openrouter: "OpenRouter API Key",
  openai: "OpenAI API Key",
  anthropic: "Anthropic API Key",
  custom: "API Key",
};

const PROVIDER_NAMES = {
  openrouter: "OpenRouter",
  openai: "OpenAI",
  anthropic: "Anthropic",
  custom: "Custom",
};

const EXPECTED_ENV_VARS = {
  openrouter: "OPENROUTER_API_KEY",
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
};

const MEMORY_STATUSES = ["current", "historical", "superseded", "archived"];
// User-selectable statuses. "superseded" is engine-managed (it implies a
// replacement chain) so it's never offered as a manual choice in the GUI.
const MEMORY_USER_STATUSES = ["current", "historical", "archived"];
const MEMORY_CONFIDENCES = ["high", "medium", "low"];
const MEMORY_SOURCES = ["user_defined", "model_extracted", "imported", "system"];
// Sane API-context default when switching to a cloud provider (local uses its
// own much smaller Local Server context). Only applied if the user hasn't set
// a deliberately larger value.
const CLOUD_DEFAULT_CONTEXT = 128000;
const LOCALISH_CONTEXT_CEILING = 32768;
const JOURNAL_ENTRY_TYPES = ["event", "preference", "topic", "tone", "open_thread", "follow_up", "reflection"];

const state = {
  personas: [],
  current: null,
  prefs: {},
  pendingTransition: null,
  dirty: false,
  canUndo: false,
  originalEditable: null,
  currentTraits: [],
  secretsRows: new Map(),
  currentLantern: null,
  lanternBaseline: null,
  currentCoreAnchor: null,
  coreBaseline: {},
  memoryBrowse: { view: "active", kind: "fact", page: 0, pageSize: 25, total: 0, hasMore: false, items: [] },
  memoryDetailId: "",
  memoryBaselines: {},
  memoryUndoStamp: "",
  journalBrowse: { view: "active", type: "open_thread", page: 0, pageSize: 25, total: 0, hasMore: false, items: [] },
  journalDetailId: "",
  journalBaseline: {},
  journalUndoStamp: "",
  taskBaselines: {},
  taskDrafts: [],
  taskOriginals: [],
  taskUndoStamp: "",
  scheduleDrafts: [],
  scheduleOriginals: [],
  scheduleUndoStamp: "",
  currentSkill: "",
};

const fallbackApi = {
  async list_personas() {
    return [
      { name: "__base__", display_name: "Base Config", provider: "local", model: "base", is_base: true, running: false },
      { name: "kai", display_name: "Kai", provider: "openrouter", model: "Grok 4.2", running: false },
    ];
  },
  async load_persona(name) {
    return {
      name,
      display_name: name === "__base__" ? "Base Config" : "Kai",
      identity: {
        name: name === "__base__" ? "Pulse" : "Kai",
        user_name: "Lena",
        model: name === "__base__" ? "" : "Grok 4.2",
        system_prompt: "pywebview is not connected. Run python pulse_gui.py for live project data.",
        traits: ["read-only", "mock data"],
        relationship_context: "",
        voice_notes: "",
      },
      summary: {
        display_name: name === "__base__" ? "Base Config" : "Kai",
        user_name: "Lena",
        model_display: name === "__base__" ? "base" : "Grok 4.2",
        provider_model: "",
        provider_type: name === "__base__" ? "local" : "openrouter",
        base_url: "",
        max_context: "",
        provider_max_context: "",
        local_max_context: "16384",
        model_file: "",
        mmproj_file: "",
        server: {
          llama_cpp_dir: "C:\\llama-cpp",
          models_dir: "C:\\llama-cpp\\models",
          host: "127.0.0.1",
          port: 8012,
          gpu_layers: -1,
          flash_attention: true,
          parallel: 1,
        },
        temperature: 0.7,
        max_response_tokens: 2048,
        frequency_penalty: 0.4,
        presence_penalty: 0.2,
        top_p: 1.0,
        reasoning: true,
        reasoning_effort: "high",
        show_reasoning: false,
        max_tool_rounds: 8,
        context_budget: {
          recent_tail_exchanges: 2,
        },
        heartbeat: {
          interval_minutes: 30,
          randomize: true,
          interval_min_minutes: 30,
          interval_max_minutes: 60,
          quiet_hours_start: 23,
          quiet_hours_end: 8,
          debug: false,
        },
        tts: {},
      },
      skills: [
        { name: "lor", label: "Lor", icon: "LR", enabled: true },
        { name: "memory", label: "Memory", icon: "ME", enabled: true },
        { name: "tts", label: "Tts", icon: "TT", enabled: true },
      ],
      channels: [
        { name: "lor", label: "Lor", enabled: true },
        { name: "telegram", label: "Telegram", enabled: true },
        { name: "toast", label: "Toast", enabled: true },
      ],
      key_status: {
        provider_type: name === "__base__" ? "local" : "openrouter",
        api_key_env: "API_KEY",
        expected_api_key_env: "OPENROUTER_API_KEY",
        api_key_set: false,
        provider_key_status: { openrouter: false, openai: false, anthropic: false },
        provider_key_sources: { openrouter: "missing", openai: "missing", anthropic: "missing" },
        api_key_source: "missing",
        telegram_key: "TELEGRAM_BOT_TOKEN",
        telegram_set: false,
        telegram_source: "missing",
        telegram_enabled: true,
      },
      config: {
        paths: { lor_data: "data/lor" },
        channels: {
          lor: { author_name: "Kai", model_name: "Grok 4.2", context_initial_lookback_hours: 72 },
        },
        context: { inject_skills: ["lantern"] },
        dev_tick: {
          enabled: false,
          interval_minutes: 720,
          schedule_time: "",
          max_rounds: 16,
        },
      },
      status: {
        running: false,
        phase: "browser fallback",
        uptime_seconds: 3723,
        telegram_connected: null,
        tts_ready: true,
        llm_available: true,
        next_heartbeat_in: 1234,
        scheduled_tasks: 2,
        last_tick_action: "silent",
      },
      paths: { config: "", persona_dir: "", avatar: "" },
    };
  },
  async get_prefs() { return {}; },
  async save_prefs() { return { ok: true }; },
  async create_persona(fields) {
    return {
      ok: true,
      name: fields?.slug || "new_persona",
      display_name: fields?.display_name || "New Persona",
      persona: {
        name: fields?.slug || "new_persona",
        display_name: fields?.display_name || "New Persona",
        provider: "local",
        model: "fallback",
        is_base: false,
        running: false,
      },
    };
  },
  async import_openrouter_chat() { return { ok: false, error: "Run through pywebview to import OpenRouter chats." }; },
  async list_reminders() { return { ok: false, error: "Run through pywebview to manage reminders." }; },
  async add_reminder() { return { ok: false, error: "Run through pywebview to manage reminders." }; },
  async delete_reminder() { return { ok: false, error: "Run through pywebview to manage reminders." }; },
  async save_reminders() { return { ok: false, error: "Run through pywebview to manage reminders." }; },
  async pick_openrouter_export() { return { ok: false, error: "Run through pywebview to pick files." }; },
  async preview_persona_save() { return { ok: false, error: "Run through pywebview to save." }; },
  async save_persona() { return { ok: false, error: "Run through pywebview to save." }; },
  async get_secrets() { return { ok: false, error: "Run through pywebview to edit secrets." }; },
  async save_secrets() { return { ok: false, error: "Run through pywebview to edit secrets." }; },
  async reveal_secret() { return { ok: false, error: "Run through pywebview to edit secrets." }; },
  async list_backups() {
    return {
      ok: true,
      backups: [
        {
          path: "gui_data/backups/kai/20260525_120000",
          created_at: "2026-05-25T12:00:00Z",
          reason: "pre-edit",
          files: ["config.yaml", "persona.yaml"],
        },
      ],
    };
  },
  async restore_backup() { return { ok: false, error: "Run through pywebview to restore backups." }; },
  async restore_last_backup() { return { ok: false, error: "Run through pywebview to undo." }; },
  async get_lantern() { return { ok: false, error: "Run through pywebview to read lantern." }; },
  async set_lantern() { return { ok: false, error: "Run through pywebview to update lantern." }; },
  async dim_lantern() { return { ok: false, error: "Run through pywebview to dim lantern." }; },
  async get_garden_summary() {
    const grid = Array.from({ length: 8 }, (_, y) => (
      Array.from({ length: 12 }, (_, x) => ({ x, y, emoji: ".", empty: true, tooltip: `Empty plot (${x}, ${y})` }))
    ));
    return { ok: true, width: 12, height: 8, plant_count: 0, wilted_count: 0, needs_water_count: 0, plants: [], grid };
  },
  async get_paint_gallery() { return { ok: true, total: 0, items: [], paintings_dir: "" }; },
  async get_web_search_status() {
    return {
      ok: true,
      ddgs_available: false,
      requests_available: false,
      tools: [
        { label: "Web search", status: "pywebview only", description: "Searches DuckDuckGo for current web results." },
        { label: "Image search", status: "pywebview only", description: "Finds image results." },
        { label: "Fetch URL", status: "pywebview only", description: "Reads a page through Jina Reader." },
      ],
    };
  },
  async get_sticker_summary() { return { ok: true, ready: false, count: 0, with_embeddings: 0, packs: [] }; },
  async get_tasks_summary() { return { ok: true, pending: 0, completed: 0, recent: [] }; },
  async list_tasks() { return { ok: true, tasks: [] }; },
  async add_task() { return { ok: false, error: "Run through pywebview to add tasks." }; },
  async update_task() { return { ok: false, error: "Run through pywebview to edit tasks." }; },
  async delete_task() { return { ok: false, error: "Run through pywebview to delete tasks." }; },
  async save_tasks() { return { ok: false, error: "Run through pywebview to save tasks." }; },
  async list_memories() { return { ok: false, error: "Run through pywebview to browse memories." }; },
  async get_memory_detail() { return { ok: false, error: "Run through pywebview to inspect memories." }; },
  async add_memory() { return { ok: false, error: "Run through pywebview to add memories." }; },
  async update_memory() { return { ok: false, error: "Run through pywebview to edit memories." }; },
  async delete_memory() { return { ok: false, error: "Run through pywebview to delete memories." }; },
  async list_journal_entries() { return { ok: false, error: "Run through pywebview to browse journal entries." }; },
  async get_journal_entry() { return { ok: false, error: "Run through pywebview to inspect journal entries." }; },
  async update_journal_entry() { return { ok: false, error: "Run through pywebview to edit journal entries." }; },
  async delete_journal_entry() { return { ok: false, error: "Run through pywebview to delete journal entries." }; },
  async restore_db_before_image() { return { ok: false, error: "Run through pywebview to undo database edits." }; },
  async get_core_anchor() { return { ok: false, error: "Run through pywebview to inspect core anchors." }; },
  async set_core_anchor() { return { ok: false, error: "Run through pywebview to edit core anchors." }; },
  async pick_voice_sample() { return { ok: false, error: "Run through pywebview to pick files." }; },
  async pick_folder() { return { ok: false, error: "Run through pywebview to pick folders." }; },
  async pick_model_file() { return { ok: false, error: "Run through pywebview to pick model files." }; },
  async get_log_tail() { return "Run through pywebview to read logs."; },
  async start_pulse() { return { ok: false, error: "Run through pywebview to start Pulse." }; },
  async stop_pulse() { return { ok: false, error: "Run through pywebview to stop Pulse." }; },
  async open_folder() { return { ok: false, error: "Run through pywebview to open folders." }; },
  async stop_all_and_close() { return { ok: false, error: "Run through pywebview." }; },
  async close_keep_running() { return { ok: false, error: "Run through pywebview." }; },
  async check_close_request() { return null; },
};

function api() {
  return window.pywebview?.api || fallbackApi;
}

function el(id) {
  return document.getElementById(id);
}

function fieldValue(id) {
  return el(id)?.value || "";
}

let noticeTimer = null;
let secretsBackdropPointerDown = false;
let lanternBackdropPointerDown = false;
let memoriesBackdropPointerDown = false;
let journalBackdropPointerDown = false;
let coreBackdropPointerDown = false;
let skillBackdropPointerDown = false;
let confirmBackdropPointerDown = false;
let newPersonaBackdropPointerDown = false;

function setNotice(message, kind = "info", autoHideMs = 0) {
  const box = el("notice");
  if (noticeTimer) { clearTimeout(noticeTimer); noticeTimer = null; }
  if (!message) {
    box.classList.add("hidden");
    return;
  }
  box.textContent = message;
  box.dataset.kind = kind;
  box.classList.remove("hidden");
  if (autoHideMs > 0) {
    noticeTimer = setTimeout(() => { box.classList.add("hidden"); noticeTimer = null; }, autoHideMs);
  }
}

function setFooterNotice(message) {
  const box = el("footerNotice");
  if (!box) return;
  if (!message) {
    box.classList.add("hidden");
    box.textContent = "";
    return;
  }
  box.textContent = message;
  box.classList.remove("hidden");
}

function setDirty(value) {
  state.dirty = Boolean(value);
  const isBase = state.current?.name === "__base__";
  el("saveBtn").disabled = isBase || !state.dirty;
}

function setCanUndo(value) {
  state.canUndo = Boolean(value);
  const isBase = state.current?.name === "__base__";
  el("undoBtn").disabled = isBase || !state.canUndo;
}

function setBackupState() {
  const isBase = state.current?.name === "__base__";
  el("backupsBtn").disabled = !state.current || isBase;
}

function text(value, fallback = "") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function firstLetter(name) {
  return (name || "P").trim().slice(0, 1).toUpperCase();
}

function renderPersonaMenu() {
  const menu = el("personaMenu");
  menu.innerHTML = "";
  state.personas.forEach((persona) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.innerHTML = `<span>${persona.display_name}</span><small>${persona.provider || ""} ${persona.model || ""}</small>`;
    btn.addEventListener("click", () => {
      menu.classList.add("hidden");
      el("personaPicker").classList.remove("open");
      loadPersona(persona.name);
    });
    menu.appendChild(btn);
  });
  // Nudge brand-new users toward their first companion: glow the "+ New"
  // button while no real (non-base) persona exists yet.
  const hasRealPersona = state.personas.some((persona) => !persona.is_base);
  el("newPersonaBtn").classList.toggle("attention", !hasRealPersona);
}

function renderTuning(summary) {
  const s = summary || {};
  const sliders = [
    { label: "Temperature", key: "temperature", min: 0, max: 2, fmt: 2, tip: "How creative vs predictable. 0.7-0.85 is the sweet spot. Above 1.0 gets chaotic fast" },
    { label: "Max Response", key: "max_response_tokens", min: 256, max: 32768, fmt: 0, tip: "Max reply length in tokens. 2048-4096 is a good range; too high and replies can get rambly" },
    { label: "Freq Penalty", key: "frequency_penalty", min: 0, max: 2, fmt: 2, tip: "Discourages repeating the same words. 0.3-0.5 works well; above 1.0 can sound awkward" },
    { label: "Pres Penalty", key: "presence_penalty", min: 0, max: 2, fmt: 2, tip: "Encourages new topics. 0.1-0.3 is natural; too high and the companion jumps topics" },
    { label: "Top P", key: "top_p", min: 0, max: 1, fmt: 2, tip: "Word choice diversity. 0.95-1.0 is good. Lower values make responses safer but can feel flat" },
  ];

  let html = "";
  for (const sl of sliders) {
    const raw = s[sl.key];
    const val = parseFloat(raw);
    const ok = !isNaN(val);
    const pct = ok ? Math.min(100, Math.max(0, ((val - sl.min) / (sl.max - sl.min)) * 100)) : 0;
    const display = ok ? val.toFixed(sl.fmt) : "default";
    html += `<div class="slider-row" data-key="${sl.key}" data-min="${sl.min}" data-max="${sl.max}" data-fmt="${sl.fmt}" title="${escapeHtml(sl.tip)}">
      <span class="slider-label">${escapeHtml(sl.label)}</span>
      <div class="slider-track">
        <div class="slider-fill" style="width:${pct}%"></div>
        <div class="slider-handle" style="left:${ok ? pct : 0}%"></div>
      </div>
      <span class="slider-value${ok ? "" : " default"}">${escapeHtml(display)}</span>
    </div>`;
  }

  const reasoning = s.reasoning;
  const effort = text(s.reasoning_effort, "");
  const showR = s.show_reasoning;
  const rounds = text(s.max_tool_rounds, "default");
  const tail = text(s.context_budget?.recent_tail_exchanges, "2");
  const topK = text(s.top_k, "");

  html += `<div class="slider-row tuning-number-row" title="Top-K sampler. Local/OpenRouter/Anthropic can use it; OpenAI ignores it. Leave blank to use the provider default. 40 is common; Gemma recommends 64">
    <span class="slider-label">Top K</span>
    <input id="tuningTopK" class="tuning-inline-input" type="number" min="0" max="1000" placeholder="default" value="${escapeHtml(topK)}">
    <span class="tuning-number-spacer" aria-hidden="true"></span>
  </div>
  <div class="tuning-extras">
    <label class="check-label-inline" title="Let the model think step-by-step before answering. Uses more tokens but improves quality on complex tasks">
      <input id="tuningReasoning" type="checkbox" ${reasoning ? "checked" : ""}>
      Reasoning
    </label>
    <select id="tuningEffort" class="tuning-select" title="How hard the model thinks. High gives better answers but is slower and costs more. Some cheap models need this bumped up">
      <option value="" ${effort === "" ? "selected" : ""}>Default</option>
      <option value="low" ${effort === "low" ? "selected" : ""}>Low</option>
      <option value="medium" ${effort === "medium" ? "selected" : ""}>Medium</option>
      <option value="high" ${effort === "high" ? "selected" : ""}>High</option>
    </select>
    <label class="check-label-inline" title="Show the model's inner monologue in Telegram as an expandable blockquote. Fun for thinking models">
      <input id="tuningShowReasoning" type="checkbox" ${showR ? "checked" : ""}>
      Show Reasoning
    </label>
  </div>
  <div class="tuning-extras tuning-context-row">
    <label class="tuning-inline-label" title="Message pairs kept word-for-word after summarization. 2-4 is good; keeps continuity without eating up context">
      Recent Tail Exchanges
      <input id="tuningTailExchanges" class="tuning-inline-input" type="number" min="1" max="10" value="${escapeHtml(tail)}">
    </label>
    <label class="tuning-inline-label" title="How many tool calls the model can chain per turn. Default 8 is good; higher lets it do more autonomously but takes longer">
      Max Tool Rounds
      <input id="tuningMaxRounds" class="tuning-inline-input" type="number" min="1" max="32" value="${escapeHtml(rounds)}">
    </label>
  </div>`;

  el("tuningGrid").innerHTML = html;
  syncTuningProviderHints();
  wireTuningControls();
}

function syncTuningProviderHints() {
  const providerType = selectedProviderType(state.current?.key_status || {});
  const reasoningLabel = el("tuningReasoning")?.closest("label");
  const effortField = el("tuningEffort");
  const showLabel = el("tuningShowReasoning")?.closest("label");

  if (providerType === "anthropic") {
    const hint = "Anthropic thinking summaries are not currently requested or displayed by Pulse. This setting is saved for future support, but Claude API reasoning will not appear yet.";
    if (reasoningLabel) reasoningLabel.title = hint;
    if (effortField) effortField.title = "Saved for future Anthropic thinking support. Pulse currently does not send Anthropic adaptive thinking or effort parameters.";
    if (showLabel) showLabel.title = "Anthropic thinking summaries are not currently displayed by Pulse. For now, Show Reasoning only works with local/OpenRouter/OpenAI paths that expose reasoning.";
    return;
  }

  if (reasoningLabel) {
    reasoningLabel.title = "Let the model think step-by-step before answering. Uses more tokens but improves quality on complex tasks";
  }
  if (effortField) {
    effortField.title = "How hard the model thinks. High gives better answers but is slower and costs more. Some cheap models need this bumped up";
  }
  if (showLabel) {
    showLabel.title = "Show the model's inner monologue in Telegram as an expandable blockquote. Fun for thinking models";
  }
}

function renderStatus(status) {
  const pill = el("statusPill");
  const label = pill.querySelector("span");
  const pulse = document.querySelector(".brand-pulse");
  pill.classList.remove("running", "warning", "stopped");
  if (status?.phase === "stopping") {
    pill.classList.add("warning");
    label.textContent = "Stopping...";
    pulse?.classList.remove("flatline");
  } else if (status?.running) {
    pill.classList.add(status.stale ? "warning" : "running");
    label.textContent = status.stale
      ? "Unresponsive"
      : status.source === "status-file" ? "Running (external)" : "Running";
    pulse?.classList.remove("flatline");
  } else {
    pill.classList.add("stopped");
    label.textContent = text(status?.phase, "Stopped");
    pulse?.classList.add("flatline");
  }
}

function selectedProviderType(status = {}) {
  const field = el("providerType");
  return field?.value || status.provider_type || state.current?.summary?.provider_type || "local";
}

function normalizeKeyStatus(status = {}) {
  const providerType = selectedProviderType(status);
  const providerChanged = Boolean(status.provider_type) && status.provider_type !== providerType;
  const expected = status.provider_type === providerType
    ? status.expected_api_key_env || EXPECTED_ENV_VARS[providerType] || ""
    : EXPECTED_ENV_VARS[providerType] || "";
  const telegramChannel = state.current?.channels?.find((item) => item.name === "telegram");
  const apiKeyEnv = providerChanged ? "" : status.api_key_env || "";
  const configuredKeyMatches = Boolean(apiKeyEnv && apiKeyEnv === expected);
  const providerKeyStatus = status.provider_key_status || {};
  const providerKeySet = providerKeyStatus[providerType];
  return {
    provider_type: providerType,
    api_key_env: apiKeyEnv,
    expected_api_key_env: expected,
    api_key_set: providerKeySet !== undefined
      ? Boolean(providerKeySet)
      : providerChanged && providerType !== "local" && !configuredKeyMatches
        ? false
        : Boolean(status.api_key_set),
    provider_key_status: providerKeyStatus,
    provider_key_sources: status.provider_key_sources || {},
    api_key_source: status.api_key_source || "missing",
    telegram_key: status.telegram_key || "TELEGRAM_BOT_TOKEN",
    telegram_set: Boolean(status.telegram_set),
    telegram_source: status.telegram_source || "missing",
    telegram_enabled: telegramChannel ? Boolean(telegramChannel.enabled) : Boolean(status.telegram_enabled),
  };
}

function currentKeyStatus() {
  return normalizeKeyStatus(state.current?.key_status || {});
}

function providerKeyEnv(status) {
  return status.api_key_env || status.expected_api_key_env || "";
}

function providerKeyMissing(status = currentKeyStatus()) {
  return status.provider_type !== "local" && !status.api_key_set;
}

function renderProcessButton(data) {
  const btn = el("pulseToggle");
  const isBase = data?.name === "__base__";
  const status = data?.status || {};
  const active = Boolean(status.running) && !status.stale;
  const stopping = status.phase === "stopping";
  const keyStatus = currentKeyStatus();
  const missingKey = providerKeyMissing(keyStatus);
  const summary = data?.summary || state.current?.summary || {};
  const providerType = summary.provider_type || "local";
  const missingLocalModel = providerType === "local" && !text(summary.model_file).trim();

  if (active || stopping) {
    btn.textContent = stopping ? "Stopping..." : "■ Stop";
    btn.className = "hero-stop-btn";
    btn.disabled = isBase || stopping;
    btn.title = "";
  } else {
    btn.textContent = "▶ Start";
    btn.className = "hero-start-btn";
    btn.disabled = isBase || missingKey || missingLocalModel;
    btn.title = missingKey
      ? `API key missing - set ${providerKeyEnv(keyStatus) || "provider.api_key_env"} in .env to start`
      : missingLocalModel
        ? "No local model file set - choose a .gguf in Local Server settings before starting"
        : "";
  }
}

function formatDuration(value) {
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

function formatCountdown(value) {
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds <= 0) return "now";
  if (seconds < 60) return "<1m";
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.ceil((seconds % 3600) / 60);
  return minutes >= 60 ? `${hours + 1}h` : `${hours}h ${minutes}m`;
}

function formatTimeAgo(iso) {
  if (!iso) return "never";
  const when = new Date(iso).getTime();
  if (Number.isNaN(when)) return "unknown";
  const diff = Math.max(0, Math.floor((Date.now() - when) / 1000));
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m ago`;
}

function formatBackupTime(iso) {
  if (!iso) return "Unknown time";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return iso;
  return when.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function humanizeReason(reason) {
  const map = { "pre-edit": "Before save", "pre-restore": "Before restore" };
  if (!reason) return "Backup";
  return map[reason] || reason.replace(/[-_]/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function runtimeState(value, labels) {
  if (value === true) return labels.true;
  if (value === false) return labels.false;
  return labels.nullish;
}

function runtimeRows(status, summary) {
  const telegram = runtimeState(status?.telegram_connected, {
    true: "Connected",
    false: "Disconnected",
    nullish: "Not configured",
  });
  const tts = runtimeState(status?.tts_ready, {
    true: "Ready",
    false: "Warming up",
    nullish: "Not configured",
  });
  const llm = runtimeState(status?.llm_available, {
    true: "Available",
    false: "Unavailable",
    nullish: "Unknown",
  });
  return [
    ["Phase", text(status?.phase, status?.running ? "running" : "stopped")],
    ["PID", text(status?.pid, "none")],
    ["Uptime", status?.uptime_seconds != null ? formatDuration(status.uptime_seconds) : "-"],
    ["Provider", text(summary.provider_type, "unknown")],
    ["Model", text(summary.provider_model || summary.model_display, "not set")],
    ["Telegram", telegram],
    ["TTS", tts],
    ["LLM", llm],
    ["Next heartbeat", status?.next_heartbeat_in != null ? formatCountdown(status.next_heartbeat_in) : "-"],
    ["Tasks scheduled", text(status?.scheduled_tasks, "0")],
    ["Last activity", formatTimeAgo(status?.last_activity_at || status?.last_heartbeat)],
    ["Last tick", formatTimeAgo(status?.last_tick_at)],
    ["Last action", text(status?.last_tick_action, "none")],
    ["Last error", text(status?.last_error, "none")],
  ];
}

function renderRuntime(status, summary) {
  el("runtimeGrid").innerHTML = runtimeRows(status || {}, summary || {}).map(([k, v]) => (
    `<div class="runtime-item"><span>${escapeHtml(k)}</span><strong>${escapeHtml(v)}</strong></div>`
  )).join("");
}

function renderSkills(skills) {
  const grid = el("skillsGrid");
  if (!skills?.length) {
    grid.innerHTML = `<div class="runtime-item"><span>No skills discovered</span><strong></strong></div>`;
    return;
  }
  const editable = state.current?.name && state.current.name !== "__base__";
  grid.innerHTML = skills.map((skill) => `
    <div class="skill-card ${skill.enabled ? "on" : ""} ${editable ? "" : "disabled"}"
      data-skill="${escapeHtml(skill.name)}" role="button"
      aria-disabled="${editable ? "false" : "true"}" tabindex="${editable ? "0" : "-1"}">
      <div class="skill-icon">${escapeHtml(skill.icon)}</div>
      <div class="skill-info">
        <div class="skill-name">${escapeHtml(skill.label)}</div>
        <div class="skill-state">${skill.enabled ? "Enabled" : "Disabled"}</div>
      </div>
      <button class="skill-toggle" type="button"
        data-skill-toggle="${escapeHtml(skill.name)}"
        aria-label="${skill.enabled ? "Disable" : "Enable"} ${escapeHtml(skill.label)}"
        aria-pressed="${skill.enabled ? "true" : "false"}" ${editable ? "" : "disabled"}></button>
    </div>
  `).join("");
  grid.querySelectorAll(".skill-card").forEach((card) => {
    card.addEventListener("click", () => {
      if (!editable) return;
      openSkillDialog(card.dataset.skill);
    });
    card.addEventListener("keydown", (event) => {
      if (!editable || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      openSkillDialog(card.dataset.skill);
    });
  });
  grid.querySelectorAll(".skill-toggle").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!editable) return;
      const skill = currentSkill(button.dataset.skillToggle);
      if (!skill) return;
      skill.enabled = !skill.enabled;
      renderSkills(state.current.skills);
      setDirty(hasEditableChanges());
    });
  });
}

function currentSkill(name) {
  return state.current?.skills?.find((item) => item.name === name) || null;
}

function skillLabel(skill) {
  return skill?.label || (skill?.name || "Skill").replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

function lorDataPath() {
  return text(state.current?.config?.paths?.lor_data);
}

function setLorDataPath(value) {
  state.current.config = state.current.config || {};
  state.current.config.paths = state.current.config.paths || {};
  state.current.config.paths.lor_data = value;
}

function lorConfig() {
  state.current.config = state.current.config || {};
  state.current.config.channels = state.current.config.channels || {};
  state.current.config.channels.lor = state.current.config.channels.lor || {};
  return state.current.config.channels.lor;
}

function lorAuthorValue() {
  const lor = state.current?.config?.channels?.lor || {};
  return text(lor.author_name);
}

function lorAuthorFallback() {
  return text(state.current?.identity?.name || state.current?.display_name || state.current?.name, "Companion");
}

function lorModelValue() {
  const lor = state.current?.config?.channels?.lor || {};
  return text(lor.model_name);
}

function lorModelFallback() {
  return text(state.current?.summary?.provider_model || state.current?.summary?.model_display, "not set");
}

function lorLookbackHours() {
  const lor = state.current?.config?.channels?.lor || {};
  return lor.context_initial_lookback_hours ?? 72;
}

function setLorChannelField(key, value) {
  const lor = lorConfig();
  lor[key] = value;
}

function contextInjectSkills() {
  const skills = state.current?.config?.context?.inject_skills;
  return Array.isArray(skills) ? [...skills] : [];
}

function skillContextInjected(skillName) {
  return contextInjectSkills().includes(skillName);
}

function setSkillContextInjected(skillName, enabled) {
  state.current.config = state.current.config || {};
  state.current.config.context = state.current.config.context || {};
  const current = contextInjectSkills().filter((name) => name !== skillName);
  if (enabled) current.push(skillName);
  state.current.config.context.inject_skills = current;
}

function devTickConfig() {
  state.current.config = state.current.config || {};
  state.current.config.dev_tick = state.current.config.dev_tick || {};
  return state.current.config.dev_tick;
}

function devTickValue(key, fallback = "") {
  const devTick = state.current?.config?.dev_tick || {};
  return devTick[key] ?? fallback;
}

function setDevTickField(key, value) {
  devTickConfig()[key] = value;
}

function ttsSkillMode() {
  const tts = state.current?.config?.tts || {};
  const hasDescription = Boolean(text(tts.voice_description).trim());
  const hasSample = Boolean(text(tts.voice_sample).trim());
  const hasSampleText = Boolean(text(tts.voice_sample_text).trim());
  if (hasSample && hasSampleText) return "Clone mode";
  if (hasDescription) return "Design mode";
  return "Not configured";
}

function ttsSkillModel() {
  const engine = state.current?.summary?.tts_engine || {};
  const mode = ttsSkillMode();
  if (mode === "Clone mode") return engine.clone_model || "Qwen3-TTS clone model";
  if (mode === "Design mode") return engine.design_model || "Qwen3-TTS VoiceDesign";
  return engine.design_model || "Qwen3-TTS VoiceDesign";
}

function ttsSkillBackend() {
  const engine = state.current?.summary?.tts_engine || {};
  return engine.engine || "Qwen3-TTS";
}

function ttsSkillBackendNote() {
  const engine = state.current?.summary?.tts_engine || {};
  return engine.backend_note || "Uses faster-qwen3-tts when installed; otherwise falls back to upstream qwen_tts.";
}

function ttsSkillDescription() {
  const mode = ttsSkillMode();
  if (mode === "Clone mode") return "Reference audio and transcript are set.";
  if (mode === "Design mode") return "Voice description is set; samples will vary slightly.";
  return "Add a voice description, or choose a reference sample in the TTS section.";
}

function renderSkillDialogBody(skill) {
  if (skill?.name === "dev") {
    return `
      <div class="skill-setting-stack">
        <label class="check-row skill-setting-check" title="Let this companion periodically inspect Pulse code and suggest improvements. Changes still require human review.">
          <span>Autonomous dev tick</span>
          <input id="skillDevTickEnabled" type="checkbox" ${devTickValue("enabled", false) ? "checked" : ""}>
        </label>
        <p class="field-hint">Runs only when the persona is restarted with Dev Tick enabled.</p>
      </div>
      <div class="skill-setting-grid">
        <label class="sample-path-label" title="Minutes between autonomous dev checks when no daily schedule time is set.">Interval Minutes
          <input id="skillDevTickInterval" type="number" min="1" max="10080" step="1" value="${escapeHtml(String(devTickValue("interval_minutes", 720)))}">
        </label>
        <label class="sample-path-label" title="Optional local 24-hour time for one daily dev check. Leave blank to use the interval.">Daily Schedule Time
          <input id="skillDevTickSchedule" type="time" value="${escapeHtml(text(devTickValue("schedule_time", "")))}">
        </label>
      </div>
      <div class="skill-setting-stack">
        <label class="sample-path-label" title="Maximum tool-calling rounds for one autonomous dev session.">Max Rounds
          <input id="skillDevTickMaxRounds" type="number" min="1" max="32" step="1" value="${escapeHtml(String(devTickValue("max_rounds", 16)))}">
        </label>
      </div>
    `;
  }

  if (skill?.name === "lantern") {
    return `
      <div class="skill-setting-stack">
        <label class="check-row skill-setting-check" title="Injects the current lantern state into model context on each turn. Takes effect after restart.">
          <span>Inject lantern into context</span>
          <input id="skillLanternContextInject" type="checkbox" ${skillContextInjected("lantern") ? "checked" : ""}>
        </label>
        <div class="skill-empty-settings">
          <strong>Lantern content lives in Continuity</strong>
          <span>Use the Continuity section to view, dim, clear, or update the lantern itself.</span>
        </div>
      </div>
    `;
  }

  if (skill?.name === "garden") {
    return `
      <div class="skill-setting-stack">
        <label class="check-row skill-setting-check" title="Injects a compact garden status into model context on each turn. Takes effect after restart.">
          <span>Inject garden into context</span>
          <input id="skillGardenContextInject" type="checkbox" ${skillContextInjected("garden") ? "checked" : ""}>
        </label>
        <div id="skillGardenPreview" class="skill-garden-preview">
          <div class="skill-empty-settings">
            <strong>Loading garden</strong>
            <span>Reading the current garden grid...</span>
          </div>
        </div>
        <div class="skill-empty-settings">
          <strong>Companion-tended for now</strong>
          <span>Planting, watering, and pruning are still companion tool actions.</span>
        </div>
      </div>
    `;
  }

  if (skill?.name === "paint") {
    return `
      <div class="skill-setting-stack">
        <div id="skillPaintGallery" class="skill-paint-gallery">
          <div class="skill-empty-settings">
            <strong>Loading gallery</strong>
            <span>Reading recent paintings...</span>
          </div>
        </div>
        <div class="skill-empty-settings">
          <strong>Companion-painted for now</strong>
          <span>Starting, editing, and finishing paintings are still companion tool actions.</span>
        </div>
      </div>
    `;
  }

  if (skill?.name === "web_search") {
    return `
      <div class="skill-setting-stack">
        <div id="skillWebStatus" class="skill-tool-panel">
          <div class="skill-empty-settings">
            <strong>Loading web tools</strong>
            <span>Checking local dependencies...</span>
          </div>
        </div>
      </div>
    `;
  }

  if (skill?.name === "memory") {
    return `
      <div class="skill-setting-stack">
        <div class="skill-empty-settings">
          <strong>Memory browsing lives in Continuity</strong>
          <span>Use Continuity to browse, add, edit, archive, or restore memories. This skill controls whether the companion can use memory tools.</span>
        </div>
        <button id="skillOpenMemories" class="subtle-wide-btn" type="button">Open Memories</button>
      </div>
    `;
  }

  if (skill?.name === "journal") {
    return `
      <div class="skill-setting-stack">
        <div class="skill-empty-settings">
          <strong>Journal browsing lives in Continuity</strong>
          <span>Use Continuity to browse, edit, resolve, or delete journal entries. This skill controls whether the companion can use journal tools.</span>
        </div>
        <button id="skillOpenJournal" class="subtle-wide-btn" type="button">Open Journal</button>
      </div>
    `;
  }

  if (skill?.name === "tts") {
    return `
      <div class="skill-setting-stack">
        <div class="skill-setting-summary">
          <div><span>Mode</span><strong>${escapeHtml(ttsSkillMode())}</strong></div>
          <div><span>Tool</span><strong>speak</strong></div>
        </div>
        <div class="skill-setting-summary">
          <div><span>Active model</span><strong>${escapeHtml(ttsSkillModel())}</strong></div>
          <div><span>Backend</span><strong>${escapeHtml(ttsSkillBackend())}</strong></div>
        </div>
        <div class="skill-empty-settings">
          <strong>${escapeHtml(ttsSkillDescription())}</strong>
          <span>${escapeHtml(ttsSkillBackendNote())} Model choice is fixed for now; future versions may expose more TTS backends here.</span>
        </div>
        <button id="skillOpenTtsSection" class="subtle-wide-btn" type="button">Open TTS Voice Section</button>
      </div>
    `;
  }

  if (skill?.name === "sticker") {
    return `
      <div class="skill-setting-stack">
        <div id="skillStickerSummary" class="skill-tool-panel">
          <div class="skill-empty-settings">
            <strong>Loading stickers</strong>
            <span>Checking the sticker database...</span>
          </div>
        </div>
        <div class="skill-empty-settings">
          <strong>Companion-sent for now</strong>
          <span>Stickers are selected by mood or situation through companion tools.</span>
        </div>
      </div>
    `;
  }

  if (skill?.name === "tasks") {
    return `
      <div class="skill-setting-stack">
        <label class="check-row skill-setting-check" title="Injects a compact numbered pending-task list into model context on each turn. Save from the main footer; takes effect after restart.">
          <span>Inject pending tasks into context</span>
          <input id="skillTasksContextInject" type="checkbox" ${skillContextInjected("tasks") ? "checked" : ""}>
        </label>
        <div id="skillTasksSummary" class="skill-tool-panel">
          <div class="skill-empty-settings">
            <strong>Loading tasks</strong>
            <span>Reading the current task list...</span>
          </div>
        </div>
        <div id="skillTasksEditor" class="skill-tool-panel">
          <div class="skill-empty-settings">
            <strong>Loading task editor</strong>
            <span>Preparing editable checklist...</span>
          </div>
        </div>
      </div>
    `;
  }

  if (skill?.name === "schedule") {
    return `
      <div class="skill-setting-stack">
        <div id="skillScheduleSummary" class="skill-tool-panel">
          <div class="skill-empty-settings">
            <strong>Loading reminders</strong>
            <span>Reading active scheduled reminders...</span>
          </div>
        </div>
        <div class="skill-tool-card task-editor-card">
          <div class="skill-garden-head schedule-add-head">
            <span>Add reminder</span>
            <div class="schedule-head-actions">
              <strong>Staged until Save</strong>
              <div id="skillScheduleNotice" class="schedule-local-status hidden"></div>
            </div>
          </div>
          <div class="task-add-row schedule-add-row">
            <label class="task-editor-field wide" title="What should this persona remind you about or follow up on?">
              <span>Reminder</span>
              <input id="scheduleTaskText" type="text" placeholder="What should happen later?">
            </label>
            <label class="task-editor-field" title="Choose whether this is a one-time reminder or a recurring schedule.">
              <span>Type</span>
              <select id="scheduleMode">
                <option value="once">One-time</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
                <option value="yearly">Yearly</option>
              </select>
            </label>
            <label class="task-editor-field" title="Priority shown to the companion when the reminder is due.">
              <span>Priority</span>
              <select id="schedulePriority">
                <option value="routine">Routine</option>
                <option value="urgent">Urgent</option>
                <option value="creative">Creative</option>
              </select>
            </label>
            <div class="schedule-detail-block">
              <div id="scheduleOnceFields" class="schedule-detail-grid">
                <label class="task-editor-field" title="Local time for the one-time reminder.">
                  <span>Time</span>
                  <input id="scheduleOnceTime" type="time">
                </label>
                <label class="task-editor-field" title="Local calendar date for the one-time reminder.">
                  <span>Date</span>
                  <input id="scheduleDate" type="date">
                </label>
              </div>
              <div id="scheduleRecurringFields" class="schedule-detail-grid hidden">
                <label class="task-editor-field schedule-time-field" title="Local 24-hour time for the recurring reminder.">
                  <span>Time</span>
                  <input id="scheduleTime" type="time">
                </label>
                <label id="scheduleWeekdayField" class="task-editor-field hidden" title="Weekly reminder day.">
                  <span>Day</span>
                  <select id="scheduleWeekday">
                    <option value="monday">Monday</option>
                    <option value="tuesday">Tuesday</option>
                    <option value="wednesday">Wednesday</option>
                    <option value="thursday">Thursday</option>
                    <option value="friday">Friday</option>
                    <option value="saturday">Saturday</option>
                    <option value="sunday">Sunday</option>
                  </select>
                </label>
                <label id="scheduleMonthdayField" class="task-editor-field hidden" title="Monthly reminder day.">
                  <span>Day</span>
                  <input id="scheduleMonthday" type="number" min="1" max="31" step="1" value="1">
                </label>
                <label id="scheduleYearDateField" class="task-editor-field hidden" title="Pick the yearly month and day. The year is ignored.">
                  <span>Date</span>
                  <input id="scheduleYearDate" type="date">
                </label>
              </div>
              <p id="schedulePreview" class="schedule-preview">Choose a date and time for this one-time reminder.</p>
            </div>
            <button id="scheduleAddBtn" class="subtle-wide-btn task-add-btn schedule-add-btn" type="button">Add Reminder</button>
          </div>
          <p class="task-editor-hint">One-time reminders use a calendar date and local time. Recurring reminders repeat daily, weekly, monthly, or yearly.</p>
        </div>
      </div>
    `;
  }

  if (skill?.name !== "lor") {
    return `
      <div class="skill-empty-settings">
        <strong>No extra settings yet.</strong>
        <span>Use the switch on the skill card to enable or disable this skill. More settings can live here later.</span>
      </div>
    `;
  }

  return `
    <div class="skill-setting-block">
      <label class="sample-path-label" title="Shared forum folder for this persona. Use the same folder to share one forum, or a different folder for separate forums.">LoR Data Folder
        <div class="sample-input-row">
          <input id="skillLorDataPath" type="text" value="${escapeHtml(lorDataPath())}" placeholder="D:\\Claude\\LoR\\lor_data">
          <button id="skillLorDataPicker" class="icon-btn sample-picker-btn" type="button" title="Choose LoR data folder">📂</button>
        </div>
      </label>
      <p class="field-hint">Shared forum storage for this persona.</p>
    </div>
    <div class="skill-setting-grid">
      <label class="sample-path-label" title="Shown as this persona's name on LoR posts. Blank falls back to the persona name.">Author Name
        <input id="skillLorAuthor" type="text" value="${escapeHtml(lorAuthorValue())}" placeholder="${escapeHtml(lorAuthorFallback())}">
      </label>
      <label class="sample-path-label" title="Shown beside the author on LoR. Blank falls back to the provider model.">Model Name
        <input id="skillLorModel" type="text" value="${escapeHtml(lorModelValue())}" placeholder="${escapeHtml(lorModelFallback())}">
      </label>
    </div>
    <div class="skill-setting-stack">
      <label class="sample-path-label" title="Hours of recent LoR activity to surface the first time this persona has no saved inbox cursor.">Initial Inbox Lookback
        <input id="skillLorLookback" type="number" min="1" max="8760" step="1" value="${escapeHtml(String(lorLookbackHours()))}">
      </label>
      <label class="check-row skill-setting-check" title="Lets the model see unread LoR posts/replies in prompt context without calling a tool. Takes effect after restart.">
        <span>Inject unread inbox into context</span>
        <input id="skillLorContextInject" type="checkbox" ${skillContextInjected("lor") ? "checked" : ""}>
      </label>
    </div>
    <div class="skill-setting-summary">
      <div><span>Effective author</span><strong>${escapeHtml(lorAuthorValue() || lorAuthorFallback())}</strong></div>
      <div><span>Effective model</span><strong>${escapeHtml(lorModelValue() || lorModelFallback())}</strong></div>
    </div>
  `;
}

function openSkillDialog(skillName) {
  if (!state.current || state.current.name === "__base__") return;
  const skill = currentSkill(skillName);
  if (!skill) return;
  state.currentSkill = skillName;
  resetSkillDialogActions();
  el("skillTitle").textContent = skillLabel(skill);
  el("skillSubtitle").textContent = skill.name === "lor"
    ? "Configure this persona's forum identity and inbox behavior."
    : skill.name === "dev"
      ? "Configure autonomous development checks for this persona."
      : skill.name === "lantern"
        ? "Configure how Lantern participates in this persona's context."
        : skill.name === "garden"
          ? "Configure garden context and peek at the current grid."
          : skill.name === "paint"
            ? "Browse recent tiny paintings saved by this persona."
            : skill.name === "web_search"
              ? "Check web, image, and page-fetch search tools."
              : skill.name === "memory"
                ? "Jump to the Continuity memory browser."
                : skill.name === "journal"
                  ? "Jump to the Continuity journal browser."
                  : skill.name === "tts"
                    ? "Review voice-message status and jump to voice setup."
                    : skill.name === "sticker"
                      ? "Check sticker pack readiness for this persona."
                      : skill.name === "tasks"
                        ? "Manage DB-backed checklist tasks."
                        : skill.name === "schedule"
                          ? "Manage DB-backed reminders and scheduled follow-ups."
                        : "Configure this skill for this persona. Save from the main footer when you're done.";
  el("skillNotice").textContent = skill.name === "paint"
    ? "Gallery is read-only here; painting still happens through companion tools."
    : skill.name === "web_search"
      ? "These tools use the network only when the companion calls them."
      : skill.name === "memory"
        ? "Memory records are managed in the Continuity section."
        : skill.name === "journal"
          ? "Journal records are managed in the Continuity section."
          : skill.name === "tts"
            ? "Voice settings are edited in the TTS Voice section."
            : skill.name === "sticker"
              ? "Sticker selection still happens through companion tools."
              : skill.name === "tasks"
                ? "Task edits use this dialog's Save. Context injection is saved from the main footer."
                : skill.name === "schedule"
                  ? "Reminder changes are staged here until you press Save."
    : "Changes here are staged until you use the main Save button.";
  el("skillNotice").classList.remove("hidden");
  el("skillBody").innerHTML = renderSkillDialogBody(skill);
  bindSkillDialogControls(skill);

  el("skillDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  if (skill.name === "garden") {
    loadGardenSkillPreview();
  } else if (skill.name === "paint") {
    loadPaintSkillGallery();
  } else if (skill.name === "web_search") {
    loadWebSearchSkillStatus();
  } else if (skill.name === "sticker") {
    loadStickerSkillSummary();
  } else if (skill.name === "tasks") {
    state.taskUndoStamp = "";
    loadTasksSkillSummary();
    loadTasksSkillEditor();
    syncTaskSaveButton();
  } else if (skill.name === "schedule") {
    state.scheduleUndoStamp = "";
    state.scheduleOriginals = [];
    state.scheduleDrafts = [];
    loadScheduleSkillSummary();
    syncScheduleSaveButton();
  }
}

function bindSkillDialogControls(skill) {
  if (skill?.name === "dev") {
    el("skillDevTickEnabled")?.addEventListener("change", (event) => {
      setDevTickField("enabled", event.target.checked);
      setDirty(hasEditableChanges());
    });
    el("skillDevTickInterval")?.addEventListener("input", (event) => {
      const raw = event.target.value;
      setDevTickField("interval_minutes", raw === "" ? "" : Number(raw));
      setDirty(hasEditableChanges());
    });
    el("skillDevTickSchedule")?.addEventListener("input", (event) => {
      setDevTickField("schedule_time", event.target.value);
      setDirty(hasEditableChanges());
    });
    el("skillDevTickMaxRounds")?.addEventListener("input", (event) => {
      const raw = event.target.value;
      setDevTickField("max_rounds", raw === "" ? "" : Number(raw));
      setDirty(hasEditableChanges());
    });
    return;
  }

  if (skill?.name === "lantern") {
    el("skillLanternContextInject")?.addEventListener("change", (event) => {
      setSkillContextInjected("lantern", event.target.checked);
      setDirty(hasEditableChanges());
    });
    return;
  }

  if (skill?.name === "garden") {
    el("skillGardenContextInject")?.addEventListener("change", (event) => {
      setSkillContextInjected("garden", event.target.checked);
      setDirty(hasEditableChanges());
    });
    return;
  }

  if (skill?.name === "tasks") {
    el("skillTasksContextInject")?.addEventListener("change", (event) => {
      setSkillContextInjected("tasks", event.target.checked);
      setDirty(hasEditableChanges());
    });
    return;
  }

  if (skill?.name === "tts") {
    el("skillOpenTtsSection")?.addEventListener("click", () => {
      closeSkillDialog();
      openTtsVoiceSection();
    });
    return;
  }

  if (skill?.name === "memory") {
    el("skillOpenMemories")?.addEventListener("click", () => openContinuityTarget("browse-memories"));
    return;
  }

  if (skill?.name === "journal") {
    el("skillOpenJournal")?.addEventListener("click", () => openContinuityTarget("browse-journal"));
    return;
  }

  if (skill?.name === "schedule") {
    wireScheduleSkillEditor();
    return;
  }

  if (skill?.name !== "lor") return;
  const pathInput = el("skillLorDataPath");
  pathInput?.addEventListener("input", () => {
    setLorDataPath(pathInput.value);
    setDirty(hasEditableChanges());
  });
  el("skillLorAuthor")?.addEventListener("input", (event) => {
    setLorChannelField("author_name", event.target.value);
    setDirty(hasEditableChanges());
  });
  el("skillLorModel")?.addEventListener("input", (event) => {
    setLorChannelField("model_name", event.target.value);
    setDirty(hasEditableChanges());
  });
  el("skillLorLookback")?.addEventListener("input", (event) => {
    const raw = event.target.value;
    setLorChannelField("context_initial_lookback_hours", raw === "" ? "" : Number(raw));
    setDirty(hasEditableChanges());
  });
  el("skillLorContextInject")?.addEventListener("change", (event) => {
    setSkillContextInjected("lor", event.target.checked);
    setDirty(hasEditableChanges());
  });
  el("skillLorDataPicker")?.addEventListener("click", async () => {
    const result = await api().pick_folder(pathInput.value || "");
    if (result?.ok && result.path) {
      pathInput.value = result.path;
      setLorDataPath(result.path);
      setDirty(hasEditableChanges());
    } else if (result && !result.ok && result.error) {
      el("skillNotice").textContent = result.error;
      el("skillNotice").classList.remove("hidden");
    }
  });
}

function closeSkillDialog() {
  el("skillDialog").classList.add("hidden");
  document.body.classList.remove("modal-open");
  state.currentSkill = "";
  resetSkillDialogActions();
}

function resetSkillDialogActions() {
  const done = el("skillDoneBtn");
  if (done) {
    done.textContent = "Done";
    done.disabled = false;
  }
  const close = el("skillCloseBtn");
  if (close) close.textContent = "Close";
}

async function handleSkillDone() {
  if (state.currentSkill === "tasks") {
    await saveTaskDrafts();
    return;
  }
  if (state.currentSkill === "schedule") {
    await saveReminderDrafts();
    return;
  }
  closeSkillDialog();
}

async function handleSkillClose() {
  if (state.currentSkill === "tasks" && taskDraftDirty()) {
    const ok = await showConfirm(
      "Discard task changes?",
      "Task changes are staged in this window. Closing now will discard unsaved edits.",
      "Discard",
      "secondary"
    );
    if (!ok) return;
  }
  if (state.currentSkill === "schedule" && scheduleDraftDirty()) {
    const ok = await showConfirm(
      "Discard reminder changes?",
      "Reminder changes are staged in this window. Closing now will discard unsaved edits.",
      "Discard",
      "secondary"
    );
    if (!ok) return;
  }
  closeSkillDialog();
}

function openTtsVoiceSection() {
  const field = el("ttsVoice");
  const section = field?.closest(".section");
  if (!section) return;
  section.classList.add("open");
  section.scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => field.focus({ preventScroll: true }), 250);
}

function openContinuityTarget(action) {
  closeSkillDialog();
  const section = el("continuitySection");
  section?.classList.add("open");
  section?.scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => handleContinuityAction(action), 220);
}

async function loadGardenSkillPreview() {
  const node = el("skillGardenPreview");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = `
    <div class="skill-empty-settings">
      <strong>Loading garden</strong>
      <span>Reading the current garden grid...</span>
    </div>
  `;
  const result = await api().get_garden_summary(state.current.name);
  if (!node.isConnected || state.currentSkill !== "garden") return;
  if (!result.ok) {
    node.innerHTML = `
      <div class="skill-empty-settings">
        <strong>Garden unavailable</strong>
        <span>${escapeHtml(result.error || "Could not read garden.")}</span>
      </div>
    `;
    return;
  }
  node.innerHTML = renderGardenSkillPreview(result);
}

function renderGardenSkillPreview(data) {
  const rows = data.grid || [];
  const grid = rows.map((row) => `
    <div class="skill-garden-row">
      ${(row || []).map((plot) => `
        <span class="skill-garden-plot ${plot.empty ? "empty" : "planted"}"
              title="${escapeHtml(plot.tooltip || "")}"
              aria-label="${escapeHtml(plot.tooltip || "Garden plot")}">${escapeHtml(plot.emoji || ".")}</span>
      `).join("")}
    </div>
  `).join("");
  const count = Number(data.plant_count || 0);
  const needsWater = Number(data.needs_water_count || 0);
  const wilted = Number(data.wilted_count || 0);
  const stats = count
    ? `${count} planted · ${needsWater} can water · ${wilted} wilted`
    : "No plants yet";
  return `
    <div class="skill-garden-card">
      <div class="skill-garden-head">
        <span>Garden preview</span>
        <strong>${escapeHtml(stats)}</strong>
      </div>
      <div class="skill-garden-grid" role="img" aria-label="Read-only memory garden grid">
        ${grid}
      </div>
      <p class="field-hint">Hover a plant to see its linked memory preview.</p>
    </div>
  `;
}

async function loadPaintSkillGallery() {
  const node = el("skillPaintGallery");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = `
    <div class="skill-empty-settings">
      <strong>Loading gallery</strong>
      <span>Reading recent paintings...</span>
    </div>
  `;
  const result = await api().get_paint_gallery(state.current.name, 24);
  if (!node.isConnected || state.currentSkill !== "paint") return;
  if (!result.ok) {
    node.innerHTML = `
      <div class="skill-empty-settings">
        <strong>Gallery unavailable</strong>
        <span>${escapeHtml(result.error || "Could not read paintings.")}</span>
      </div>
    `;
    return;
  }
  node.innerHTML = renderPaintSkillGallery(result);
}

function renderPaintSkillGallery(data) {
  const items = data.items || [];
  if (!items.length) {
    return `
      <div class="skill-empty-settings">
        <strong>No paintings yet</strong>
        <span>When this persona finishes a painting, it will appear here.</span>
      </div>
    `;
  }
  const cards = items.map((item) => `
    <article class="skill-paint-card" title="${escapeHtml(item.tooltip || item.title || "")}">
      <div class="skill-paint-thumb">
        ${item.image
          ? `<img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.title || "Painting")}">`
          : `<span>?</span>`}
      </div>
      <div class="skill-paint-meta">
        <strong>${escapeHtml(item.title || "Untitled")}</strong>
        <span>${escapeHtml((item.date || "").slice(0, 10) || "undated")}</span>
        ${item.caption ? `<p>${escapeHtml(item.caption)}</p>` : ""}
      </div>
    </article>
  `).join("");
  return `
    <div class="skill-paint-panel">
      <div class="skill-garden-head">
        <span>Recent paintings</span>
        <strong>${escapeHtml(String(data.total || items.length))} saved</strong>
      </div>
      <div class="skill-paint-grid">
        ${cards}
      </div>
    </div>
  `;
}

async function loadWebSearchSkillStatus() {
  const node = el("skillWebStatus");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = loadingSkillPanel("Loading web tools", "Checking local dependencies...");
  const result = await api().get_web_search_status(state.current.name);
  if (!node.isConnected || state.currentSkill !== "web_search") return;
  if (!result.ok) {
    node.innerHTML = loadingSkillPanel("Web status unavailable", result.error || "Could not check web tools.");
    return;
  }
  node.innerHTML = renderWebSearchSkillStatus(result);
}

function renderWebSearchSkillStatus(data) {
  const tools = data.tools || [];
  const rows = tools.map((tool) => `
    <div class="skill-tool-row">
      <div>
        <strong>${escapeHtml(tool.label || tool.name || "Tool")}</strong>
        <span>${escapeHtml(tool.description || "")}</span>
      </div>
      <b class="${String(tool.status || "").startsWith("ready") ? "ready" : "warn"}">${escapeHtml(tool.status || "unknown")}</b>
    </div>
  `).join("");
  const overall = data.ddgs_available && data.requests_available
    ? "All web helpers ready"
    : "Some helpers need packages";
  return `
    <div class="skill-tool-card">
      <div class="skill-garden-head">
        <span>Tool status</span>
        <strong>${escapeHtml(overall)}</strong>
      </div>
      <div class="skill-tool-list">${rows}</div>
    </div>
  `;
}

async function loadStickerSkillSummary() {
  const node = el("skillStickerSummary");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = loadingSkillPanel("Loading stickers", "Checking the sticker database...");
  const result = await api().get_sticker_summary(state.current.name);
  if (!node.isConnected || state.currentSkill !== "sticker") return;
  if (!result.ok) {
    node.innerHTML = loadingSkillPanel("Sticker status unavailable", result.error || "Could not read stickers.");
    return;
  }
  node.innerHTML = renderStickerSkillSummary(result);
}

function renderStickerSkillSummary(data) {
  const packText = (data.packs || []).length ? data.packs.join(", ") : "No packs listed";
  return `
    <div class="skill-tool-card">
      <div class="skill-garden-head">
        <span>Sticker database</span>
        <strong>${data.ready ? "Ready" : "Not built"}</strong>
      </div>
      <div class="skill-setting-summary">
        <div><span>Stickers</span><strong>${escapeHtml(String(data.count || 0))}</strong></div>
        <div><span>Embeddings</span><strong>${escapeHtml(String(data.with_embeddings || 0))}</strong></div>
      </div>
      <div class="skill-setting-summary single">
        <div><span>Available packs</span><strong>${escapeHtml(packText)}</strong></div>
      </div>
    </div>
  `;
}

async function loadScheduleSkillSummary() {
  const node = el("skillScheduleSummary");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = loadingSkillPanel("Loading reminders", "Reading active scheduled reminders...");
  const result = await api().list_reminders(state.current.name, true);
  if (!node.isConnected || state.currentSkill !== "schedule") return;
  if (!result.ok) {
    node.innerHTML = loadingSkillPanel("Reminder summary unavailable", result.error || "Could not read reminders.");
    return;
  }
  const reminders = result.items || [];
  state.scheduleOriginals = reminders.map(normalizeScheduleDraft);
  state.scheduleDrafts = reminders.map(normalizeScheduleDraft);
  renderScheduleDraftPanels();
  syncScheduleSaveButton();
}

function renderScheduleSkillSummary(data) {
  const items = data.items || [];
  const active = items.filter((item) => item.enabled && !item.completed);
  const rows = active.length ? active.map((item) => `
    <div class="skill-tool-row schedule-summary-row">
      <div>
        <strong>${escapeHtml(item.task || "Untitled reminder")}</strong>
        <span>${escapeHtml(item.when_label || "")}</span>
      </div>
      <b class="${item.priority === "urgent" ? "warn" : "ready"}">${escapeHtml(item.priority || "routine")}</b>
      <button class="task-delete-mini" type="button" data-schedule-delete="${escapeHtml(scheduleDraftKey(item))}" title="Delete reminder">×</button>
    </div>
  `).join("") : `
    <div class="skill-empty-settings">
      <strong>No active reminders</strong>
      <span>When this persona schedules reminders, they will appear here.</span>
    </div>
  `;
  return `
    <div class="skill-tool-card">
      <div class="skill-garden-head">
        <span>Reminder list</span>
        <strong>${escapeHtml(String(data.active_count || active.length || 0))} active</strong>
      </div>
      <div class="skill-setting-summary">
        <div><span>Active</span><strong>${escapeHtml(String(data.active_count || active.length || 0))}</strong></div>
        <div><span>Completed</span><strong>${escapeHtml(String(data.completed_count || 0))}</strong></div>
      </div>
      <div class="skill-tool-list task-list-scroll">${rows}</div>
    </div>
  `;
}

function renderScheduleDraftPanels() {
  const node = el("skillScheduleSummary");
  if (!node) return;
  const active = state.scheduleDrafts.filter((item) => item.enabled && !item.completed && !item.deleted);
  const completedCount = state.scheduleDrafts.filter((item) => item.completed && !item.deleted).length;
  node.innerHTML = renderScheduleSkillSummary({
    active_count: active.length,
    completed_count: completedCount,
    items: active,
  });
  wireScheduleSummaryControls(node);
  syncScheduleSaveButton();
}

function normalizeScheduleDraft(item) {
  return {
    id: item.id || "",
    temp_id: item.temp_id || "",
    task: item.task || "",
    short_task: item.short_task || item.task || "",
    priority: item.priority || "routine",
    enabled: item.enabled !== false,
    completed: Boolean(item.completed),
    schedule_type: item.schedule_type || "once",
    when_label: item.when_label || "",
    when: item.when || "",
    deleted: Boolean(item.deleted),
  };
}

function scheduleDraftKey(item) {
  return item.id ? `id:${item.id}` : `tmp:${item.temp_id || ""}`;
}

function findScheduleDraft(key) {
  return state.scheduleDrafts.find((item) => scheduleDraftKey(item) === key);
}

function scheduleDraftPayload(items) {
  return items
    .filter((item) => !item.deleted && item.enabled && !item.completed)
    .map((item) => {
      const payload = {
        task: item.task || "",
        priority: item.priority || "routine",
      };
      if (item.id) {
        payload.id = item.id;
      } else {
        payload.when = item.when || "";
      }
      return payload;
    });
}

function scheduleDraftDirty() {
  return JSON.stringify(scheduleDraftPayload(state.scheduleDrafts)) !== JSON.stringify(scheduleDraftPayload(state.scheduleOriginals));
}

function syncScheduleSaveButton() {
  const save = el("skillDoneBtn");
  if (!save || state.currentSkill !== "schedule") return;
  save.textContent = "Save";
  save.disabled = !scheduleDraftDirty();
}

function wireScheduleSummaryControls(root) {
  root.querySelectorAll("[data-schedule-delete]").forEach((button) => {
    button.addEventListener("click", () => { void deleteReminderFromSkillEditor(button.dataset.scheduleDelete); });
  });
}

function wireScheduleSkillEditor() {
  el("scheduleMode")?.addEventListener("change", syncScheduleFields);
  document.querySelectorAll(".schedule-add-btn").forEach((button) => {
    button.addEventListener("click", () => { void addReminderFromSkillEditor(); });
  });
  [
    "scheduleTaskText",
    "scheduleDate",
    "scheduleOnceTime",
    "scheduleTime",
    "scheduleMonthday",
    "scheduleYearDate",
  ].forEach((id) => {
    el(id)?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
      event.preventDefault();
      void addReminderFromSkillEditor();
    });
    el(id)?.addEventListener("input", syncSchedulePreview);
  });
  ["scheduleWeekday"].forEach((id) => {
    el(id)?.addEventListener("change", syncSchedulePreview);
  });
  syncScheduleFields();
}

function syncScheduleFields() {
  const mode = el("scheduleMode")?.value || "once";
  el("scheduleOnceFields")?.classList.toggle("hidden", mode !== "once");
  el("scheduleRecurringFields")?.classList.toggle("hidden", mode === "once");
  el("scheduleWeekdayField")?.classList.toggle("hidden", mode !== "weekly");
  el("scheduleMonthdayField")?.classList.toggle("hidden", mode !== "monthly");
  el("scheduleYearDateField")?.classList.toggle("hidden", mode !== "yearly");
  syncSchedulePreview();
}

function scheduleWhenFromFields() {
  const mode = el("scheduleMode")?.value || "once";
  if (mode === "once") {
    const dateValue = text(el("scheduleDate")?.value).trim();
    const timeValue = text(el("scheduleOnceTime")?.value).trim();
    if (!dateValue || !timeValue) return "";
    return `${dateValue} ${timeValue}`;
  }
  const timeValue = text(el("scheduleTime")?.value).trim();
  if (!timeValue) return "";
  if (mode === "daily") return `daily ${timeValue}`;
  if (mode === "weekly") return `weekly ${el("scheduleWeekday")?.value || "monday"} ${timeValue}`;
  if (mode === "monthly") return `monthly ${Number(el("scheduleMonthday")?.value || 1)} ${timeValue}`;
  if (mode === "yearly") {
    const yearly = scheduleYearMonthDay();
    if (!yearly) return "";
    return `yearly ${yearly.monthName} ${yearly.day} ${timeValue}`;
  }
  return "";
}

function scheduleYearMonthDay() {
  const dateValue = text(el("scheduleYearDate")?.value).trim();
  if (!dateValue) return null;
  const parts = dateValue.split("-");
  if (parts.length < 3) return null;
  const monthIndex = Number(parts[1]) - 1;
  const day = Number(parts[2]);
  const months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
  if (monthIndex < 0 || monthIndex >= months.length || !day) return null;
  return {
    monthName: months[monthIndex],
    monthLabel: titleCase(months[monthIndex]),
    day,
  };
}

function schedulePreviewText() {
  const mode = el("scheduleMode")?.value || "once";
  if (mode === "once") {
    const dateValue = text(el("scheduleDate")?.value).trim();
    const timeValue = text(el("scheduleOnceTime")?.value).trim();
    if (!dateValue && !timeValue) return "Choose a date and time for this one-time reminder.";
    if (!dateValue) return "Choose a calendar date.";
    if (!timeValue) return "Choose a local time.";
    return `Runs once on ${dateValue} at ${timeValue}.`;
  }
  const timeValue = text(el("scheduleTime")?.value).trim();
  if (!timeValue) return "Choose a local time for this recurring reminder.";
  if (mode === "daily") return `Repeats daily at ${timeValue}.`;
  if (mode === "weekly") {
    const day = titleCase(el("scheduleWeekday")?.value || "monday");
    return `Repeats every ${day} at ${timeValue}.`;
  }
  if (mode === "monthly") {
    const day = Number(el("scheduleMonthday")?.value || 1);
    return `Repeats monthly on day ${day} at ${timeValue}.`;
  }
  const yearly = scheduleYearMonthDay();
  if (!yearly) return "Choose the yearly month and day. The year is ignored.";
  return `Repeats every ${yearly.monthLabel} ${yearly.day} at ${timeValue}.`;
}

function titleCase(value) {
  const raw = text(value).trim();
  return raw ? raw.charAt(0).toUpperCase() + raw.slice(1) : "";
}

function syncSchedulePreview() {
  const preview = el("schedulePreview");
  if (preview) preview.textContent = schedulePreviewText();
}

async function addReminderFromSkillEditor() {
  if (!state.current) return;
  const task = text(el("scheduleTaskText")?.value).trim();
  const when = scheduleWhenFromFields();
  const priority = el("schedulePriority")?.value || "routine";
  if (!task) {
    setScheduleSkillNotice("Reminder text is required.");
    return;
  }
  if (!when) {
    const mode = el("scheduleMode")?.value || "once";
    setScheduleSkillNotice(
      mode === "once"
        ? "Reminder date and time are required."
        : mode === "yearly"
          ? "Reminder time and yearly date are required."
        : "Reminder time is required."
    );
    return;
  }
  state.scheduleDrafts.push(normalizeScheduleDraft({
    temp_id: `new-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    task,
    short_task: task,
    priority,
    enabled: true,
    completed: false,
    schedule_type: el("scheduleMode")?.value === "once" ? "once" : "recurring",
    when,
    when_label: when,
  }));
  if (el("scheduleTaskText")) el("scheduleTaskText").value = "";
  if (el("scheduleDate")) el("scheduleDate").value = "";
  if (el("scheduleOnceTime")) el("scheduleOnceTime").value = "";
  renderScheduleDraftPanels();
  syncSchedulePreview();
  setScheduleSkillNotice("Reminder staged. Press Save to write it to the database.");
}

async function deleteReminderFromSkillEditor(reminderId) {
  if (!state.current || !reminderId) return;
  const ok = await showConfirm(
    "Delete this reminder?",
    "This removes the reminder from the draft. It will not touch the database until you press Save.",
    "Delete",
    "danger"
  );
  if (!ok) return;
  const reminder = findScheduleDraft(reminderId);
  if (!reminder) return;
  reminder.deleted = true;
  renderScheduleDraftPanels();
  setScheduleSkillNotice("Reminder removed from draft. Press Save to confirm.");
}

async function saveReminderDrafts() {
  if (!state.current || state.current.name === "__base__") return false;
  if (!scheduleDraftDirty()) {
    setScheduleSkillNotice("No reminder changes to save.");
    return false;
  }
  const result = await api().save_reminders(state.current.name, scheduleDraftPayload(state.scheduleDrafts));
  if (!result.ok) {
    setScheduleSkillNotice(result.error || "Could not save reminders.", state.scheduleUndoStamp);
    return false;
  }
  state.scheduleUndoStamp = result.undo_stamp || "";
  state.scheduleOriginals = (result.items || []).map(normalizeScheduleDraft);
  state.scheduleDrafts = (result.items || []).map(normalizeScheduleDraft);
  renderScheduleDraftPanels();
  setScheduleSkillNotice(
    result.running
      ? "Reminders saved. Running persona can see them next time it lists reminders."
      : "Reminders saved.",
    state.scheduleUndoStamp
  );
  return true;
}

async function undoScheduleDbEdit() {
  if (!state.current || !state.scheduleUndoStamp) return;
  const stamp = state.scheduleUndoStamp;
  const result = await api().restore_db_before_image(state.current.name, stamp);
  if (!result.ok) {
    setScheduleSkillNotice(result.error || "Could not undo reminder edit.", stamp);
    return;
  }
  state.scheduleUndoStamp = "";
  await loadScheduleSkillSummary();
  setScheduleSkillNotice("Reminder edit undone.");
}

function setScheduleSkillNotice(message, undoStamp = "") {
  const notice = el("skillScheduleNotice");
  if (!notice) return;
  notice.innerHTML = `
    <span>${escapeHtml(message || "")}</span>
    ${undoStamp ? `<button class="notice-undo-btn" type="button">Undo</button>` : ""}`;
  notice.classList.toggle("hidden", !message);
  const button = notice.querySelector(".notice-undo-btn");
  if (button) button.addEventListener("click", () => { void undoScheduleDbEdit(); });
}

async function loadTasksSkillSummary() {
  const node = el("skillTasksSummary");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = loadingSkillPanel("Loading tasks", "Reading the current task list...");
  const result = await api().list_tasks(state.current.name, true);
  if (!node.isConnected || state.currentSkill !== "tasks") return;
  if (!result.ok) {
    node.innerHTML = loadingSkillPanel("Task summary unavailable", result.error || "Could not read tasks.");
    return;
  }
  const tasks = result.tasks || [];
  state.taskOriginals = tasks.map(normalizeTaskDraft);
  state.taskDrafts = tasks.map(normalizeTaskDraft);
  renderTaskDraftPanels();
  syncTaskSaveButton();
}

function renderTasksSkillSummary(data) {
  const recent = data.recent || [];
  const rows = recent.length ? recent.map((task) => `
    <div class="skill-tool-row task-summary-row">
      <label class="task-quick-check" title="Mark complete">
        <input type="checkbox" data-task-quick-toggle="${escapeHtml(taskDraftKey(task))}" data-completed="${task.completed ? "1" : "0"}">
        <span aria-hidden="true"></span>
      </label>
      <div>
        <strong>${escapeHtml(task.list || "Daily")}</strong>
        <span>${escapeHtml(task.short_description || task.description || "")}</span>
      </div>
      <button class="task-delete-mini" type="button" data-task-quick-delete="${escapeHtml(taskDraftKey(task))}" title="Delete task">×</button>
    </div>
  `).join("") : `
    <div class="skill-empty-settings">
      <strong>No pending tasks</strong>
      <span>When this persona tracks tasks, they will appear here.</span>
    </div>
  `;
  return `
    <div class="skill-tool-card">
      <div class="skill-garden-head">
        <span>Task list</span>
        <strong>${escapeHtml(String(data.pending || 0))} pending</strong>
      </div>
      <div class="skill-setting-summary">
        <div><span>Pending</span><strong>${escapeHtml(String(data.pending || 0))}</strong></div>
        <div><span>Completed</span><strong>${escapeHtml(String(data.completed || 0))}</strong></div>
      </div>
      <div class="skill-tool-list task-list-scroll">${rows}</div>
    </div>
  `;
}

function renderTaskDraftPanels() {
  const node = el("skillTasksSummary");
  if (!node) return;
  const pendingTasks = state.taskDrafts.filter((task) => !task.completed && !task.deleted);
  const completedCount = state.taskDrafts.filter((task) => task.completed && !task.deleted).length;
  node.innerHTML = renderTasksSkillSummary({
    pending: pendingTasks.length,
    completed: completedCount,
    recent: pendingTasks,
  });
  wireTasksSkillSummaryControls(node);
  syncTaskSaveButton();
}

function normalizeTaskDraft(task) {
  return {
    id: task.id ?? null,
    temp_id: task.temp_id || "",
    description: task.description || "",
    short_description: task.short_description || task.description || "",
    list: task.list || "Daily",
    completed: Boolean(task.completed),
    deleted: Boolean(task.deleted),
  };
}

function taskDraftKey(task) {
  return task.id !== null && task.id !== undefined && task.id !== ""
    ? `id:${task.id}`
    : `tmp:${task.temp_id || ""}`;
}

function findTaskDraft(key) {
  return state.taskDrafts.find((task) => taskDraftKey(task) === key);
}

function taskDraftPayload(tasks) {
  return tasks
    .filter((task) => !task.deleted)
    .map((task) => {
      const payload = {
        description: task.description || "",
        list: task.list || "Daily",
        completed: Boolean(task.completed),
      };
      if (task.id !== null && task.id !== undefined && task.id !== "") {
        payload.id = Number(task.id);
      }
      return payload;
    });
}

function taskDraftDirty() {
  return JSON.stringify(taskDraftPayload(state.taskDrafts)) !== JSON.stringify(taskDraftPayload(state.taskOriginals));
}

function syncTaskSaveButton() {
  const save = el("skillDoneBtn");
  if (!save || state.currentSkill !== "tasks") return;
  save.textContent = "Save";
  save.disabled = !taskDraftDirty();
}

async function loadTasksSkillEditor() {
  const node = el("skillTasksEditor");
  if (!node || !state.current || state.current.name === "__base__") return;
  node.innerHTML = renderTasksSkillEditor();
  wireTasksSkillEditor(node);
}

function renderTasksSkillEditor() {
  return `
    <div class="skill-tool-card task-editor-card">
      <div class="skill-garden-head">
        <span>Add task</span>
        <strong>Staged until Save</strong>
      </div>
      <div id="skillTaskNotice" class="notice-pill task-local-notice hidden"></div>
      <div class="task-add-row">
        <label class="task-editor-field wide" title="Add a new checklist item for this persona.">
          <span>New Task</span>
          <input id="taskNewDescription" type="text" placeholder="What should this persona track?">
        </label>
        <label class="task-editor-field" title="Optional list name. Daily is the default.">
          <span>List Name</span>
          <input id="taskNewList" type="text" value="Daily">
        </label>
        <button id="taskAddBtn" class="subtle-wide-btn task-add-btn" type="button">Add</button>
      </div>
      <p class="task-editor-hint">List name is a free-text category, not a dropdown. Use names like Daily, Pulse, Home, or leave Daily.</p>
    </div>`;
}

function wireTasksSkillEditor(root) {
  root.querySelector("#taskAddBtn")?.addEventListener("click", () => { void addTaskFromSkillEditor(); });
  ["taskNewDescription", "taskNewList"].forEach((id) => {
    root.querySelector(`#${id}`)?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
      event.preventDefault();
      void addTaskFromSkillEditor();
    });
  });
}

function wireTasksSkillSummaryControls(root) {
  root.querySelectorAll("[data-task-quick-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      void toggleTaskFromSkillEditor(button.dataset.taskQuickToggle, button.dataset.completed === "1");
    });
  });
  root.querySelectorAll("[data-task-quick-delete]").forEach((button) => {
    button.addEventListener("click", () => { void deleteTaskFromSkillEditor(button.dataset.taskQuickDelete); });
  });
}

async function addTaskFromSkillEditor() {
  if (!state.current) return;
  const description = el("taskNewDescription")?.value || "";
  const listName = el("taskNewList")?.value || "Daily";
  if (!text(description).trim()) {
    setTaskSkillNotice("Task description is required.");
    return;
  }
  state.taskDrafts.push(normalizeTaskDraft({
    temp_id: `new-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    description,
    short_description: description,
    list: text(listName).trim() || "Daily",
    completed: false,
  }));
  if (el("taskNewDescription")) el("taskNewDescription").value = "";
  if (el("taskNewList")) el("taskNewList").value = "Daily";
  renderTaskDraftPanels();
  setTaskSkillNotice("Task staged. Press Save to write it to the database.");
}

async function toggleTaskFromSkillEditor(taskKey, completed) {
  if (!state.current || !taskKey) return;
  const task = findTaskDraft(taskKey);
  if (!task) return;
  task.completed = !completed;
  renderTaskDraftPanels();
  setTaskSkillNotice(`Task ${completed ? "reopened" : "completed"} in draft. Press Save to confirm.`);
}

async function deleteTaskFromSkillEditor(taskKey) {
  if (!state.current || !taskKey) return;
  const ok = await showConfirm(
    "Delete this task?",
    "This removes the task from the draft. It will not touch the database until you press Save.",
    "Delete",
    "danger"
  );
  if (!ok) return;
  const task = findTaskDraft(taskKey);
  if (!task) return;
  task.deleted = true;
  renderTaskDraftPanels();
  setTaskSkillNotice("Task removed from draft. Press Save to confirm.");
}

async function saveTaskDrafts() {
  if (!state.current || state.current.name === "__base__") return false;
  if (!taskDraftDirty()) {
    setTaskSkillNotice("No task changes to save.");
    return false;
  }
  const result = await api().save_tasks(state.current.name, taskDraftPayload(state.taskDrafts));
  if (!result.ok) {
    setTaskSkillNotice(result.error || "Could not save tasks.", state.taskUndoStamp);
    return false;
  }
  state.taskUndoStamp = result.undo_stamp || "";
  state.taskOriginals = (result.tasks || []).map(normalizeTaskDraft);
  state.taskDrafts = (result.tasks || []).map(normalizeTaskDraft);
  renderTaskDraftPanels();
  setTaskSkillNotice(
    result.running
      ? "Tasks saved. Running persona can see them next time it lists tasks."
      : "Tasks saved.",
    state.taskUndoStamp
  );
  return true;
}

async function undoTaskDbEdit() {
  if (!state.current || !state.taskUndoStamp) return;
  const stamp = state.taskUndoStamp;
  const result = await api().restore_db_before_image(state.current.name, stamp);
  if (!result.ok) {
    setTaskSkillNotice(result.error || "Could not undo task edit.", stamp);
    return;
  }
  state.taskUndoStamp = "";
  await refreshTaskSkillPanels();
  setTaskSkillNotice("Task edit undone.");
}

async function refreshTaskSkillPanels() {
  await loadTasksSkillSummary();
  await loadTasksSkillEditor();
}

function setTaskSkillNotice(message, undoStamp = "") {
  const notice = el("skillTaskNotice");
  if (!notice) return;
  notice.innerHTML = `
    <span>${escapeHtml(message || "")}</span>
    ${undoStamp ? `<button class="notice-undo-btn" type="button">Undo</button>` : ""}`;
  notice.classList.toggle("hidden", !message);
  const button = notice.querySelector(".notice-undo-btn");
  if (button) button.addEventListener("click", () => { void undoTaskDbEdit(); });
}

function loadingSkillPanel(title, message) {
  return `
    <div class="skill-empty-settings">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(message)}</span>
    </div>
  `;
}

function renderChannelsLegacy(channels) {
  const node = el("channels");
  if (!channels?.length) {
    node.innerHTML = `<span class="channel off">No channels configured</span>`;
    return;
  }
  node.innerHTML = channels.map((channel) => (
    `<span class="channel ${channel.enabled ? "" : "off"}">${channel.enabled ? "on" : "off"} · ${escapeHtml(channel.label)}</span>`
  )).join("");
}

function renderChannels(channels) {
  const node = el("channels");
  if (!channels?.length) {
    node.innerHTML = `<span class="channel off">No channels configured</span>`;
    return;
  }
  const editable = state.current?.name && state.current.name !== "__base__";
  const tooltips = {
    telegram: "Bidirectional chat - receives messages and sends notifications via Telegram bot",
    toast: "Windows desktop notifications when the persona has something to say",
    lor: "Posts to the Local Reddit forum. Also requires the LoR skill to be enabled",
  };
  node.innerHTML = channels.map((channel) => (
    `<button class="channel channel-toggle ${channel.enabled ? "on" : ""}" type="button"
      data-channel="${escapeHtml(channel.name)}" title="${escapeHtml(tooltips[channel.name] || "Toggle channel")}"
      ${editable ? "" : "disabled"}>
      <span class="channel-dot" aria-hidden="true"></span>
      <span>${escapeHtml(channel.label)}</span>
    </button>`
  )).join("");
  node.querySelectorAll(".channel-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      if (!editable) return;
      const channel = state.current.channels.find((item) => item.name === button.dataset.channel);
      if (!channel) return;
      channel.enabled = !channel.enabled;
      renderChannels(state.current.channels);
      setDirty(hasEditableChanges());
      renderSecrets(state.current.key_status || {});
    });
  });
}

function renderCoreAnchorIndicators(statuses) {
  document.querySelectorAll("[data-core-anchor]").forEach((button) => {
    const anchor = button.dataset.coreAnchor;
    const status = statuses?.[anchor] || {};
    const hasContent = Boolean(status.has_content);
    button.classList.toggle("has-content", hasContent);
    button.classList.toggle("is-empty", !hasContent);
    if (!button.dataset.baseTitle) {
      button.dataset.baseTitle = button.getAttribute("title") || "View core anchor";
    }
    button.setAttribute(
      "title",
      `${button.dataset.baseTitle} (${hasContent ? "has notes" : "empty"})`
    );
  });
}

function providerKeyLabel(status) {
  if (status.provider_type === "custom") return status.api_key_env || "API Key";
  return PROVIDER_LABELS[status.provider_type] || status.api_key_env || "API Key";
}

function providerDisplayName(status) {
  return PROVIDER_NAMES[status.provider_type] || status.provider_type || "Provider";
}

function secretStatusLabel(source, isSet) {
  if (source === "persona") return "Set";
  if (source === "inherited") return "Inherited";
  return isSet ? "Set" : "Missing";
}

function secretStatusClass(source, isSet) {
  if (source === "persona") return "set";
  if (source === "inherited") return "inherited";
  return isSet ? "set" : "missing";
}

function renderProviderWarnings(status) {
  const box = el("providerWarnings");
  const messages = [];

  if (providerKeyMissing(status)) {
    const envVar = providerKeyEnv(status);
    if (status.provider_type === "custom" && !envVar) {
      messages.push("No API key configured - set provider.api_key_env in config.yaml");
    } else {
      messages.push(`${providerDisplayName(status)} API key not found - set ${envVar || "provider.api_key_env"} in your .env file`);
    }
  }

  if (status.telegram_enabled && !status.telegram_set) {
    messages.push("Telegram enabled but TELEGRAM_BOT_TOKEN not found in .env");
  }

  if (!messages.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }

  box.innerHTML = messages.map((message) => (
    `<div class="provider-warning">${escapeHtml(message)}</div>`
  )).join("");
  box.classList.remove("hidden");
}

function renderSecrets(status) {
  const normalized = normalizeKeyStatus(status || {});

  const isLocal = normalized.provider_type === "local";
  el("apiKeyRow").classList.toggle("hidden", isLocal);
  if (!isLocal) {
    el("apiKeyName").textContent = providerKeyLabel(normalized);
    el("apiKeyStatus").textContent = secretStatusLabel(normalized.api_key_source, normalized.api_key_set);
    el("apiKeyStatus").className = secretStatusClass(normalized.api_key_source, normalized.api_key_set);
  }

  el("telegramStatus").textContent = secretStatusLabel(normalized.telegram_source, normalized.telegram_set);
  el("telegramStatus").className = secretStatusClass(normalized.telegram_source, normalized.telegram_set);
  el("editSecretsBtn").disabled = !state.current || state.current.name === "__base__";
  renderProviderWarnings(normalized);
  renderProcessButton(state.current);
}

function secretBadgeLabel(source) {
  if (source === "persona") return "Set";
  if (source === "inherited") return "Inherited";
  return "Missing";
}

function secretPlaceholder(key) {
  const placeholders = {
    OPENROUTER_API_KEY: "sk-or-v1-...",
    OPENAI_API_KEY: "sk-proj-...",
    ANTHROPIC_API_KEY: "sk-ant-...",
    TELEGRAM_BOT_TOKEN: "123456:ABC-...",
  };
  return placeholders[key] || "paste key here";
}

function secretInputPlaceholder(secret) {
  if (secret.source === "persona") return "Paste replacement, or use Reveal";
  if (secret.source === "inherited") return "Inherited from root .env";
  return secretPlaceholder(secret.key);
}

function selectedSecretKeys(data) {
  const providerType = selectedProviderType({ provider_type: data.provider_type });
  const keys = ["TELEGRAM_BOT_TOKEN"];
  const expectedKey = EXPECTED_ENV_VARS[providerType];
  if (expectedKey) {
    keys.unshift(expectedKey);
  } else if (providerType === "custom") {
    const standardKeys = new Set(Object.values(EXPECTED_ENV_VARS).concat("TELEGRAM_BOT_TOKEN"));
    const customSecret = (data.secrets || []).find((secret) => !standardKeys.has(secret.key));
    if (customSecret) keys.unshift(customSecret.key);
  }
  return new Set(keys);
}

function updateSecretsSaveState() {
  const dirty = Array.from(state.secretsRows.values()).some((row) => row.dirty);
  el("secretsSave").disabled = !dirty;
}

async function revealSecret(row, input, button) {
  if (row.dirty) {
    const showing = input.type === "text";
    input.type = showing ? "password" : "text";
    button.textContent = showing ? "Reveal" : "Hide";
    button.title = showing ? "Reveal value" : "Hide value";
    return;
  }
  if (row.revealed) {
    input.type = "password";
    input.value = row.displayValue || "";
    row.revealed = false;
    button.textContent = "Reveal";
    button.title = "Reveal value";
    return;
  }
  if (!row.valueLoaded) {
    const result = await api().reveal_secret(state.current.name, row.key);
    if (!result.ok) {
      setNotice(result.error || "Could not reveal secret.", "warning");
      return;
    }
    row.loadedValue = result.value || "";
    row.valueLoaded = true;
  }
  input.type = "text";
  input.value = row.loadedValue;
  row.revealed = true;
  button.textContent = "Hide";
  button.title = "Hide value";
}

async function copySecretToPersona(row, input) {
  if (!row.valueLoaded) {
    const result = await api().reveal_secret(state.current.name, row.key);
    if (!result.ok) {
      setNotice(result.error || "Could not copy inherited key.", "warning");
      return;
    }
    row.loadedValue = result.value || "";
    row.valueLoaded = true;
  }
  input.value = row.loadedValue;
  input.type = "password";
  row.revealed = false;
  row.dirty = true;
  row.remove = false;
  row.value = row.loadedValue;
  row.displayValue = row.loadedValue ? "********" : "";
  updateSecretsSaveState();
}

function renderSecretsModal(data) {
  const body = el("secretsBody");
  state.secretsRows = new Map();
  const providerType = selectedProviderType({ provider_type: data.provider_type });
  el("secretsTitle").textContent = providerType === "local"
    ? "Persona Keys"
    : `${providerDisplayName({ provider_type: providerType })} Keys`;
  body.innerHTML = "";

  const visibleKeys = selectedSecretKeys(data);
  (data.secrets || []).filter((secret) => visibleKeys.has(secret.key)).forEach((secret) => {
    const rowState = {
      key: secret.key,
      source: secret.source,
      dirty: false,
      remove: false,
      revealed: false,
      valueLoaded: false,
      loadedValue: "",
      value: "",
      displayValue: "",
    };
    state.secretsRows.set(secret.key, rowState);

    const row = document.createElement("div");
    row.className = "secret-edit-row";
    row.dataset.key = secret.key;
    row.innerHTML = `
      <div class="secret-edit-head">
        <span class="secret-edit-label">${escapeHtml(secret.label || secret.key)}</span>
        <span class="secret-badge secret-badge-${escapeHtml(secret.source)}">${escapeHtml(secretBadgeLabel(secret.source))}</span>
      </div>
      <div class="secret-input-row">
        <input class="secret-input" type="password" placeholder="${escapeHtml(secretInputPlaceholder(secret))}">
      </div>
      <div class="secret-row-actions"></div>
      <div class="secret-hint"></div>
    `;
    const input = row.querySelector(".secret-input");
    const actions = row.querySelector(".secret-row-actions");
    const hint = row.querySelector(".secret-hint");

    if (secret.source === "persona") {
      hint.textContent = `Stored in this persona's .env (${secret.masked || "masked"}). Paste a new value to replace it, or remove this persona's copy. Root .env is not touched.`;
    } else if (secret.source === "inherited") {
      hint.textContent = `Inherited from root .env (${secret.masked || "masked"}). Copy to persona if you want this companion to carry its own key.`;
    } else {
      hint.textContent = "Not set. Paste a key here to save it in this persona's .env.";
    }

    input.addEventListener("input", () => {
      rowState.dirty = true;
      rowState.remove = false;
      rowState.value = input.value;
      rowState.displayValue = input.value;
      updateSecretsSaveState();
    });

    if (secret.source !== "missing") {
      const revealBtn = document.createElement("button");
      revealBtn.type = "button";
      revealBtn.className = "secret-mini-btn";
      revealBtn.textContent = "Reveal";
      revealBtn.title = "Reveal value";
      revealBtn.addEventListener("click", () => revealSecret(rowState, input, revealBtn));
      actions.appendChild(revealBtn);
    }

    if (secret.source === "inherited") {
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "secret-mini-btn";
      copyBtn.textContent = "Copy to persona";
      copyBtn.addEventListener("click", () => copySecretToPersona(rowState, input));
      actions.appendChild(copyBtn);
    }

    if (secret.source === "persona") {
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "secret-mini-btn danger";
      removeBtn.textContent = "Remove persona copy";
      removeBtn.title = "Delete this key only from the persona .env. Root .env is not touched.";
      removeBtn.addEventListener("click", () => {
        input.value = "";
        input.type = "password";
        rowState.dirty = true;
        rowState.remove = true;
        rowState.value = "";
        rowState.displayValue = "";
        hint.textContent = "This persona's copy will be removed on save. Root .env is not touched.";
        updateSecretsSaveState();
      });
      actions.appendChild(removeBtn);
    }

    body.appendChild(row);
  });
  updateSecretsSaveState();
}

async function openSecretsModal() {
  const persona = state.current?.name;
  if (!persona || persona === "__base__") return;
  const confirmed = await showConfirm(
    "Edit API Keys",
    "This can reveal sensitive values. Continue?",
    "Open",
    "secondary"
  );
  if (!confirmed) return;
  const result = await api().get_secrets(persona);
  if (!result.ok) {
    setNotice(result.error || "Could not load secrets.", "warning");
    return;
  }
  renderSecretsModal(result);
  el("secretsDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function closeSecretsModal() {
  el("secretsDialog").classList.add("hidden");
  document.body.classList.remove("modal-open");
  state.secretsRows = new Map();
}

async function saveSecrets() {
  const persona = state.current?.name;
  if (!persona || persona === "__base__") return;

  const updates = {};
  state.secretsRows.forEach((row) => {
    if (!row.dirty) return;
    updates[row.key] = row.remove ? null : row.value;
  });
  if (!Object.keys(updates).length) {
    closeSecretsModal();
    return;
  }

  const wasRunning = currentPersonaIsRunning();
  const result = await api().save_secrets(persona, updates);
  if (!result.ok) {
    setNotice(result.error || "Could not save keys.", "warning");
    return;
  }
  const status = await api().get_key_status(persona);
  state.current.key_status = status;
  renderSecrets(status);
  closeSecretsModal();
  if (result.changed && wasRunning) {
    setNotice("Keys saved. Restart the persona for changes to take effect.", "warning", NOTICE_WARNING_MS);
    setFooterNotice("Restart needed: saved keys take effect after restart.");
  } else {
    setNotice(result.changed ? "Keys saved." : "No key changes to save.", "info", NOTICE_INFO_MS);
  }
}

function renderIdentity(data) {
  const identity = data.identity || {};
  const summary = data.summary || {};
  const fields = {
    identityName: text(identity.name),
    identityUser: text(identity.user_name),
    identityModel: text(identity.model || summary.model_display),
    systemPrompt: text(identity.system_prompt),
    relationshipContext: text(identity.relationship_context),
    voiceNotes: text(identity.voice_notes),
  };
  Object.entries(fields).forEach(([id, value]) => {
    const field = el(id);
    if (!field) return;
    field.value = value;
    if (field.tagName === "TEXTAREA") {
      field.scrollTop = 0;
      field.setSelectionRange(0, 0);
    }
  });
  state.currentTraits = Array.isArray(identity.traits) ? [...identity.traits] : [];
  renderTraits();
}

function renderTraits() {
  const container = el("traits");
  const editable = Boolean(state.current?.name && state.current.name !== "__base__");
  container.innerHTML = "";

  if (!state.currentTraits.length && !editable) {
    const empty = document.createElement("span");
    empty.className = "chip";
    empty.textContent = "No traits listed";
    container.appendChild(empty);
    return;
  }

  state.currentTraits.forEach((trait, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = trait;
    if (editable) chip.addEventListener("click", () => startTraitEdit(index));
    container.appendChild(chip);
  });

  if (editable) {
    const add = document.createElement("span");
    add.className = "chip-add";
    add.textContent = "+ add";
    add.addEventListener("click", startTraitAdd);
    container.appendChild(add);
  }
}

function markDirtyIfChanged() {
  setDirty(hasEditableChanges());
}

function traitExists(text, ignoreIndex = -1) {
  const normalized = text.trim().toLowerCase();
  return state.currentTraits.some((trait, index) => (
    index !== ignoreIndex && trait.trim().toLowerCase() === normalized
  ));
}

function startTraitEdit(index) {
  if (state.current?.name === "__base__") return;
  const container = el("traits");
  const chip = container.children[index];
  if (!chip) return;

  const wrapper = document.createElement("span");
  wrapper.className = "chip-editing";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "chip-edit";
  input.value = state.currentTraits[index] || "";
  input.maxLength = 200;

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "chip-cancel";
  cancel.textContent = "×";
  cancel.title = "Cancel edit";
  cancel.addEventListener("click", () => renderTraits());

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "chip-delete";
  remove.textContent = "Delete";
  remove.title = "Delete trait";
  remove.addEventListener("click", () => {
    state.currentTraits.splice(index, 1);
    renderTraits();
    markDirtyIfChanged();
  });

  wrapper.appendChild(input);
  wrapper.appendChild(cancel);
  wrapper.appendChild(remove);
  chip.replaceWith(wrapper);
  input.focus();
  input.select();

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitTraitEdit(index, input.value);
    }
    if (e.key === "Escape") {
      renderTraits();
    }
  });
  input.addEventListener("blur", (e) => {
    if (e.relatedTarget === cancel || e.relatedTarget === remove) return;
    commitTraitEdit(index, input.value);
  });
}

function startTraitAdd() {
  if (state.current?.name === "__base__") return;
  const container = el("traits");
  const addBtn = container.querySelector(".chip-add");
  if (!addBtn) return;

  const wrapper = document.createElement("span");
  wrapper.className = "chip-editing";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "chip-edit";
  input.placeholder = "new trait";
  input.maxLength = 200;

  wrapper.appendChild(input);
  addBtn.before(wrapper);
  input.focus();

  let committed = false;
  function commit() {
    if (committed) return;
    committed = true;
    const text = input.value.trim();
    if (text && !traitExists(text)) {
      state.currentTraits.push(text);
      markDirtyIfChanged();
    }
    renderTraits();
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    }
    if (e.key === "Escape") {
      renderTraits();
    }
  });
  input.addEventListener("blur", commit);
}

function commitTraitEdit(index, value) {
  const text = value.trim();
  if (text && !traitExists(text, index)) {
    state.currentTraits[index] = text;
  } else if (!text) {
    state.currentTraits.splice(index, 1);
  }
  renderTraits();
  markDirtyIfChanged();
}

function renderHero(data) {
  const summary = data.summary || {};
  const displayName = data.display_name || summary.display_name || data.name;
  el("selectedPersonaName").textContent = displayName;
  el("displayName").textContent = displayName;
  el("personaMeta").textContent = [
    summary.model_display,
    summary.provider_type,
    summary.max_context ? `${summary.max_context} context` : "",
    data.status?.pid ? `pid ${data.status.pid}` : "",
  ].filter(Boolean).join(" · ");

  const avatar = el("avatar");
  avatar.innerHTML = "";
  if (data.paths?.avatar) {
    const img = document.createElement("img");
    img.src = data.paths.avatar;
    img.alt = "";
    avatar.appendChild(img);
  } else {
    avatar.textContent = firstLetter(displayName);
  }
}

function syncBaseUrlVisibility() {
  const isCustom = el("providerType").value === "custom";
  el("baseUrlRow").classList.toggle("hidden", !isCustom);
}

function syncAnthropicCacheVisibility() {
  const isAnthropic = el("providerType").value === "anthropic";
  el("anthropicCacheRow").classList.toggle("hidden", !isAnthropic);
}

function syncLocalServerVisibility() {
  const isLocal = el("providerType").value === "local";
  el("localServerSection").classList.toggle("hidden", !isLocal);
}

// When the user switches to a cloud provider, nudge the API Context up to a
// sane default — but only if it's blank or still at a small/local-ish value,
// so a deliberately-chosen budget is never overwritten.
function maybeBumpCloudContext() {
  if (el("providerType").value === "local") return;
  const field = el("maxContext");
  const current = parseInt(field.value, 10);
  const uncustomized = !field.value.trim() || (Number.isFinite(current) && current <= LOCALISH_CONTEXT_CEILING);
  if (!uncustomized) return;
  field.value = String(CLOUD_DEFAULT_CONTEXT);
  setNotice(
    "Set API context to 128K for this cloud provider — adjust in the Provider section if you need a different budget.",
    "info",
    NOTICE_INFO_MS
  );
}

function renderProvider(data) {
  const summary = data.summary || {};
  const tts = summary.tts || {};
  el("providerType").value = text(summary.provider_type);
  el("providerModel").value = text(summary.provider_model);
  el("providerBaseUrl").value = text(summary.base_url);
  el("maxContext").value = text(summary.provider_max_context ?? summary.max_context);
  el("anthropicCacheTtl").value = text(summary.cache_ttl, "5m");
  el("anthropicCacheAutomatic").checked = Boolean(summary.cache_automatic);
  el("anthropicCacheDiagnostics").checked = Boolean(summary.cache_diagnostics);
  syncBaseUrlVisibility();
  syncAnthropicCacheVisibility();
  syncLocalServerVisibility();
  el("ttsVoice").value = text(tts.voice_description);
  el("ttsSample").value = text(tts.voice_sample);
  el("ttsSampleText").value = text(tts.voice_sample_text);
  syncTtsMode();
}

function renderLocalServer(data) {
  const summary = data.summary || {};
  const server = summary.server || {};
  el("lsLlamaCppDir").value = text(server.llama_cpp_dir);
  el("lsModelsDir").value = text(server.models_dir);
  el("lsModelFile").value = text(summary.model_file);
  el("lsHost").value = text(server.host, "127.0.0.1");
  el("lsPort").value = text(server.port, 8012);
  el("lsGpuLayers").value = text(server.gpu_layers, -1);
  el("lsMaxContext").value = text(summary.local_max_context, 16384);
  el("lsFlashAttn").checked = server.flash_attention !== false;
  el("lsParallel").value = text(server.parallel, 1);
  el("lsMmproj").value = text(summary.mmproj_file);
}

function toggleLocalServerAdvanced() {
  const panel = el("lsAdvanced");
  const isNowHidden = panel.classList.toggle("hidden");
  el("lsAdvancedChevron").textContent = isNowHidden ? "▶" : "▼";
}

function renderHeartbeat(summary) {
  const hb = summary?.heartbeat || {};
  el("hbInterval").value = text(hb.interval_minutes);
  el("hbMin").value = text(hb.interval_min_minutes);
  el("hbMax").value = text(hb.interval_max_minutes);
  el("hbQuietStart").value = text(hb.quiet_hours_start);
  el("hbQuietEnd").value = text(hb.quiet_hours_end);
  el("hbRandomize").checked = Boolean(hb.randomize);
  el("hbDebug").checked = Boolean(hb.debug);
}

function syncTtsMode() {
  const hasSample = Boolean(el("ttsSample").value);
  const badge = el("ttsModeBadge");
  badge.textContent = hasSample ? "Clone mode" : "Design mode";
  badge.className = "tts-badge " + (hasSample ? "clone" : "design");
  el("ttsVoiceHint").textContent = hasSample
    ? "Used in design mode only."
    : "Describe the voice — pitch, tone, warmth, accent. Each generation sounds slightly different.";
  el("ttsSampleTextGroup").classList.toggle("hidden", !hasSample);
  el("ttsSampleClear").classList.toggle("hidden", !hasSample);
}

function editableSnapshot() {
  return {
    identity: {
      name: el("identityName").value,
      user_name: el("identityUser").value,
      model: el("identityModel").value,
      system_prompt: el("systemPrompt").value,
      relationship_context: fieldValue("relationshipContext"),
      voice_notes: el("voiceNotes").value,
      traits: [...state.currentTraits],
    },
    provider: {
      type: el("providerType").value,
      model: el("providerModel").value,
      base_url: el("providerBaseUrl").value,
      max_context: el("maxContext").value,
      cache_ttl: el("anthropicCacheTtl").value,
      cache_automatic: Boolean(el("anthropicCacheAutomatic").checked),
      cache_diagnostics: Boolean(el("anthropicCacheDiagnostics").checked),
    },
    model: {
      model_file: el("lsModelFile").value,
      mmproj_file: el("lsMmproj").value,
      max_context: el("lsMaxContext").value,
      temperature: readSliderValue("temperature"),
      max_response_tokens: readSliderValue("max_response_tokens"),
      frequency_penalty: readSliderValue("frequency_penalty"),
      presence_penalty: readSliderValue("presence_penalty"),
      top_p: readSliderValue("top_p"),
      top_k: el("tuningTopK")?.value || "",
      reasoning: Boolean(el("tuningReasoning")?.checked),
      reasoning_effort: el("tuningEffort")?.value || "",
      show_reasoning: Boolean(el("tuningShowReasoning")?.checked),
      max_tool_rounds: el("tuningMaxRounds")?.value || "",
    },
    server: {
      llama_cpp_dir: el("lsLlamaCppDir").value,
      models_dir: el("lsModelsDir").value,
      host: el("lsHost").value,
      port: el("lsPort").value,
      gpu_layers: el("lsGpuLayers").value,
      flash_attention: el("lsFlashAttn").checked,
      parallel: el("lsParallel").value,
    },
    context_budget: {
      recent_tail_exchanges: el("tuningTailExchanges")?.value || "",
    },
    tts: {
      voice_description: el("ttsVoice").value,
      voice_sample: el("ttsSample").value,
      voice_sample_text: el("ttsSampleText").value,
    },
    skills: Object.fromEntries(
      (state.current?.skills || []).map((skill) => [skill.name, Boolean(skill.enabled)])
    ),
    channels: Object.fromEntries(
      (state.current?.channels || []).map((channel) => {
        if (channel.name !== "lor") {
          return [channel.name, Boolean(channel.enabled)];
        }
        const lor = state.current?.config?.channels?.lor || {};
        return [channel.name, {
          enabled: Boolean(channel.enabled),
          author_name: text(lor.author_name),
          model_name: text(lor.model_name),
          context_initial_lookback_hours: lor.context_initial_lookback_hours ?? 72,
        }];
      })
    ),
    paths: {
      lor_data: lorDataPath(),
    },
    context: {
      inject_skills: contextInjectSkills(),
    },
    dev_tick: {
      enabled: Boolean(devTickValue("enabled", false)),
      interval_minutes: devTickValue("interval_minutes", 720),
      schedule_time: text(devTickValue("schedule_time", "")),
      max_rounds: devTickValue("max_rounds", 16),
    },
    heartbeat: {
      interval_minutes: el("hbInterval").value,
      interval_min_minutes: el("hbMin").value,
      interval_max_minutes: el("hbMax").value,
      quiet_hours_start: el("hbQuietStart").value,
      quiet_hours_end: el("hbQuietEnd").value,
      randomize: el("hbRandomize").checked,
      debug: el("hbDebug").checked,
    },
  };
}

function setEditableState(data) {
  const editable = data?.name && data.name !== "__base__";
  [
    "identityName",
    "identityUser",
    "identityModel",
    "systemPrompt",
    "relationshipContext",
    "voiceNotes",
    "providerModel",
    "providerBaseUrl",
    "maxContext",
    "lsLlamaCppDir",
    "lsModelsDir",
    "lsModelFile",
    "lsHost",
    "lsPort",
    "lsGpuLayers",
    "lsMaxContext",
    "lsParallel",
    "lsMmproj",
    "ttsVoice",
    "ttsSampleText",
    "hbInterval",
    "hbMin",
    "hbMax",
    "hbQuietStart",
    "hbQuietEnd",
    "tuningMaxRounds",
    "tuningTailExchanges",
  ].forEach((id) => {
    const field = el(id);
    if (field) field.readOnly = !editable;
  });
  [
    "providerType",
    "anthropicCacheTtl",
    "anthropicCacheAutomatic",
    "anthropicCacheDiagnostics",
    "lsFlashAttn",
    "tuningReasoning",
    "tuningEffort",
    "tuningShowReasoning",
    "hbDebug",
  ].forEach((id) => {
    const field = el(id);
    if (field) field.disabled = !editable;
  });
  el("ttsSample").readOnly = true;
  el("hbRandomize").disabled = !editable;
  el("ttsSamplePicker").disabled = !editable;
  el("ttsSampleClear").disabled = !editable;
  el("lsLlamaCppDirPicker").disabled = !editable;
  el("lsModelsDirPicker").disabled = !editable;
  el("lsModelFilePicker").disabled = !editable;
  el("lsMmprojPicker").disabled = !editable;
  el("lsMmprojClear").disabled = !editable;
  state.originalEditable = editableSnapshot();
  setDirty(false);
  setCanUndo(false);
}

function parseHeartbeatNumber(value) {
  return value === "" ? "" : Number(value);
}

function readSliderValue(key) {
  const row = document.querySelector(`.slider-row[data-key="${key}"]`);
  if (!row) return "";
  const raw = row.querySelector(".slider-value")?.textContent || "";
  const value = Number(raw);
  if (Number.isNaN(value)) return "";
  return row.dataset.fmt === "0" ? String(Math.round(value)) : value.toFixed(2);
}

function numberOrEmpty(value) {
  return value === "" ? "" : Number(value);
}

function collectEditableChanges() {
  const current = editableSnapshot();
  const original = state.originalEditable || { identity: {}, provider: {}, server: {}, model: {}, context_budget: {}, tts: {}, skills: {}, channels: {}, paths: {}, context: {}, dev_tick: {}, heartbeat: {} };
  const changes = { identity: {}, provider: {}, server: {}, model: {}, context_budget: {}, tts: {}, skills: {}, channels: {}, paths: {}, context: {}, dev_tick: {}, heartbeat: {} };

  for (const [key, value] of Object.entries(current.identity)) {
    if (key === "traits") {
      if (JSON.stringify(value) !== JSON.stringify(original.identity?.traits || [])) {
        changes.identity.traits = value;
      }
    } else if (value !== original.identity?.[key]) {
      changes.identity[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.provider)) {
    if (value !== original.provider?.[key]) {
      changes.provider[key] = key === "max_context" ? numberOrEmpty(value) : value;
    }
  }

  for (const [key, value] of Object.entries(current.server)) {
    if (value !== original.server?.[key]) {
      changes.server[key] = key === "flash_attention"
        ? value
        : ["llama_cpp_dir", "models_dir", "host"].includes(key)
          ? value
          : numberOrEmpty(value);
    }
  }

  for (const [key, value] of Object.entries(current.model)) {
    if (value !== original.model?.[key]) {
      changes.model[key] = ["reasoning", "show_reasoning"].includes(key)
        ? value
        : ["max_response_tokens", "max_context", "max_tool_rounds", "top_k"].includes(key)
          ? numberOrEmpty(value)
          : ["temperature", "frequency_penalty", "presence_penalty", "top_p"].includes(key)
            ? numberOrEmpty(value)
            : value;
    }
  }

  for (const [key, value] of Object.entries(current.context_budget)) {
    if (value !== original.context_budget?.[key]) {
      changes.context_budget[key] = numberOrEmpty(value);
    }
  }

  for (const [key, value] of Object.entries(current.tts)) {
    if (value !== original.tts?.[key]) {
      changes.tts[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.skills)) {
    if (value !== original.skills?.[key]) {
      changes.skills[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.channels)) {
    const originalValue = original.channels?.[key];
    if (typeof value === "object" && value !== null) {
      const channelChanges = {};
      for (const [field, fieldValue] of Object.entries(value)) {
        if (fieldValue !== originalValue?.[field]) {
          channelChanges[field] = fieldValue;
        }
      }
      if (Object.keys(channelChanges).length > 0) {
        changes.channels[key] = channelChanges;
      }
    } else if (value !== originalValue) {
      changes.channels[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.paths)) {
    if (value !== original.paths?.[key]) {
      changes.paths[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.context)) {
    const originalValue = original.context?.[key] || [];
    if (JSON.stringify(value) !== JSON.stringify(originalValue)) {
      changes.context[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.dev_tick)) {
    if (value !== original.dev_tick?.[key]) {
      changes.dev_tick[key] = key === "enabled" ? value : key === "schedule_time" ? value : numberOrEmpty(value);
    }
  }

  for (const [key, value] of Object.entries(current.heartbeat)) {
    if (value !== original.heartbeat?.[key]) {
      changes.heartbeat[key] = ["randomize", "debug"].includes(key) ? value : parseHeartbeatNumber(value);
    }
  }

  return changes;
}

function hasEditableChanges() {
  const changes = collectEditableChanges();
  return Object.keys(changes.identity).length > 0
    || Object.keys(changes.provider).length > 0
    || Object.keys(changes.server).length > 0
    || Object.keys(changes.model).length > 0
    || Object.keys(changes.context_budget).length > 0
    || Object.keys(changes.tts).length > 0
    || Object.keys(changes.skills).length > 0
    || Object.keys(changes.channels).length > 0
    || Object.keys(changes.paths).length > 0
    || Object.keys(changes.context).length > 0
    || Object.keys(changes.dev_tick).length > 0
    || Object.keys(changes.heartbeat).length > 0;
}

function validateHeartbeatFields() {
  const randomize = el("hbRandomize").checked;
  const numeric = [
    ["hbInterval", "Heartbeat interval", 1, 1440],
    ["hbQuietStart", "Quiet start", 0, 23],
    ["hbQuietEnd", "Quiet end", 0, 23],
  ];
  // Min/max interval are only used when randomization is on, so only require
  // and validate them in that case — otherwise they may legitimately be blank.
  if (randomize) {
    numeric.push(["hbMin", "Heartbeat min interval", 1, 1440]);
    numeric.push(["hbMax", "Heartbeat max interval", 1, 1440]);
  }
  for (const [id, label, min, max] of numeric) {
    const raw = el(id).value;
    if (raw === "") {
      return `${label} must be set.`;
    }
    const value = Number(raw);
    if (!Number.isInteger(value) || value < min || value > max) {
      return `${label} must be a whole number between ${min} and ${max}.`;
    }
  }
  if (randomize) {
    const hbMin = Number(el("hbMin").value);
    const hbMax = Number(el("hbMax").value);
    if (hbMin > hbMax) {
      return "Heartbeat min interval cannot be greater than max interval.";
    }
  }
  return "";
}

function validateProviderAndTuningFields() {
  const maxContext = el("maxContext").value;
  if (el("providerType").value !== "local" || maxContext !== "") {
    if (maxContext === "") return "API Context must be set.";
    const contextValue = Number(maxContext);
    if (!Number.isInteger(contextValue) || contextValue < 1024 || contextValue > 1048576) {
      return "API Context must be a whole number between 1024 and 1048576.";
    }
  }

  const numeric = [
    ["tuningMaxRounds", "Max Tool Rounds", 1, 32],
    ["tuningTailExchanges", "Recent Tail Exchanges", 1, 10],
  ];
  for (const [id, label, min, max] of numeric) {
    const raw = el(id).value;
    if (raw === "") return `${label} must be set.`;
    const value = Number(raw);
    if (!Number.isInteger(value) || value < min || value > max) {
      return `${label} must be a whole number between ${min} and ${max}.`;
    }
  }
  if (el("providerType").value === "local") {
    const serverNumeric = [
      ["lsPort", "Port", 1024, 65535],
      ["lsGpuLayers", "GPU Layers", -1, 512],
      ["lsMaxContext", "Context Size", 1024, 1048576],
      ["lsParallel", "Parallel Slots", 1, 8],
    ];
    for (const [id, label, min, max] of serverNumeric) {
      const raw = el(id).value;
      if (raw === "") return `${label} must be set.`;
      const value = Number(raw);
      if (!Number.isInteger(value) || value < min || value > max) {
        return `${label} must be a whole number between ${min} and ${max}.`;
      }
    }
  }
  return "";
}

function currentPersonaIsRunning() {
  return Boolean(state.current?.status?.running) && !state.current?.status?.stale;
}

function handleContinuityAction(action) {
  if (action === "view-lantern") {
    openLanternDialog();
    return;
  }
  if (action === "update-lantern") {
    openLanternDialog();
    return;
  }
  if (action === "browse-memories") {
    openMemoriesDialog();
    return;
  }
  if (action === "add-memory") {
    void openAddMemoryDialog();
    return;
  }
  if (action === "browse-journal") {
    openJournalDialog();
    return;
  }
  if (action === "core-self") {
    openCoreDialog("_self");
    return;
  }
  if (action === "core-user") {
    openCoreDialog("_user");
    return;
  }
  if (action === "core-relationship") {
    openCoreDialog("_relationship");
    return;
  }
  const messages = {
    "import-content": "Continuity import will start with pasted companion-written notes and preview before writing.",
    "learn-more": "Continuity planning lives in designs/continuity_section.md.",
  };
  setNotice(messages[action] || "Continuity action coming soon.", "info", NOTICE_INFO_MS);
}

async function loadPersona(name) {
  setNotice("");
  setFooterNotice("");
  const data = await api().load_persona(name);
  state.current = data;
  renderHero(data);
  renderStatus(data.status || {});
  renderIdentity(data);
  renderProvider(data);
  renderLocalServer(data);
  renderHeartbeat(data.summary || {});
  renderSecrets(data.key_status || {});
  renderTuning(data.summary || {});
  renderRuntime(data.status || {}, data.summary || {});
  renderSkills(data.skills || []);
  renderChannels(data.channels || []);
  renderCoreAnchorIndicators(data.core_anchors || {});
  renderProcessButton(data);
  setEditableState(data);
  setBackupState();
  el("filePath").textContent = data.paths?.config
    ? `Editable safe fields: ${data.paths.config}`
    : "Safe edit mode";
  state.prefs.last_persona = name;
  await api().save_prefs(state.prefs);
  await loadLogs();

  if (name === "__base__") {
    setNotice("This is the base configuration every companion inherits — it can't be started or edited here. Use “+ New” to create a companion of your own.");
  }
}

async function loadLogs() {
  if (!state.current) return;
  el("logTail").textContent = await api().get_log_tail(state.current.name, 100);
}

function slugFromDisplayName(value) {
  return text(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/[-_]{2,}/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 64);
}

function openNewPersonaDialog() {
  el("newPersonaName").value = "";
  el("newPersonaSlug").value = "";
  el("newPersonaSlug").dataset.autoSlug = "";
  el("newPersonaUser").value = state.current?.identity?.user_name || state.current?.summary?.user_name || "";
  el("newPersonaImportEnabled").checked = false;
  el("newPersonaImportPath").value = "";
  el("newPersonaImportFields").classList.add("hidden");
  el("newPersonaError").classList.add("hidden");
  el("newPersonaError").textContent = "";
  el("newPersonaCreate").disabled = false;
  el("newPersonaDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  setTimeout(() => el("newPersonaName").focus(), 0);
}

function closeNewPersonaDialog() {
  el("newPersonaDialog").classList.add("hidden");
  document.body.classList.remove("modal-open");
}

function showNewPersonaError(message) {
  const box = el("newPersonaError");
  box.textContent = message;
  box.classList.remove("hidden");
}

function syncNewPersonaSlug() {
  const nameInput = el("newPersonaName");
  const slugInput = el("newPersonaSlug");
  const previousAuto = slugInput.dataset.autoSlug || "";
  const current = slugInput.value.trim();
  if (current && current !== previousAuto) return;
  const next = slugFromDisplayName(nameInput.value);
  slugInput.value = next;
  slugInput.dataset.autoSlug = next;
}

async function submitNewPersona() {
  const displayName = el("newPersonaName").value.trim();
  const slug = el("newPersonaSlug").value.trim().toLowerCase();
  const userName = el("newPersonaUser").value.trim();
  const importPath = el("newPersonaImportEnabled").checked ? el("newPersonaImportPath").value.trim() : "";
  if (!displayName || !slug || !userName) {
    showNewPersonaError("Display name, slug, and user name are required.");
    return;
  }
  if (el("newPersonaImportEnabled").checked && !importPath) {
    showNewPersonaError("Choose an OpenRouter JSON export or turn import off.");
    return;
  }
  if (state.dirty || hasEditableChanges()) {
    const ok = await showConfirm(
      "Discard unsaved edits?",
      "Creating a persona will switch the page after it succeeds. Unsaved edits on the current persona will be discarded.",
      "Create persona",
      "secondary"
    );
    if (!ok) return;
  }

  const button = el("newPersonaCreate");
  button.disabled = true;
  showNewPersonaError("Creating persona...");
  const result = await api().create_persona({
    display_name: displayName,
    slug,
    user_name: userName,
  });
  if (!result.ok) {
    button.disabled = false;
    showNewPersonaError(result.error || "Could not create persona.");
    return;
  }

  let importResult = null;
  if (importPath) {
    showNewPersonaError("Persona created. Importing transcript...");
    importResult = await api().import_openrouter_chat(result.name || slug, importPath);
  }

  closeNewPersonaDialog();
  state.personas = await api().list_personas();
  if (result.persona && !state.personas.some((p) => p.name === result.persona.name)) {
    state.personas.push(result.persona);
  }
  renderPersonaMenu();
  await loadPersona(result.name || slug);
  if (importResult && !importResult.ok) {
    setNotice(`Created ${result.display_name || displayName}, but import failed: ${importResult.error || "unknown error"}`, "warning", NOTICE_WARNING_MS);
  } else if (importResult) {
    setNotice(
      `Created ${result.display_name || displayName}. Imported ${importResult.imported_messages} messages; skipped ${importResult.skipped_reasoning} reasoning blocks. Review setup before starting.`,
      "warning",
      NOTICE_WARNING_MS
    );
  } else {
    setNotice(`Created ${result.display_name || displayName}. Review setup before starting.`, "warning", NOTICE_WARNING_MS);
  }
}

async function pickFolder(inputId) {
  if (!state.current || state.current.name === "__base__") return;
  const result = await api().pick_folder(el(inputId).value || "");
  if (result?.ok && result.path) {
    el(inputId).value = result.path;
    setDirty(hasEditableChanges());
  } else if (result && !result.ok && result.error) {
    setNotice(result.error, "warning");
  }
}

async function pickModelFile(inputId) {
  if (!state.current || state.current.name === "__base__") return;
  const result = await api().pick_model_file(el("lsModelsDir").value || "", el(inputId).value || "");
  if (result?.ok && result.path) {
    el(inputId).value = result.path;
    setDirty(hasEditableChanges());
  } else if (result && !result.ok && result.error) {
    setNotice(result.error, "warning");
  }
}

function validateSaveFields() {
  if (el("ttsSample").value && !el("ttsSampleText").value) {
    return "Voice sample is set but transcript is empty - clone mode needs both.";
  }
  return validateHeartbeatFields() || validateProviderAndTuningFields();
}

async function previewCurrentSave() {
  if (!state.current || state.current.name === "__base__") {
    return { ok: false, error: "Choose a persona first." };
  }
  const validationError = validateSaveFields();
  if (validationError) {
    return { ok: false, error: validationError };
  }
  const changes = collectEditableChanges();
  const preview = await api().preview_persona_save(state.current.name, changes);
  if (!preview.ok) {
    return { ok: false, error: preview.error || "Could not preview changes." };
  }
  return { ok: true, changes, preview: preview.preview };
}

function savePreviewBody(preview, intro) {
  const changed = preview.changes.map((item) => item.file).join(", ");
  const diff = preview.diff || "";
  return `${intro} <strong>${escapeHtml(changed)}</strong>. A backup will be created first.`
    + (diff ? `<pre class="diff-block">${formatDiff(diff)}</pre>` : "");
}

async function applyCurrentSave(changes, { notice = true } = {}) {
  const wasRunning = currentPersonaIsRunning();
  const personaName = state.current.name;
  const result = await api().save_persona(personaName, changes);
  if (!result.ok) {
    setNotice(result.error || "Save failed.", "warning");
    return { ok: false };
  }
  await loadPersona(personaName);
  setDirty(false);
  setCanUndo(Boolean(result.changed));
  if (notice) {
    if (result.changed && wasRunning) {
      const restartMessage = "Restart needed: saved changes take effect after restart.";
      setNotice("Saved. Restart the persona for changes to take effect.", "warning", NOTICE_WARNING_MS);
      setFooterNotice(restartMessage);
    } else {
      setNotice(result.changed ? "Saved with backup." : "No changes to save.", "info", NOTICE_INFO_MS);
      setFooterNotice("");
    }
  }
  return { ok: true, changed: Boolean(result.changed), personaName };
}

async function saveCurrentPersona() {
  const prepared = await previewCurrentSave();
  if (!prepared.ok) {
    setNotice(prepared.error, "warning");
    el("notice").scrollIntoView({ behavior: "smooth", block: "center" });
    return { ok: false };
  }
  if (!prepared.preview.has_changes) {
    setDirty(false);
    setNotice("No changes to save.");
    return { ok: true, changed: false };
  }
  const ok = await showConfirm(
    "Save changes?",
    savePreviewBody(prepared.preview, "Saving to"),
    "Save",
    "secondary"
  );
  if (!ok) return { ok: false, canceled: true };
  return applyCurrentSave(prepared.changes);
}

async function saveBeforeStartChoice() {
  const prepared = await previewCurrentSave();
  if (!prepared.ok) {
    setNotice(prepared.error, "warning");
    el("notice").scrollIntoView({ behavior: "smooth", block: "center" });
    return { ok: false };
  }
  if (!prepared.preview.has_changes) {
    setDirty(false);
    return { ok: true, action: "start" };
  }
  const choice = await showChoice(
    "Save before starting?",
    savePreviewBody(prepared.preview, "Pulse starts from saved config. Saving to"),
    {
      okLabel: "Save & Start",
      okStyle: "secondary",
      secondaryLabel: "Save",
      cancelLabel: "Cancel",
    }
  );
  if (choice === "cancel") return { ok: false, canceled: true };
  const result = await applyCurrentSave(prepared.changes, { notice: choice === "secondary" });
  if (!result.ok) return { ok: false };
  return { ok: true, action: choice === "ok" ? "start" : "saved", personaName: result.personaName };
}

async function saveBeforeStopChoice() {
  const prepared = await previewCurrentSave();
  if (!prepared.ok) {
    const ok = await showConfirm(
      "Stop with unsaved changes?",
      `${escapeHtml(prepared.error)} Pulse can stop now, but these unsaved edits will be lost when the persona reloads.`,
      "Stop",
      "danger"
    );
    return { ok, action: ok ? "stop" : "cancel" };
  }
  if (!prepared.preview.has_changes) {
    setDirty(false);
    return { ok: true, action: "stop" };
  }
  const choice = await showChoice(
    "Save before stopping?",
    savePreviewBody(prepared.preview, "Stopping will reload from saved config. Saving to"),
    {
      okLabel: "Save & Stop",
      okStyle: "secondary",
      secondaryLabel: "Stop",
      cancelLabel: "Cancel",
    }
  );
  if (choice === "cancel") return { ok: false, action: "cancel" };
  if (choice === "secondary") return { ok: true, action: "stop" };
  const result = await applyCurrentSave(prepared.changes, { notice: false });
  if (!result.ok) return { ok: false, action: "cancel" };
  return { ok: true, action: "stop", personaName: result.personaName };
}

async function refreshAll() {
  state.prefs = await api().get_prefs();
  state.personas = await api().list_personas();
  el("newPersonaBtn").disabled = false;
  renderPersonaMenu();
  const preferred = state.prefs.last_persona || state.personas.find((p) => !p.is_base)?.name || "__base__";
  const exists = state.personas.some((p) => p.name === preferred);
  await loadPersona(exists ? preferred : "__base__");
}

async function refreshCurrentStatus() {
  const closeReq = await api().check_close_request();
  if (closeReq && closeReq.length) {
    showCloseDialog(closeReq);
  }
  if (!state.current) return;
  const status = await api().get_status(state.current.name);
  const prev = state.current.status || {};
  state.current.status = status;
  renderStatus(status);
  renderRuntime(status, state.current.summary || {});
  renderProcessButton(state.current);
  if (!status.running) setFooterNotice("");
  await loadLogs();

  if (state.pendingTransition === "starting") {
    if (status.phase === "running" && status.tts_ready !== false) {
      state.pendingTransition = null;
      setNotice("");
    } else if (status.running) {
      const details = [];
      if (status.phase !== "running") details.push(status.phase);
      if (status.tts_ready === false) details.push("loading TTS voice");
      setNotice(`Starting Pulse...${details.length ? " " + details.join(", ") : ""}`);
    }
  } else if (state.pendingTransition === "stopping") {
    if (!status.running) {
      state.pendingTransition = null;
      setNotice("");
    }
  }
}

function escapeHtml(value) {
  return text(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}

function formatDiff(raw) {
  return raw.split("\n").map((line) => {
    const esc = escapeHtml(line);
    if (line.startsWith("+++") || line.startsWith("---"))
      return `<span class="diff-file">${esc}</span>`;
    if (line.startsWith("@@"))
      return `<span class="diff-hunk">${esc}</span>`;
    if (line.startsWith("+"))
      return `<span class="diff-add">${esc}</span>`;
    if (line.startsWith("-"))
      return `<span class="diff-del">${esc}</span>`;
    return esc;
  }).join("\n");
}

function wireTuningControls() {
  ["tuningTopK", "tuningTailExchanges", "tuningMaxRounds"].forEach((id) => {
    const node = el(id);
    if (node) node.addEventListener("input", () => setDirty(hasEditableChanges()));
  });
  ["tuningReasoning", "tuningEffort", "tuningShowReasoning"].forEach((id) => {
    const node = el(id);
    if (node) node.addEventListener("change", () => setDirty(hasEditableChanges()));
  });
}

function wireEvents() {
  el("personaPicker").addEventListener("click", () => {
    const menu = el("personaMenu");
    menu.classList.toggle("hidden");
    el("personaPicker").classList.toggle("open", !menu.classList.contains("hidden"));
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".selector-row")) {
      el("personaMenu").classList.add("hidden");
      el("personaPicker").classList.remove("open");
    }
  });
  el("newPersonaBtn").addEventListener("click", openNewPersonaDialog);
  el("newPersonaCancel").addEventListener("click", closeNewPersonaDialog);
  el("newPersonaCreate").addEventListener("click", submitNewPersona);
  el("newPersonaName").addEventListener("input", syncNewPersonaSlug);
  el("newPersonaSlug").addEventListener("input", () => {
    el("newPersonaSlug").dataset.autoSlug = "";
  });
  el("newPersonaImportEnabled").addEventListener("change", () => {
    el("newPersonaImportFields").classList.toggle("hidden", !el("newPersonaImportEnabled").checked);
  });
  el("newPersonaImportPicker").addEventListener("click", async () => {
    const result = await api().pick_openrouter_export(el("newPersonaImportPath").value || "");
    if (result?.ok && result.path) {
      el("newPersonaImportPath").value = result.path;
      el("newPersonaImportEnabled").checked = true;
      el("newPersonaImportFields").classList.remove("hidden");
    } else if (result && !result.ok && result.error) {
      showNewPersonaError(result.error);
    }
  });
  ["newPersonaName", "newPersonaSlug", "newPersonaUser", "newPersonaImportPath"].forEach((id) => {
    el(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void submitNewPersona();
      } else if (event.key === "Escape") {
        closeNewPersonaDialog();
      }
    });
  });
  el("newPersonaDialog").addEventListener("pointerdown", (event) => {
    newPersonaBackdropPointerDown = event.target === el("newPersonaDialog");
  });
  el("newPersonaDialog").addEventListener("click", (event) => {
    if (newPersonaBackdropPointerDown && event.target === el("newPersonaDialog")) closeNewPersonaDialog();
    newPersonaBackdropPointerDown = false;
  });
  el("reloadBtn").addEventListener("click", refreshAll);
  el("openFolderBtn").addEventListener("click", async () => {
    if (!state.current) return;
    const result = await api().open_folder(state.current.name);
    if (!result.ok) setNotice(result.error || "Could not open folder.", "warning");
  });
  el("backupsBtn").addEventListener("click", openBackupsDialog);
  el("backupCloseBtn").addEventListener("click", hideBackupsDialog);
  el("backupsDialog").addEventListener("click", (e) => {
    if (e.target === el("backupsDialog")) hideBackupsDialog();
  });
  el("lanternCloseBtn").addEventListener("click", closeLanternDialog);
  el("lanternSaveBtn").addEventListener("click", saveLanternEditor);
  el("lanternDimBtn").addEventListener("click", clearLanternState);
  el("lanternDialog").addEventListener("pointerdown", (event) => {
    lanternBackdropPointerDown = event.target === el("lanternDialog");
  });
  el("lanternDialog").addEventListener("click", (e) => {
    if (lanternBackdropPointerDown && e.target === el("lanternDialog")) closeLanternDialog();
    lanternBackdropPointerDown = false;
  });
  el("memoriesCloseBtn").addEventListener("click", () => { void hideMemoriesDialog(); });
  el("memoriesDialog").addEventListener("pointerdown", (event) => {
    memoriesBackdropPointerDown = event.target === el("memoriesDialog");
  });
  el("memoriesDialog").addEventListener("click", (event) => {
    if (memoriesBackdropPointerDown && event.target === el("memoriesDialog")) void hideMemoriesDialog();
    memoriesBackdropPointerDown = false;
  });
  el("memoryLoadMoreBtn").addEventListener("click", () => loadMemoryPage(state.memoryBrowse.page + 1));
  el("memoryViewTabs").querySelectorAll("[data-memory-view]").forEach((button) => {
    button.addEventListener("click", () => { void setMemoryView(button.dataset.memoryView); });
  });
  el("memoryTypeTabs").querySelectorAll("[data-memory-kind]").forEach((button) => {
    button.addEventListener("click", () => { void setMemoryKind(button.dataset.memoryKind); });
  });
  el("journalCloseBtn").addEventListener("click", () => { void hideJournalDialog(); });
  el("journalDialog").addEventListener("pointerdown", (event) => {
    journalBackdropPointerDown = event.target === el("journalDialog");
  });
  el("journalDialog").addEventListener("click", (event) => {
    if (journalBackdropPointerDown && event.target === el("journalDialog")) void hideJournalDialog();
    journalBackdropPointerDown = false;
  });
  el("journalLoadMoreBtn").addEventListener("click", () => loadJournalPage(state.journalBrowse.page + 1));
  el("journalViewTabs").querySelectorAll("[data-journal-view]").forEach((button) => {
    button.addEventListener("click", () => { void setJournalView(button.dataset.journalView); });
  });
  el("journalTypeTabs").querySelectorAll("[data-journal-type]").forEach((button) => {
    button.addEventListener("click", () => { void setJournalType(button.dataset.journalType); });
  });
  el("coreCloseBtn").addEventListener("click", closeCoreDialog);
  el("coreSaveBtn").addEventListener("click", saveCoreAnchor);
  el("coreDialog").addEventListener("pointerdown", (event) => {
    coreBackdropPointerDown = event.target === el("coreDialog");
  });
  el("coreDialog").addEventListener("click", (event) => {
    if (coreBackdropPointerDown && event.target === el("coreDialog")) closeCoreDialog();
    coreBackdropPointerDown = false;
  });
  el("skillDoneBtn").addEventListener("click", () => { void handleSkillDone(); });
  el("skillCloseBtn").addEventListener("click", () => { void handleSkillClose(); });
  el("skillDialog").addEventListener("pointerdown", (event) => {
    skillBackdropPointerDown = event.target === el("skillDialog");
  });
  el("skillDialog").addEventListener("click", (event) => {
    if (skillBackdropPointerDown && event.target === el("skillDialog")) void handleSkillClose();
    skillBackdropPointerDown = false;
  });
  document.querySelectorAll(".section-header").forEach((header) => {
    header.addEventListener("click", () => header.parentElement.classList.toggle("open"));
  });
  document.querySelectorAll("[data-continuity-action]").forEach((button) => {
    button.addEventListener("click", () => handleContinuityAction(button.dataset.continuityAction));
  });
  [
    "identityName",
    "identityUser",
    "identityModel",
    "systemPrompt",
    "relationshipContext",
    "voiceNotes",
    "providerModel",
    "providerBaseUrl",
    "maxContext",
    "lsLlamaCppDir",
    "lsModelsDir",
    "lsModelFile",
    "lsHost",
    "lsPort",
    "lsGpuLayers",
    "lsMaxContext",
    "lsParallel",
    "lsMmproj",
    "ttsVoice",
    "ttsSampleText",
    "hbInterval",
    "hbMin",
    "hbMax",
    "hbQuietStart",
    "hbQuietEnd",
  ].forEach((id) => {
    el(id)?.addEventListener("input", () => setDirty(hasEditableChanges()));
  });
  ["anthropicCacheTtl", "anthropicCacheAutomatic", "anthropicCacheDiagnostics"].forEach((id) => {
    el(id)?.addEventListener("change", () => setDirty(hasEditableChanges()));
  });
  el("providerType").addEventListener("change", () => {
    maybeBumpCloudContext();
    syncBaseUrlVisibility();
    syncAnthropicCacheVisibility();
    syncLocalServerVisibility();
    syncTuningProviderHints();
    setDirty(hasEditableChanges());
    renderSecrets(state.current?.key_status || {});
  });
  el("lsFlashAttn").addEventListener("change", () => setDirty(hasEditableChanges()));
  el("lsLlamaCppDirPicker").addEventListener("click", () => pickFolder("lsLlamaCppDir"));
  el("lsModelsDirPicker").addEventListener("click", () => pickFolder("lsModelsDir"));
  el("lsModelFilePicker").addEventListener("click", () => pickModelFile("lsModelFile"));
  el("lsMmprojPicker").addEventListener("click", () => pickModelFile("lsMmproj"));
  el("lsMmprojClear").addEventListener("click", () => {
    if (!el("lsMmproj").value) return;
    el("lsMmproj").value = "";
    setDirty(hasEditableChanges());
  });
  el("lsAdvancedToggle").addEventListener("click", toggleLocalServerAdvanced);
  el("lsAdvancedToggle").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleLocalServerAdvanced();
    }
  });
  el("hbRandomize").addEventListener("change", () => setDirty(hasEditableChanges()));
  el("hbDebug").addEventListener("change", () => setDirty(hasEditableChanges()));
  el("ttsSamplePicker").addEventListener("click", async () => {
    if (!state.current || state.current.name === "__base__") return;
    const result = await api().pick_voice_sample(state.current.name, el("ttsSample").value || "");
    if (result?.ok && result.path) {
      el("ttsSample").value = result.path;
      syncTtsMode();
      setDirty(hasEditableChanges());
    } else if (result && !result.ok && result.error) {
      setNotice(result.error, "warning");
    }
  });
  el("ttsSampleClear").addEventListener("click", async () => {
    if (!el("ttsSample").value) return;
    const ok = await showConfirm(
      "Switch to design mode?",
      "This will remove the voice sample and transcript. The voice will be generated from the description instead.",
      "Remove sample"
    );
    if (!ok) return;
    el("ttsSample").value = "";
    el("ttsSampleText").value = "";
    syncTtsMode();
    setDirty(hasEditableChanges());
  });
  el("saveBtn").addEventListener("click", async () => {
    await saveCurrentPersona();
  });
  el("editSecretsBtn").addEventListener("click", openSecretsModal);
  el("secretsSave").addEventListener("click", saveSecrets);
  el("secretsCancel").addEventListener("click", closeSecretsModal);
  el("secretsDialog").addEventListener("pointerdown", (event) => {
    secretsBackdropPointerDown = event.target === el("secretsDialog");
  });
  el("secretsDialog").addEventListener("click", (event) => {
    if (secretsBackdropPointerDown && event.target === el("secretsDialog")) closeSecretsModal();
    secretsBackdropPointerDown = false;
  });
  el("confirmDialog").addEventListener("pointerdown", (event) => {
    confirmBackdropPointerDown = event.target === el("confirmDialog");
  });
  el("undoBtn").addEventListener("click", async () => {
    if (!state.current || state.current.name === "__base__") return;
    const ok = await showConfirm(
      "Undo last save?",
      "This will restore the newest backup for this persona. A safety backup will be created first.",
      "Undo",
      "secondary"
    );
    if (!ok) return;
    const wasRunning = currentPersonaIsRunning();
    const result = await api().restore_last_backup(state.current.name);
    if (!result.ok) {
      setNotice(result.error || "Undo failed.", "warning");
      return;
    }
    await loadPersona(state.current.name);
    setCanUndo(false);
    if (result.changed && wasRunning) {
      setNotice("Restored. Restart the persona for changes to take effect.", "warning", NOTICE_WARNING_MS);
      setFooterNotice("Restart needed: restored changes take effect after restart.");
    } else {
      setNotice(result.changed ? "Restored latest backup." : "Nothing to restore.", "info", NOTICE_INFO_MS);
      setFooterNotice("");
    }
  });
  el("pulseToggle").addEventListener("click", async () => {
    if (!state.current || state.current.name === "__base__") return;
    const status = state.current.status || {};
    const active = Boolean(status.running) && !status.stale;

    if (active) {
      if (state.dirty || hasEditableChanges()) {
        const choice = await saveBeforeStopChoice();
        if (!choice.ok) return;
      }
      setNotice("Requesting graceful shutdown...");
      setFooterNotice("");
      state.pendingTransition = "stopping";
      const result = await api().stop_pulse(state.current.name);
      if (!result.ok) {
        state.pendingTransition = null;
        setNotice(result.error || "Failed to stop Pulse.", "warning");
        return;
      }
      state.current.status = result.status || {};
    } else {
      if (state.dirty || hasEditableChanges()) {
        const choice = await saveBeforeStartChoice();
        if (!choice.ok || choice.action === "saved") return;
      }
      setNotice("Starting Pulse...");
      setFooterNotice("");
      state.pendingTransition = "starting";
      const result = await api().start_pulse(state.current.name);
      if (!result.ok) {
        state.pendingTransition = null;
        setNotice(result.error || "Failed to start Pulse.", "warning");
        return;
      }
      state.current.status = result.status || {};
    }
    renderStatus(state.current.status);
    renderRuntime(state.current.status, state.current.summary || {});
    renderProcessButton(state.current);
  });
}

async function openBackupsDialog() {
  if (!state.current || state.current.name === "__base__") return;
  const list = el("backupList");
  list.innerHTML = `<div class="backup-empty">Loading backups...</div>`;
  el("backupsDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");

  const result = await api().list_backups(state.current.name);
  if (!result.ok) {
    list.innerHTML = `<div class="backup-empty">${escapeHtml(result.error || "Could not list backups.")}</div>`;
    return;
  }
  const backups = result.backups || [];
  if (!backups.length) {
    list.innerHTML = `<div class="backup-empty">No backups found for this persona.</div>`;
    return;
  }
  list.innerHTML = backups.map((backup) => {
    const files = Array.isArray(backup.files) ? backup.files.join(", ") : "";
    const reason = humanizeReason(backup.reason);
    return `<div class="backup-item">
      <div class="backup-info">
        <strong>${escapeHtml(formatBackupTime(backup.created_at))}</strong>
        <span>${escapeHtml(reason)}${files ? " · " + escapeHtml(files) : ""}</span>
      </div>
      <button class="backup-restore-btn" type="button"
        data-path="${escapeHtml(backup.path)}">Restore</button>
    </div>`;
  }).join("");
  list.querySelectorAll(".backup-restore-btn").forEach((button) => {
    button.addEventListener("click", () => restoreBackup(button.dataset.path));
  });
}

function hideBackupsDialog() {
  el("backupsDialog").classList.add("hidden");
  document.body.classList.remove("modal-open");
}

async function openLanternDialog() {
  if (!state.current || state.current.name === "__base__") {
    setNotice("Choose a persona first.", "warning", NOTICE_WARNING_MS);
    return;
  }
  el("lanternSubtitle").classList.remove("hidden");
  el("lanternSubtitle").textContent = "Loading lantern...";
  el("lanternStatePill").textContent = "Loading";
  el("lanternStatePill").dataset.state = "loading";
  el("lanternBody").innerHTML = `<div class="lantern-empty">Reading lantern...</div>`;
  el("lanternDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  setLanternActionButtons("edit");

  const result = await api().get_lantern(state.current.name);
  if (!result.ok) {
    el("lanternSubtitle").textContent = "Could not read lantern.";
    el("lanternStatePill").textContent = "Error";
    el("lanternStatePill").dataset.state = "error";
    el("lanternBody").innerHTML = `<div class="lantern-empty">${escapeHtml(result.error || "Lantern unavailable.")}</div>`;
    return;
  }
  renderLanternEditor(result);
}

function renderLanternEditor(data) {
  state.currentLantern = data;
  setLanternActionButtons("edit");
  el("lanternSubtitle").classList.remove("hidden");
  const persona = data.resident_id || data.persona || state.current?.display_name || "persona";
  if (!data.exists) {
    el("lanternSubtitle").textContent = `No lantern set for ${persona} yet — set the current-state fields below.`;
    el("lanternStatePill").textContent = "Not set";
    el("lanternStatePill").dataset.state = "missing";
  } else {
    const uiState = lanternUiState(data);
    el("lanternSubtitle").textContent = `${persona} · ${data.age_label || "unknown age"} · updated ${data.updated_at_display || data.updated_at || "unknown"}`;
    el("lanternStatePill").textContent = uiState.label;
    el("lanternStatePill").dataset.state = uiState.state;
  }

  const fields = data.fields || {};
  state.lanternBaseline = {
    mode: fields.mode || "",
    mood: fields.mood || "",
    focus: fields.focus || "",
    open_thread: fields.open_thread || "",
    note: fields.note || "",
  };
  el("lanternBody").innerHTML = `
    <div id="lanternLocalNotice" class="notice-pill"></div>
    <div class="lantern-edit-grid">
      ${lanternInput("mode", "Mode", fields.mode)}
      ${lanternInput("mood", "Mood", fields.mood)}
      ${lanternInput("focus", "Focus", fields.focus)}
      ${lanternInput("open_thread", "Open Thread", fields.open_thread)}
      ${lanternInput("note", "Note", fields.note, true)}
    </div>`;
  // The notice is always present from open so dimming/saving only swaps its
  // text — the modal never changes height (matches the Core editor).
  if (data.expired) {
    setLanternLocalNotice("This lantern is expired. Treat it as historical, not current-state context.");
  } else if (data.stale) {
    setLanternLocalNotice("This lantern is older than 24 hours. Verify before treating it as current.");
  } else if (lanternUiState(data).state === "aging") {
    setLanternLocalNotice("This lantern is getting old. Verify before leaning on it.");
  } else {
    setLanternLocalNotice("Edit any current-state field, then save.");
  }
}

function lanternUiState(data) {
  if (data?.state === "expired" || data?.expired) return { label: "Expired", state: "expired" };
  if (data?.state === "stale" || data?.stale) return { label: "Stale", state: "stale" };
  const age = Number(data?.age_hours);
  if (!Number.isFinite(age)) return { label: "Current", state: "current" };
  if (age < 6) return { label: "Fresh", state: "fresh" };
  if (age >= 18) return { label: "Aging", state: "aging" };
  return { label: "Current", state: "current" };
}

function hideLanternDialog() {
  el("lanternDialog").classList.add("hidden");
  state.currentLantern = null;
  state.lanternBaseline = null;
  document.body.classList.remove("modal-open");
}

async function closeLanternDialog() {
  if (lanternIsDirty()) {
    const ok = await showConfirm(
      "Discard unsaved lantern edits?",
      "You have unsaved changes to this lantern. Closing now will discard them.",
      "Discard",
      "secondary"
    );
    if (!ok) return;
  }
  hideLanternDialog();
}

function setLanternActionButtons(mode) {
  const editing = mode === "edit";
  el("lanternDimBtn").classList.toggle("hidden", !editing);
  el("lanternSaveBtn").classList.toggle("hidden", !editing);
}

function lanternCurrentValues() {
  const values = {};
  ["mode", "mood", "focus", "open_thread", "note"].forEach((key) => {
    const node = el(`lanternField-${key}`);
    if (node) values[key] = node.value;
  });
  return values;
}

function lanternIsDirty() {
  if (!state.lanternBaseline || !el("lanternField-mode")) return false;
  const baseline = state.lanternBaseline;
  const current = lanternCurrentValues();
  return ["mode", "mood", "focus", "open_thread", "note"].some(
    (key) => (baseline[key] || "") !== (current[key] || "")
  );
}

function lanternInput(name, label, value, multiline = false) {
  const tag = multiline
    ? `<textarea id="lanternField-${escapeHtml(name)}" rows="4">${escapeHtml(value || "")}</textarea>`
    : `<input id="lanternField-${escapeHtml(name)}" type="text" value="${escapeHtml(value || "")}">`;
  return `
    <label class="lantern-edit-field">
      <span>${escapeHtml(label)}</span>
      ${tag}
    </label>`;
}

async function saveLanternEditor() {
  if (!el("lanternField-mode")) return;
  const fields = lanternCurrentValues();
  const result = await api().set_lantern(state.current.name, fields);
  if (!result.ok) {
    setLanternLocalNotice(result.error || "Could not update lantern.");
    return;
  }
  if (!result.changed) {
    setLanternLocalNotice("No lantern changes to save.");
    return;
  }
  renderLanternEditor(result.lantern || await api().get_lantern(state.current.name));
  setNotice(
    result.running
      ? "Lantern updated. Running persona may pick it up on the next turn or heartbeat."
      : "Lantern updated.",
    result.running ? "warning" : "info",
    NOTICE_WARNING_MS
  );
}

// Dim / Clear: empties the active-state fields in place, keeping only the note.
// No view swap and no resize — the user reviews and then Saves like any edit.
function clearLanternState() {
  if (!el("lanternField-mode")) return;
  const activeKeys = ["mode", "mood", "focus", "open_thread"];
  const current = lanternCurrentValues();
  if (!activeKeys.some((key) => (current[key] || "").trim())) {
    setLanternLocalNotice("Lantern is already dimmed — no active state to clear.");
    return;
  }
  activeKeys.forEach((key) => {
    const node = el(`lanternField-${key}`);
    if (node) node.value = "";
  });
  const note = el("lanternField-note");
  if (note && !note.value.trim()) {
    note.value = "Lantern dimmed; no active state set.";
  }
  setLanternLocalNotice(
    "Cleared mode, mood, focus, and open thread — your note is kept. Save to dim the lantern."
  );
  if (note) {
    note.focus();
    note.setSelectionRange(note.value.length, note.value.length);
  }
}

function setLanternLocalNotice(message) {
  const node = el("lanternLocalNotice");
  if (!node) return;
  node.textContent = message;
  node.classList.toggle("hidden", !message);
}

async function openMemoriesDialog() {
  if (!state.current || state.current.name === "__base__") {
    setNotice("Choose a persona first.", "warning", NOTICE_WARNING_MS);
    return;
  }
  state.memoryBrowse = { view: "active", kind: "fact", page: 0, pageSize: 25, total: 0, hasMore: false, items: [] };
  state.memoryDetailId = "";
  state.memoryBaselines = {};
  state.memoryUndoStamp = "";
  setMemoryNotice("Select a memory to review or edit. Journal index memories are edited from Journal.");
  el("memoriesDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  syncMemoryTabs();
  syncMemoryTypeTabs();
  showMemoryList();
  await loadMemoryPage(1);
}

async function openAddMemoryDialog() {
  if (!state.current || state.current.name === "__base__") {
    setNotice("Choose a persona first.", "warning", NOTICE_WARNING_MS);
    return;
  }
  state.memoryBrowse = { view: "active", kind: "fact", page: 0, pageSize: 25, total: 0, hasMore: false, items: [] };
  state.memoryUndoStamp = "";
  el("memoriesDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  syncMemoryTabs();
  syncMemoryTypeTabs();
  showAddMemoryForm();
}

async function hideMemoriesDialog() {
  if (!(await confirmDiscardMemoryEdits())) return;
  el("memoriesDialog").classList.add("hidden");
  state.memoryDetailId = "";
  state.memoryBaselines = {};
  state.memoryUndoStamp = "";
  document.body.classList.remove("modal-open");
}

async function setMemoryView(view) {
  if (!view || view === state.memoryBrowse.view) return;
  if (!(await confirmDiscardMemoryEdits())) return;
  state.memoryBrowse = { ...state.memoryBrowse, view, page: 0, total: 0, hasMore: false, items: [] };
  syncMemoryTabs();
  showMemoryList();
  await loadMemoryPage(1);
}

async function setMemoryKind(kind) {
  if (!kind || kind === state.memoryBrowse.kind) return;
  if (!(await confirmDiscardMemoryEdits())) return;
  state.memoryBrowse = { ...state.memoryBrowse, kind, page: 0, total: 0, hasMore: false, items: [] };
  syncMemoryTypeTabs();
  showMemoryList();
  await loadMemoryPage(1);
}

function syncMemoryTabs() {
  el("memoryViewTabs").querySelectorAll("[data-memory-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.memoryView === state.memoryBrowse.view);
  });
}

function syncMemoryTypeTabs() {
  el("memoryTypeTabs").querySelectorAll("[data-memory-kind]").forEach((button) => {
    button.classList.toggle("active", button.dataset.memoryKind === state.memoryBrowse.kind);
  });
}

function showMemoryList() {
  state.memoryDetailId = "";
  state.memoryBaselines = {};
  el("memoryViewTabs").classList.remove("hidden");
  el("memoryTypeTabs").classList.remove("hidden");
  el("memoryDetail").classList.add("hidden");
  el("memoryDetail").innerHTML = "";
  el("memoriesBody").classList.remove("hidden");
  renderMemoryList();
}

function showAddMemoryForm() {
  state.memoryDetailId = "__add__";
  state.memoryBaselines = { new: memoryBaseline(defaultMemoryDraft()) };
  el("memoryViewTabs").classList.add("hidden");
  el("memoryTypeTabs").classList.add("hidden");
  el("memoryDetail").classList.remove("hidden");
  el("memoriesBody").classList.add("hidden");
  el("memoryLoadMoreBtn").classList.add("hidden");
  el("memoriesSubtitle").textContent = "Add a new fact memory.";
  el("memoryDetail").innerHTML = `
    <button class="memory-back-btn" type="button">&larr; Back to memories</button>
    <div class="memory-detail-head">
      <h3>Add Memory</h3>
      <span>Fact / user-defined</span>
    </div>
    <div class="memory-versions">
      ${renderMemoryEditor(defaultMemoryDraft(), true, "", "add")}
    </div>`;
  el("memoryDetail").querySelector(".memory-back-btn").addEventListener("click", () => { void cancelAddMemory(); });
  wireMemoryEditorEvents(el("memoryDetail"));
  setMemoryNotice("Write a fact, correction, or useful continuity note, then add it.");
}

async function loadMemoryPage(page) {
  if (!state.current || state.current.name === "__base__") return;
  if (page === 1) {
    el("memoriesBody").innerHTML = `<div class="memory-empty">Loading memories...</div>`;
    el("memoryLoadMoreBtn").classList.add("hidden");
  }
  const result = await api().list_memories(
    state.current.name,
    state.memoryBrowse.view,
    state.memoryBrowse.kind,
    page,
    state.memoryBrowse.pageSize
  );
  if (!result.ok) {
    el("memoriesSubtitle").textContent = "Could not read memories.";
    el("memoriesBody").innerHTML = `<div class="memory-empty">${escapeHtml(result.error || "Memory browse unavailable.")}</div>`;
    return;
  }
  state.memoryBrowse.page = result.page || page;
  state.memoryBrowse.total = result.total || 0;
  state.memoryBrowse.hasMore = Boolean(result.has_more);
  state.memoryBrowse.shared = Boolean(result.shared);
  state.memoryBrowse.items = page === 1
    ? (result.items || [])
    : state.memoryBrowse.items.concat(result.items || []);
  renderMemoryList(result.db_path || "");
}

function renderMemoryList(dbPath = "") {
  const browse = state.memoryBrowse;
  const viewLabel = browse.view === "archived" ? "Archived" : "Active";
  const kindLabel = humanizeMemoryKind(browse.kind);
  el("memoriesSubtitle").textContent = `${viewLabel} ${kindLabel.toLowerCase()} / showing ${browse.items.length} of ${browse.total}${browse.shared ? " / shared pool" : ""}`;
  if (!browse.items.length) {
    el("memoriesBody").innerHTML = `
      <div class="memory-empty">
        <strong>No ${browse.view} ${kindLabel.toLowerCase()} found.</strong>
        <span>${escapeHtml(emptyMemoryHint(browse.view, browse.kind))}</span>
      </div>`;
    el("memoryLoadMoreBtn").classList.add("hidden");
    return;
  }
  el("memoriesBody").innerHTML = browse.items.map(renderMemoryCard).join("");
  el("memoriesBody").querySelectorAll(".memory-card").forEach((card) => {
    card.addEventListener("click", () => { void openMemoryDetail(card.dataset.memoryId); });
  });
  const loadMore = el("memoryLoadMoreBtn");
  loadMore.classList.toggle("hidden", !browse.hasMore);
  loadMore.textContent = `Load More (${browse.items.length}/${browse.total})`;
  loadMore.dataset.dbPath = dbPath;
}

function renderMemoryCard(memory) {
  const status = memory.status || "legacy";
  const tags = (memory.tags || []).slice(0, 4).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
  const history = memory.has_history
    ? `<span class="memory-history-pill">${escapeHtml(memory.version_count)} versions</span>`
    : "";
  const badges = [
    memory.type,
    status,
    memory.confidence,
    memory.source,
    memory.time_sensitive ? "time-sensitive" : "",
  ].filter(Boolean);
  return `
    <button class="memory-card ${status === "archived" ? "archived" : ""}" type="button" data-memory-id="${escapeHtml(memory.id)}">
      <div class="memory-card-head">
        <strong>#${escapeHtml(memory.id)}</strong>
        <span>${escapeHtml(memory.date_display || "unknown")} / ${escapeHtml(memory.age_label || "unknown age")}</span>
        ${history}
      </div>
      <p>${escapeHtml(memory.preview || memory.text || "")}</p>
      <div class="memory-card-foot">
        <div class="memory-tags">${tags || "<span>untagged</span>"}</div>
        <div class="memory-badges">${badges.map((badge) => `<span>${escapeHtml(humanizeMemoryBadge(badge))}</span>`).join("")}</div>
      </div>
    </button>`;
}

async function openMemoryDetail(memoryId, force = false) {
  if (!state.current || !memoryId) return;
  if (!force && state.memoryDetailId && state.memoryDetailId !== String(memoryId) && !(await confirmDiscardMemoryEdits())) return;
  state.memoryDetailId = String(memoryId);
  state.memoryBaselines = {};
  el("memoryDetail").classList.remove("hidden");
  el("memoriesBody").classList.add("hidden");
  el("memoryLoadMoreBtn").classList.add("hidden");
  el("memoryDetail").innerHTML = `<div class="memory-empty">Loading memory #${escapeHtml(memoryId)}...</div>`;

  const result = await api().get_memory_detail(state.current.name, memoryId);
  if (!result.ok) {
    el("memoryDetail").innerHTML = `<div class="memory-empty">${escapeHtml(result.error || "Memory unavailable.")}</div>`;
    return;
  }
  const versions = result.versions || [];
  versions.forEach((memory) => {
    if (memoryIsEditable(memory)) state.memoryBaselines[String(memory.id)] = memoryBaseline(memory);
  });
  el("memoryDetail").innerHTML = `
    <button class="memory-back-btn" type="button">&larr; Back to memories</button>
    <div class="memory-detail-head">
      <h3>Memory #${escapeHtml(result.current_id || memoryId)}</h3>
      <span>${versions.length} ${versions.length === 1 ? "version" : "versions"}</span>
    </div>
    <div class="memory-versions">
      ${versions.map((memory, index) => renderMemoryVersion(memory, index === 0)).join("")}
    </div>`;
  el("memoryDetail").querySelector(".memory-back-btn").addEventListener("click", async () => {
    if (await confirmDiscardMemoryEdits()) showMemoryList();
  });
  wireMemoryEditorEvents(el("memoryDetail"));
}

function renderMemoryVersion(memory, current) {
  const sourceNote = renderMemoryDetailSourceNote(memory);
  if (memoryIsEditable(memory)) return renderMemoryEditor(memory, current, sourceNote);
  const tags = (memory.tags || []).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
  return `
    <article class="memory-version ${current ? "current" : ""}">
      <div class="memory-version-head">
        <strong>#${escapeHtml(memory.id)}${current ? " / Current" : ""}</strong>
        <span>${escapeHtml(memory.date_display || "unknown")} / ${escapeHtml(memory.age_label || "unknown age")}</span>
      </div>
      ${sourceNote}
      <p>${escapeHtml(memory.text || "")}</p>
      <div class="memory-tags">${tags || "<span>untagged</span>"}</div>
    </article>`;
}

function renderMemoryEditor(memory, current, sourceNote = "", mode = "edit") {
  const baseline = memoryBaseline(memory);
  const status = selectValue(baseline.status, MEMORY_STATUSES, "current");
  const statusLocked = mode !== "add" && status === "superseded";
  const confidence = selectValue(baseline.confidence, MEMORY_CONFIDENCES, "medium");
  const source = selectValue(baseline.source, MEMORY_SOURCES, "model_extracted");
  const editorId = mode === "add" ? "new" : String(memory.id);
  const title = mode === "add"
    ? "New memory"
    : `#${escapeHtml(memory.id)}${current ? " / Current" : ""}`;
  const meta = mode === "add"
    ? "Fact / user-defined"
    : `${escapeHtml(memory.date_display || "unknown")} / ${escapeHtml(memory.age_label || "unknown age")}`;
  return `
    <article class="memory-version memory-editor-card ${current ? "current" : ""}">
      <div class="memory-version-head">
        <strong>${title}</strong>
        <span>${meta}</span>
      </div>
      ${sourceNote}
      <div id="memoryEditor-${escapeHtml(editorId)}" class="memory-editor-grid" data-memory-editor="${escapeHtml(editorId)}">
        <label class="memory-editor-field wide" title="The fact, correction, or continuity note Pulse should remember.">
          <span>Memory</span>
          <textarea data-field="text" rows="8">${escapeHtml(baseline.text)}</textarea>
        </label>
        <label class="memory-editor-field" title="Comma-separated labels used for filtering, search hints, and context organization.">
          <span>Tags</span>
          <input data-field="tags" type="text" value="${escapeHtml(baseline.tags)}" placeholder="comma, separated, tags">
        </label>
        <label class="memory-editor-field" title="1-10. Higher means more likely to be surfaced in context and search.">
          <span>Importance</span>
          <input data-field="importance" type="number" min="1" max="10" value="${escapeHtml(baseline.importance)}">
        </label>
        <label class="memory-editor-field" title="Optional expiry date for time-sensitive memories. Blank means no expiry.">
          <span>Valid Until</span>
          <input data-field="valid_until" type="text" value="${escapeHtml(baseline.valid_until)}" placeholder="YYYY-MM-DD (optional)">
        </label>
        <label class="memory-editor-check" title="Marks this memory as something that may go stale and should be verified later.">
          <span>Time-sensitive</span>
          <div class="memory-editor-check-box">
            <input data-field="time_sensitive" type="checkbox" ${baseline.time_sensitive ? "checked" : ""}>
          </div>
        </label>
        ${mode === "add" ? `
        <label class="memory-editor-field" title="When this memory became true or happened. Blank uses the current date and time.">
          <span>Memory Date</span>
          <input data-field="date" type="text" value="${escapeHtml(baseline.date)}" placeholder="YYYY-MM-DD (blank = now)">
        </label>` : ""}
        ${mode === "add" ? `
        <div class="memory-editor-submit-cell">
          <button class="modal-btn secondary" type="button" data-memory-add>Add Memory</button>
        </div>` : ""}
        ${mode === "add" ? "" : `
        <details class="memory-editor-advanced">
          <summary>Advanced classification</summary>
          <div class="memory-editor-grid advanced-grid">
        <label class="memory-editor-field" title="How this memory should behave now. Superseded is engine-managed.">
          <span>Status</span>
          <select data-field="status"${statusLocked ? ' disabled title="Superseding is engine-managed and set automatically."' : ""}>${renderOptions(statusLocked ? MEMORY_STATUSES : MEMORY_USER_STATUSES, status)}</select>
        </label>
        <label class="memory-editor-field" title="How certain this memory is. Low confidence is shown to the companion as a caution.">
          <span>Confidence</span>
          <select data-field="confidence">${renderOptions(MEMORY_CONFIDENCES, confidence)}</select>
        </label>
        <label class="memory-editor-field" title="Where this memory came from: human, model, import, or Pulse system.">
          <span>Source</span>
          <select data-field="source">${renderOptions(MEMORY_SOURCES, source)}</select>
        </label>
          </div>
        </details>`}
      </div>
      ${mode === "add" ? "" : `
      <div class="memory-editor-actions">
        <button class="modal-btn secondary" type="button" data-memory-save="${escapeHtml(memory.id)}">Save Memory</button>
        <button class="modal-btn danger" type="button" data-memory-delete="${escapeHtml(memory.id)}">Delete</button>
      </div>`}
    </article>`;
}

function wireMemoryEditorEvents(root) {
  root.querySelectorAll("[data-memory-save]").forEach((button) => {
    button.addEventListener("click", () => { void saveMemoryEditor(button.dataset.memorySave); });
  });
  root.querySelectorAll("[data-memory-delete]").forEach((button) => {
    button.addEventListener("click", () => { void deleteMemoryFromEditor(button.dataset.memoryDelete); });
  });
  root.querySelectorAll("[data-memory-add]").forEach((button) => {
    button.addEventListener("click", addMemoryFromEditor);
  });
  root.querySelectorAll("[data-memory-editor]").forEach((editor) => {
    editor.addEventListener("input", () => syncMemoryEditorDirty(editor.dataset.memoryEditor));
    editor.addEventListener("change", () => syncMemoryEditorDirty(editor.dataset.memoryEditor));
  });
}

function memoryIsEditable(memory) {
  return memory && memory.type !== "journal" && memory.type !== "session_log";
}

function memoryBaseline(memory) {
  return {
    text: memory.text || "",
    tags: (memory.tags || []).join(", "),
    importance: memory.importance === undefined || memory.importance === null ? "" : String(memory.importance),
    status: selectValue(memory.status, MEMORY_STATUSES, "current"),
    confidence: selectValue(memory.confidence, MEMORY_CONFIDENCES, "medium"),
    source: selectValue(memory.source, MEMORY_SOURCES, "model_extracted"),
    date: memory.date_edit || "",
    valid_until: memory.valid_until || "",
    time_sensitive: Boolean(memory.time_sensitive),
  };
}

function defaultMemoryDraft() {
  return {
    id: "new",
    text: "",
    tags: [],
    type: "fact",
    importance: 5,
    status: "current",
    confidence: "medium",
    source: "user_defined",
    date_edit: "",
    valid_until: "",
    time_sensitive: false,
  };
}

function memoryCurrentValues(memoryId) {
  const editor = el(`memoryEditor-${memoryId}`);
  if (!editor) return {};
  return {
    text: editor.querySelector('[data-field="text"]').value,
    tags: editor.querySelector('[data-field="tags"]').value,
    importance: editor.querySelector('[data-field="importance"]').value,
    date: editor.querySelector('[data-field="date"]')?.value || "",
    status: editor.querySelector('[data-field="status"]')?.value || "current",
    confidence: editor.querySelector('[data-field="confidence"]')?.value || "medium",
    source: editor.querySelector('[data-field="source"]')?.value || "user_defined",
    valid_until: editor.querySelector('[data-field="valid_until"]').value,
    time_sensitive: editor.querySelector('[data-field="time_sensitive"]').checked,
  };
}

function buildMemoryChanges(memoryId) {
  const baseline = state.memoryBaselines[String(memoryId)] || {};
  const current = memoryCurrentValues(memoryId);
  const changes = {};
  Object.keys(current).forEach((key) => {
    if (current[key] !== baseline[key]) changes[key] = current[key];
  });
  return changes;
}

function syncMemoryEditorDirty(memoryId) {
  const editor = el(`memoryEditor-${memoryId}`);
  if (!editor) return;
  editor.classList.toggle("dirty", Object.keys(buildMemoryChanges(memoryId)).length > 0);
}

function memoryIsDirty() {
  return Object.keys(state.memoryBaselines || {}).some((memoryId) => Object.keys(buildMemoryChanges(memoryId)).length > 0);
}

async function confirmDiscardMemoryEdits() {
  if (!memoryIsDirty()) return true;
  return showConfirm(
    state.memoryDetailId === "__add__" ? "Discard new memory?" : "Discard unsaved memory edits?",
    state.memoryDetailId === "__add__"
      ? "This new memory has unsaved content. Leaving this view will discard it."
      : "You have unsaved memory changes. Leaving this view will discard them.",
    "Discard",
    "secondary"
  );
}

async function addMemoryFromEditor() {
  if (!state.current) return;
  const fields = memoryCurrentValues("new");
  if (!text(fields.text).trim()) {
    setMemoryNotice("Memory text is required before adding.");
    return;
  }
  const result = await api().add_memory(state.current.name, fields);
  if (!result.ok) {
    setMemoryNotice(result.error || "Could not add memory.", state.memoryUndoStamp);
    return;
  }
  state.memoryUndoStamp = result.undo_stamp || "";
  const memoryId = result.memory_id || result.memory?.current_id || "new";
  showAddMemoryForm();
  setMemoryNotice(
    result.running
      ? `Memory #${memoryId} added. Running persona will see it after the next memory search.`
      : `Memory #${memoryId} added. Its embedding will be repaired when the persona starts or searches memory.`,
    state.memoryUndoStamp
  );
}

async function cancelAddMemory() {
  if (!(await confirmDiscardMemoryEdits())) return;
  showMemoryList();
  await loadMemoryPage(1);
}

async function saveMemoryEditor(memoryId) {
  if (!state.current || !memoryId) return;
  const changes = buildMemoryChanges(memoryId);
  if (!Object.keys(changes).length) {
    setMemoryNotice("No memory changes to save.", state.memoryUndoStamp);
    return;
  }
  const result = await api().update_memory(state.current.name, Number(memoryId), changes);
  if (!result.ok) {
    setMemoryNotice(result.error || "Could not update memory.", state.memoryUndoStamp);
    return;
  }
  if (!result.changed) {
    setMemoryNotice("No memory changes to save.", state.memoryUndoStamp);
    return;
  }
  state.memoryUndoStamp = result.undo_stamp || "";
  const detailId = state.memoryDetailId || memoryId;
  await openMemoryDetail(detailId, true);
  setMemoryNotice(
    result.running
      ? "Memory saved. Running persona will pick up the updated embedding after the next search or restart."
      : "Memory saved. Embedding will be repaired when the persona starts or searches memory.",
    state.memoryUndoStamp
  );
}

async function deleteMemoryFromEditor(memoryId) {
  if (!state.current || !memoryId) return;
  const ok = await showConfirm(
    `Delete memory #${memoryId}?`,
    "This removes the selected memory row. A one-level undo snapshot will be created first.",
    "Delete",
    "danger"
  );
  if (!ok) return;
  const result = await api().delete_memory(state.current.name, Number(memoryId));
  if (!result.ok) {
    setMemoryNotice(result.error || "Could not delete memory.", state.memoryUndoStamp);
    return;
  }
  state.memoryUndoStamp = result.undo_stamp || "";
  showMemoryList();
  await loadMemoryPage(1);
  setMemoryNotice(`Memory #${memoryId} deleted.`, state.memoryUndoStamp);
}

async function undoMemoryDbEdit() {
  if (!state.current || !state.memoryUndoStamp) return;
  const stamp = state.memoryUndoStamp;
  const result = await api().restore_db_before_image(state.current.name, stamp);
  if (!result.ok) {
    setMemoryNotice(result.error || "Could not undo memory edit.", stamp);
    return;
  }
  state.memoryUndoStamp = "";
  if (state.memoryDetailId === "__add__") {
    showAddMemoryForm();
  } else if (state.memoryDetailId) {
    await openMemoryDetail(state.memoryDetailId);
  } else {
    await loadMemoryPage(1);
  }
  setMemoryNotice("Memory edit undone.");
}

function setMemoryNotice(message, undoStamp = "") {
  setEditorNotice("memoryNotice", message || "Select a memory to review or edit. Journal index memories are edited from Journal.", undoStamp, undoMemoryDbEdit);
}

function setEditorNotice(noticeId, message, undoStamp = "", undoHandler = null) {
  const notice = el(noticeId);
  if (!notice) return;
  notice.innerHTML = `
    <span>${escapeHtml(message || "")}</span>
    ${undoStamp ? `<button class="notice-undo-btn" type="button">Undo</button>` : ""}`;
  const button = notice.querySelector(".notice-undo-btn");
  if (button && undoHandler) button.addEventListener("click", () => { void undoHandler(); });
}

function renderOptions(values, selected) {
  return values.map((value) => {
    const active = value === selected ? "selected" : "";
    return `<option value="${escapeHtml(value)}" ${active}>${escapeHtml(humanizeMemoryBadge(value))}</option>`;
  }).join("");
}

function selectValue(value, allowed, fallback) {
  const candidate = text(value);
  return allowed.includes(candidate) ? candidate : fallback;
}

function renderMemoryDetailSourceNote(memory) {
  if (memory.detail_source === "journal_entry") {
    return `
      <div class="memory-source-note">
        <strong>Journal index memory</strong>
        <span>You clicked memory #${escapeHtml(memory.id)}, which points to ${escapeHtml(memory.journal_file || "a journal entry")}. Showing the linked journal entry content below.</span>
      </div>`;
  }
  if (memory.type === "journal") {
    return `
      <div class="memory-source-note">
        <strong>Journal index memory</strong>
        <span>This memory is a searchable pointer to ${escapeHtml(memory.journal_file || "a journal entry")}. The linked journal entry was not found, so this is the stored memory preview.</span>
      </div>`;
  }
  if (memory.type === "session_log") {
    return `
      <div class="memory-source-note">
        <strong>Chat summary</strong>
        <span>This is a historical conversation summary used for orientation. It is not a raw chat log.</span>
      </div>`;
  }
  return "";
}

function humanizeMemoryBadge(value) {
  return text(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function humanizeMemoryKind(kind) {
  return {
    all: "Memories",
    fact: "Facts",
    journal: "Journal index",
    session_log: "Chat summaries",
  }[kind] || "Memories";
}

function emptyMemoryHint(view, kind) {
  if (view === "archived") {
    return "Archived memories are kept separate so old or retired context does not mingle with current continuity.";
  }
  if (kind === "journal") return "Journal index memories appear when journal entries create companion-searchable summaries.";
  if (kind === "session_log") return "Chat summaries are historical orientation memories. They are useful, but should be handled carefully.";
  return "Try another filter if you expected to see journal index memories or chat summaries.";
}

async function openJournalDialog() {
  if (!state.current || state.current.name === "__base__") {
    setNotice("Choose a persona first.", "warning", NOTICE_WARNING_MS);
    return;
  }
  state.journalBrowse = { view: "active", type: "open_thread", page: 0, pageSize: 25, total: 0, hasMore: false, items: [] };
  state.journalDetailId = "";
  state.journalBaseline = {};
  state.journalUndoStamp = "";
  setJournalNotice("Select a journal entry to review or edit. Save keeps this window open.");
  el("journalDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  syncJournalTabs();
  showJournalList();
  await loadJournalPage(1);
}

async function hideJournalDialog() {
  if (!(await confirmDiscardJournalEdits())) return;
  el("journalDialog").classList.add("hidden");
  state.journalDetailId = "";
  state.journalBaseline = {};
  state.journalUndoStamp = "";
  document.body.classList.remove("modal-open");
}

async function setJournalView(view) {
  if (!view || view === state.journalBrowse.view) return;
  if (!(await confirmDiscardJournalEdits())) return;
  state.journalBrowse = { ...state.journalBrowse, view, page: 0, total: 0, hasMore: false, items: [] };
  syncJournalTabs();
  showJournalList();
  await loadJournalPage(1);
}

async function setJournalType(type) {
  if (!type || type === state.journalBrowse.type) return;
  if (!(await confirmDiscardJournalEdits())) return;
  state.journalBrowse = { ...state.journalBrowse, type, page: 0, total: 0, hasMore: false, items: [] };
  syncJournalTabs();
  showJournalList();
  await loadJournalPage(1);
}

function syncJournalTabs() {
  el("journalViewTabs").querySelectorAll("[data-journal-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.journalView === state.journalBrowse.view);
  });
  el("journalTypeTabs").querySelectorAll("[data-journal-type]").forEach((button) => {
    button.classList.toggle("active", button.dataset.journalType === state.journalBrowse.type);
  });
}

function showJournalList() {
  state.journalDetailId = "";
  state.journalBaseline = {};
  el("journalDetail").classList.add("hidden");
  el("journalDetail").innerHTML = "";
  el("journalBody").classList.remove("hidden");
  renderJournalList();
}

async function loadJournalPage(page) {
  if (!state.current || state.current.name === "__base__") return;
  if (page === 1) {
    el("journalBody").innerHTML = `<div class="memory-empty">Loading journal entries...</div>`;
    el("journalLoadMoreBtn").classList.add("hidden");
  }
  const result = await api().list_journal_entries(
    state.current.name,
    state.journalBrowse.view,
    state.journalBrowse.type,
    page,
    state.journalBrowse.pageSize
  );
  if (!result.ok) {
    el("journalSubtitle").textContent = "Could not read journal entries.";
    el("journalBody").innerHTML = `<div class="memory-empty">${escapeHtml(result.error || "Journal browse unavailable.")}</div>`;
    return;
  }
  state.journalBrowse.page = result.page || page;
  state.journalBrowse.total = result.total || 0;
  state.journalBrowse.hasMore = Boolean(result.has_more);
  state.journalBrowse.shared = Boolean(result.shared);
  state.journalBrowse.items = page === 1
    ? (result.items || [])
    : state.journalBrowse.items.concat(result.items || []);
  renderJournalList();
}

function renderJournalList() {
  const browse = state.journalBrowse;
  const viewLabel = humanizeJournalView(browse.view);
  const typeLabel = humanizeJournalType(browse.type);
  el("journalSubtitle").textContent = `${viewLabel} ${typeLabel.toLowerCase()} / showing ${browse.items.length} of ${browse.total}${browse.shared ? " / shared pool" : ""}`;
  if (!browse.items.length) {
    el("journalBody").innerHTML = `
      <div class="memory-empty">
        <strong>No ${viewLabel.toLowerCase()} ${typeLabel.toLowerCase()} found.</strong>
        <span>${escapeHtml(emptyJournalHint(browse.view, browse.type))}</span>
      </div>`;
    el("journalLoadMoreBtn").classList.add("hidden");
    return;
  }
  el("journalBody").innerHTML = browse.items.map(renderJournalCard).join("");
  el("journalBody").querySelectorAll(".journal-card").forEach((card) => {
    card.addEventListener("click", () => { void openJournalDetail(card.dataset.entryId); });
  });
  const loadMore = el("journalLoadMoreBtn");
  loadMore.classList.toggle("hidden", !browse.hasMore);
  loadMore.textContent = `Load More (${browse.items.length}/${browse.total})`;
}

function renderJournalCard(entry) {
  const tags = (entry.tags || []).slice(0, 4).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
  const badges = [
    entry.entry_type,
    entry.status === "reference" ? "" : entry.status,
    entry.pinned ? "pinned" : "",
  ].filter(Boolean);
  const why = entry.why_preview
    ? `<div class="journal-why">Why: ${escapeHtml(entry.why_preview)}</div>`
    : "";
  return `
    <button class="journal-card memory-card" type="button" data-entry-id="${escapeHtml(entry.id)}">
      <div class="memory-card-head">
        <strong>${escapeHtml(entry.id)}</strong>
        <span>${escapeHtml(entry.date_display || "unknown")} / ${escapeHtml(entry.age_label || "unknown age")}</span>
      </div>
      <h3>${escapeHtml(entry.title || "Untitled entry")}</h3>
      <p>${escapeHtml(entry.preview || "")}</p>
      ${why}
      <div class="memory-card-foot">
        <div class="memory-tags">${tags || "<span>untagged</span>"}</div>
        <div class="memory-badges">${badges.map((badge) => `<span>${escapeHtml(humanizeMemoryBadge(badge))}</span>`).join("")}</div>
      </div>
    </button>`;
}

async function openJournalDetail(entryId, force = false) {
  if (!state.current || !entryId) return;
  if (!force && state.journalDetailId && state.journalDetailId !== String(entryId) && !(await confirmDiscardJournalEdits())) return;
  state.journalDetailId = String(entryId);
  state.journalBaseline = {};
  el("journalDetail").classList.remove("hidden");
  el("journalBody").classList.add("hidden");
  el("journalLoadMoreBtn").classList.add("hidden");
  el("journalDetail").innerHTML = `<div class="memory-empty">Loading journal entry ${escapeHtml(entryId)}...</div>`;

  const result = await api().get_journal_entry(state.current.name, entryId);
  if (!result.ok) {
    el("journalDetail").innerHTML = `<div class="memory-empty">${escapeHtml(result.error || "Journal entry unavailable.")}</div>`;
    return;
  }
  const entry = result.entry || {};
  state.journalBaseline = journalBaseline(entry);
  el("journalDetail").innerHTML = `
    <button class="memory-back-btn" type="button">&larr; Back to journal</button>
    ${renderJournalEditor(entry)}`;
  el("journalDetail").querySelector(".memory-back-btn").addEventListener("click", async () => {
    if (await confirmDiscardJournalEdits()) showJournalList();
  });
  el("journalDetail").querySelector("[data-journal-save]").addEventListener("click", saveJournalEditor);
  el("journalDetail").querySelector("[data-journal-delete]").addEventListener("click", deleteJournalFromEditor);
  const editor = el("journalEditor");
  editor.addEventListener("input", syncJournalEditorState);
  editor.addEventListener("change", syncJournalEditorState);
  editor.querySelector('[data-field="entry_type"]').addEventListener("change", syncJournalResolvedField);
  syncJournalResolvedField();
}

function renderJournalEditor(entry) {
  const baseline = journalBaseline(entry);
  const entryType = selectValue(baseline.entry_type, JOURNAL_ENTRY_TYPES, "reflection");
  const tags = (entry.tags || []).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
  return `
    <article class="journal-entry-detail memory-version journal-editor-card">
      <div class="memory-version-head">
        <strong>${escapeHtml(entry.id || "entry")}</strong>
        <span>${escapeHtml(entry.date_display || "unknown")} / ${escapeHtml(entry.age_label || "unknown age")}</span>
      </div>
      <div class="memory-badges journal-detail-badges">
        <span>${escapeHtml(humanizeMemoryBadge(entry.entry_type || "entry"))}</span>
        ${entry.summary_needs_review ? "<span>Summary needs review</span>" : ""}
        ${entry.status === "reference" ? "" : `<span>${escapeHtml(humanizeMemoryBadge(entry.status || "active"))}</span>`}
        ${entry.pinned ? "<span>Pinned</span>" : ""}
      </div>
      <div id="journalEditor" class="memory-editor-grid journal-editor-grid" data-journal-editor="${escapeHtml(entry.id)}">
        <label class="memory-editor-field">
          <span>Title</span>
          <input data-field="title" type="text" value="${escapeHtml(baseline.title)}">
        </label>
        <label class="memory-editor-field">
          <span>Type</span>
          <select data-field="entry_type">${renderOptions(JOURNAL_ENTRY_TYPES, entryType)}</select>
        </label>
        <label class="memory-editor-field">
          <span>Importance</span>
          <input data-field="importance" type="number" min="1" max="10" value="${escapeHtml(baseline.importance)}">
        </label>
        <label class="memory-editor-check">
          <span>Pinned</span>
          <div class="memory-editor-check-box">
            <input data-field="pinned" type="checkbox" ${baseline.pinned ? "checked" : ""}>
          </div>
        </label>
        <label class="memory-editor-check" id="journalResolvedField">
          <span>Resolved</span>
          <div class="memory-editor-check-box">
            <input data-field="resolved" type="checkbox" ${baseline.resolved ? "checked" : ""}>
          </div>
        </label>
        <label class="memory-editor-field wide">
          <span>Content</span>
          <textarea data-field="content" rows="10">${escapeHtml(baseline.content)}</textarea>
        </label>
        <label class="memory-editor-field wide">
          <span>Search Summary</span>
          <textarea data-field="search_summary" rows="3">${escapeHtml(baseline.search_summary)}</textarea>
        </label>
        <label class="memory-editor-field wide">
          <span>Why It Mattered</span>
          <textarea data-field="why_it_mattered" rows="3">${escapeHtml(baseline.why_it_mattered)}</textarea>
        </label>
        <label class="memory-editor-field wide">
          <span>Tags</span>
          <input data-field="tags" type="text" value="${escapeHtml(baseline.tags)}" placeholder="comma, separated, tags">
        </label>
      </div>
      <div class="memory-source-note">
        <strong>Search mirror</strong>
        <span>Saving text or summary changes updates the journal's searchable memory mirror and leaves its embedding queued for engine repair.</span>
      </div>
      <div class="memory-tags">${tags || "<span>untagged</span>"}</div>
      <div class="memory-editor-actions">
        <button class="modal-btn secondary" type="button" data-journal-save>Save Entry</button>
        <button class="modal-btn danger" type="button" data-journal-delete>Delete</button>
      </div>
    </article>`;
}

function journalBaseline(entry) {
  return {
    title: entry.title || "",
    entry_type: selectValue(entry.entry_type, JOURNAL_ENTRY_TYPES, "reflection"),
    content: entry.content || "",
    why_it_mattered: entry.why_it_mattered || "",
    search_summary: entry.search_summary || "",
    tags: (entry.tags || []).join(", "),
    importance: entry.importance === undefined || entry.importance === null ? "" : String(entry.importance),
    pinned: Boolean(entry.pinned),
    resolved: Boolean(entry.resolved),
  };
}

function journalCurrentValues() {
  const editor = el("journalEditor");
  if (!editor) return {};
  return {
    title: editor.querySelector('[data-field="title"]').value,
    entry_type: editor.querySelector('[data-field="entry_type"]').value,
    content: editor.querySelector('[data-field="content"]').value,
    why_it_mattered: editor.querySelector('[data-field="why_it_mattered"]').value,
    search_summary: editor.querySelector('[data-field="search_summary"]').value,
    tags: editor.querySelector('[data-field="tags"]').value,
    importance: editor.querySelector('[data-field="importance"]').value,
    pinned: editor.querySelector('[data-field="pinned"]').checked,
    resolved: editor.querySelector('[data-field="resolved"]').checked,
  };
}

function buildJournalChanges() {
  const baseline = state.journalBaseline || {};
  const current = journalCurrentValues();
  const changes = {};
  Object.keys(current).forEach((key) => {
    if (current[key] !== baseline[key]) changes[key] = current[key];
  });
  return changes;
}

function journalIsDirty() {
  return Boolean(state.journalDetailId) && Object.keys(buildJournalChanges()).length > 0;
}

async function confirmDiscardJournalEdits() {
  if (!journalIsDirty()) return true;
  return showConfirm(
    "Discard unsaved journal edits?",
    "You have unsaved journal changes. Leaving this view will discard them.",
    "Discard",
    "secondary"
  );
}

function syncJournalEditorState() {
  const editor = el("journalEditor");
  if (!editor) return;
  editor.classList.toggle("dirty", journalIsDirty());
}

function syncJournalResolvedField() {
  const editor = el("journalEditor");
  if (!editor) return;
  const type = editor.querySelector('[data-field="entry_type"]').value;
  const resolved = editor.querySelector('[data-field="resolved"]');
  const field = el("journalResolvedField");
  const enabled = type === "open_thread" || type === "follow_up";
  resolved.disabled = !enabled;
  field.classList.toggle("disabled", !enabled);
  if (!enabled) resolved.checked = false;
  syncJournalEditorState();
}

async function saveJournalEditor() {
  if (!state.current || !state.journalDetailId) return;
  const changes = buildJournalChanges();
  if (!Object.keys(changes).length) {
    setJournalNotice("No journal changes to save.", state.journalUndoStamp);
    return;
  }
  const result = await api().update_journal_entry(state.current.name, state.journalDetailId, changes);
  if (!result.ok) {
    setJournalNotice(result.error || "Could not update journal entry.", state.journalUndoStamp);
    return;
  }
  if (!result.changed) {
    setJournalNotice("No journal changes to save.", state.journalUndoStamp);
    return;
  }
  state.journalUndoStamp = result.undo_stamp || "";
  const detailId = state.journalDetailId;
  await openJournalDetail(detailId, true);
  setJournalNotice(
    result.running
      ? "Journal entry saved. The running persona may pick up the updated mirror after the next memory search."
      : "Journal entry saved. Its search mirror will be repaired when the persona starts or searches memory.",
    state.journalUndoStamp
  );
}

async function deleteJournalFromEditor() {
  if (!state.current || !state.journalDetailId) return;
  const entryId = state.journalDetailId;
  const ok = await showConfirm(
    `Delete journal entry ${entryId}?`,
    "This removes the journal entry and its linked memory mirror. A one-level undo snapshot will be created first.",
    "Delete",
    "danger"
  );
  if (!ok) return;
  const result = await api().delete_journal_entry(state.current.name, entryId);
  if (!result.ok) {
    setJournalNotice(result.error || "Could not delete journal entry.", state.journalUndoStamp);
    return;
  }
  state.journalUndoStamp = result.undo_stamp || "";
  showJournalList();
  await loadJournalPage(1);
  setJournalNotice(`Journal entry ${entryId} deleted with its linked mirror.`, state.journalUndoStamp);
}

async function undoJournalDbEdit() {
  if (!state.current || !state.journalUndoStamp) return;
  const stamp = state.journalUndoStamp;
  const result = await api().restore_db_before_image(state.current.name, stamp);
  if (!result.ok) {
    setJournalNotice(result.error || "Could not undo journal edit.", stamp);
    return;
  }
  state.journalUndoStamp = "";
  if (state.journalDetailId) {
    await openJournalDetail(state.journalDetailId, true);
  } else {
    await loadJournalPage(1);
  }
  setJournalNotice("Journal edit undone.");
}

function setJournalNotice(message, undoStamp = "") {
  setEditorNotice("journalNotice", message || "Select a journal entry to review or edit. Save keeps this window open.", undoStamp, undoJournalDbEdit);
}

function humanizeJournalView(view) {
  return {
    active: "Active",
    resolved: "Resolved",
    all: "All",
  }[view] || "Active";
}

function humanizeJournalType(type) {
  return {
    all: "Entries",
    event: "Events",
    preference: "Preferences",
    topic: "Topics",
    tone: "Tone notes",
    open_thread: "Open threads",
    follow_up: "Follow-ups",
    reflection: "Reflections",
  }[type] || "Entries";
}

function emptyJournalHint(view, type) {
  if (view === "resolved") return "Resolved entries are completed follow-ups or closed threads. Active entries stay in the main view.";
  if (type === "open_thread") return "Open threads are unresolved topics the companion may want to return to.";
  if (type === "follow_up") return "Follow-ups appear when the companion writes down something to check later.";
  return "Try another type filter, or check All if you expected older journal context.";
}

async function openCoreDialog(anchorId) {
  if (!state.current || state.current.name === "__base__") {
    setNotice("Choose a persona first.", "warning", NOTICE_WARNING_MS);
    return;
  }
  setCoreNotice("");
  state.currentCoreAnchor = null;
  state.coreBaseline = {};
  el("coreDialog").classList.remove("hidden");
  document.body.classList.add("modal-open");
  el("coreTitle").textContent = coreAnchorLabel(anchorId);
  el("coreSubtitle").textContent = "Loading journal identity anchor...";
  el("coreBody").innerHTML = `<div class="memory-empty">Loading ${escapeHtml(coreAnchorLabel(anchorId).toLowerCase())}...</div>`;

  const result = await api().get_core_anchor(state.current.name, anchorId);
  if (!result.ok) {
    el("coreSubtitle").textContent = "Could not read core anchor.";
    el("coreBody").innerHTML = `<div class="memory-empty">${escapeHtml(result.error || "Core anchor unavailable.")}</div>`;
    return;
  }
  renderCoreAnchor(result.anchor || {}, "Edit pinned identity context, then save. Changes apply on the persona's next turn or heartbeat.");
}

function hideCoreDialog() {
  el("coreDialog").classList.add("hidden");
  setCoreNotice("");
  state.currentCoreAnchor = null;
  state.coreBaseline = {};
  document.body.classList.remove("modal-open");
}

async function closeCoreDialog() {
  if (coreIsDirty()) {
    const ok = await showConfirm(
      "Discard unsaved core edits?",
      "You have unsaved changes to this core anchor. Closing now will discard them.",
      "Discard",
      "secondary"
    );
    if (!ok) return;
  }
  hideCoreDialog();
}

function renderCoreAnchor(anchor, notice = "") {
  state.currentCoreAnchor = anchor;
  setCoreNotice(notice);
  el("coreTitle").textContent = anchor.title || coreAnchorLabel(anchor.id);
  el("coreSubtitle").textContent = `${anchor.id || "anchor"} / journal identity anchor / updated ${anchor.last_updated_display || "unknown"}`;
  const sections = coreEditableSections(anchor);
  state.coreBaseline = {};
  sections.forEach((section) => {
    state.coreBaseline[section.key] = section.value || "";
  });
  el("coreBody").innerHTML = `
    <div class="core-editor-sections">
      ${sections.map(renderCoreEditorSection).join("")}
    </div>`;
}

function coreCurrentValues() {
  const values = {};
  el("coreBody").querySelectorAll("[data-core-section]").forEach((input) => {
    values[input.dataset.coreSection] = input.value;
  });
  return values;
}

function coreIsDirty() {
  if (!state.currentCoreAnchor) return false;
  const baseline = state.coreBaseline || {};
  const current = coreCurrentValues();
  const keys = new Set([...Object.keys(baseline), ...Object.keys(current)]);
  for (const key of keys) {
    if ((baseline[key] || "") !== (current[key] || "")) return true;
  }
  return false;
}

function renderCoreEditorSection(section) {
  return `
    <label class="core-editor-section">
      <span>${escapeHtml(section.label || section.key || "Section")}</span>
      <textarea data-core-section="${escapeHtml(section.key)}" rows="5">${escapeHtml(section.value || "")}</textarea>
    </label>`;
}

async function saveCoreAnchor() {
  const anchor = state.currentCoreAnchor;
  if (!anchor || !state.current) return;
  const sections = coreCurrentValues();
  const result = await api().set_core_anchor(state.current.name, anchor.id, sections);
  if (!result.ok) {
    setCoreLocalNotice(result.error || "Could not update core anchor.");
    return;
  }
  if (!result.changed) {
    setCoreLocalNotice("No core anchor changes to save.");
    return;
  }
  const updatedAnchor = result.anchor || anchor;
  const message = result.running
    ? "Core anchor updated. Running persona may pick it up on the next turn or heartbeat."
    : "Core anchor updated.";
  renderCoreAnchor(updatedAnchor, message);
  setCoreAnchorIndicator(updatedAnchor.id, !updatedAnchor.empty);
}

function setCoreLocalNotice(message) {
  setCoreNotice(message);
}

function setCoreNotice(message) {
  const notice = el("coreNotice");
  if (!notice) return;
  notice.textContent = message || "";
  notice.classList.toggle("hidden", !message);
}

function coreEditableSections(anchor) {
  const sections = Array.isArray(anchor.sections) ? anchor.sections : [];
  return sections.length ? sections : defaultCoreSections(anchor.id);
}

function defaultCoreSections(anchorId) {
  const keys = {
    _self: ["who_i_am", "what_im_like", "my_preferences", "how_i_present_myself", "what_im_working_on", "extra_notes"],
    _user: ["who_they_are", "what_theyre_like", "their_preferences", "how_they_communicate", "extra_notes"],
    _relationship: ["how_we_relate", "our_dynamic", "shared_context", "boundaries_or_norms", "extra_notes"],
  }[anchorId] || ["extra_notes"];
  return keys.map((key) => ({ key, label: coreSectionLabel(key), value: "" }));
}

function coreSectionLabel(key) {
  return {
    what_theyre_like: "What They're Like",
    how_they_communicate: "How They Communicate",
    their_preferences: "Their Preferences",
    who_they_are: "Who They Are",
    what_im_like: "What I'm Like",
    what_im_working_on: "What I'm Working On",
    who_i_am: "Who I Am",
    my_preferences: "My Preferences",
    how_i_present_myself: "How I Present Myself",
    how_we_relate: "How We Relate",
    our_dynamic: "Our Dynamic",
    shared_context: "Shared Context",
    boundaries_or_norms: "Boundaries Or Norms",
    extra_notes: "Extra Notes",
  }[key] || key.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function setCoreAnchorIndicator(anchorId, hasContent) {
  const button = Array.from(document.querySelectorAll("[data-core-anchor]"))
    .find((candidate) => candidate.dataset.coreAnchor === anchorId);
  if (!button) return;
  button.classList.toggle("has-content", hasContent);
  button.classList.toggle("is-empty", !hasContent);
}

function coreAnchorLabel(anchorId) {
  return {
    _self: "Core Self",
    _user: "Core User",
    _relationship: "Core Relationship",
  }[anchorId] || "Core Anchor";
}

async function restoreBackup(path) {
  if (!state.current || !path) return;
  hideBackupsDialog();
  const dirtyNote = state.dirty
    ? " Unsaved GUI edits will be discarded."
    : "";
  const ok = await showConfirm(
    "Restore this backup?",
    `This will restore the selected config backup for this persona. A safety backup will be created first.${dirtyNote}`,
    "Restore",
    "secondary"
  );
  if (!ok) return;
  const wasRunning = currentPersonaIsRunning();
  const persona = state.current.name;
  const result = await api().restore_backup(persona, path);
  if (!result.ok) {
    setNotice(result.error || "Restore failed.", "warning");
    return;
  }
  await loadPersona(persona);
  setCanUndo(Boolean(result.changed));
  if (result.changed && wasRunning) {
    setNotice("Restored. Restart the persona for changes to take effect.", "warning", NOTICE_WARNING_MS);
    setFooterNotice("Restart needed: restored changes take effect after restart.");
  } else {
    setNotice(result.changed ? "Restored selected backup." : "Nothing to restore.", "info", NOTICE_INFO_MS);
    setFooterNotice("");
  }
}

function wireSliders() {
  let active = null;

  function pctFromEvent(e, track) {
    const rect = track.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    return Math.min(100, Math.max(0, (x / rect.width) * 100));
  }

  function applySlider(row, pct) {
    const fill = row.querySelector(".slider-fill");
    const handle = row.querySelector(".slider-handle");
    const valEl = row.querySelector(".slider-value");
    fill.style.width = pct + "%";
    handle.style.left = pct + "%";
    const min = parseFloat(row.dataset.min);
    const max = parseFloat(row.dataset.max);
    const fmt = parseInt(row.dataset.fmt, 10);
    const val = min + (pct / 100) * (max - min);
    valEl.textContent = val.toFixed(fmt);
    valEl.classList.remove("default");
  }

  function finishSliderDrag() {
    if (!active) return;
    active = null;
    setDirty(hasEditableChanges());
  }

  document.addEventListener("mousedown", (e) => {
    const track = e.target.closest(".slider-track");
    if (!track) return;
    if (state.current?.name === "__base__") return;
    const row = track.closest(".slider-row");
    if (!row) return;
    active = { row, track };
    applySlider(row, pctFromEvent(e, track));
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!active) return;
    applySlider(active.row, pctFromEvent(e, active.track));
  });

  document.addEventListener("mouseup", finishSliderDrag);

  document.addEventListener("touchstart", (e) => {
    const track = e.target.closest(".slider-track");
    if (!track) return;
    if (state.current?.name === "__base__") return;
    const row = track.closest(".slider-row");
    if (!row) return;
    active = { row, track };
    applySlider(row, pctFromEvent(e, track));
  }, { passive: true });

  document.addEventListener("touchmove", (e) => {
    if (!active) return;
    applySlider(active.row, pctFromEvent(e, active.track));
  }, { passive: true });

  document.addEventListener("touchend", finishSliderDrag);

  document.addEventListener("click", (e) => {
    const valEl = e.target.closest(".slider-value");
    if (!valEl || valEl.dataset.editing) return;
    if (state.current?.name === "__base__") return;
    const row = valEl.closest(".slider-row");
    if (!row) return;

    const oldText = valEl.textContent;
    const fmt = parseInt(row.dataset.fmt, 10);
    const min = parseFloat(row.dataset.min);
    const max = parseFloat(row.dataset.max);

    const input = document.createElement("input");
    input.type = "number";
    input.className = "slider-edit";
    input.value = oldText;
    input.step = fmt === 0 ? "1" : "0.01";
    input.min = row.dataset.min;
    input.max = row.dataset.max;
    valEl.replaceWith(input);
    input.focus();
    input.select();

    function commit() {
      const num = parseFloat(input.value);
      const clamped = isNaN(num) ? min : Math.min(max, Math.max(min, num));
      const pct = ((clamped - min) / (max - min)) * 100;

      const span = document.createElement("span");
      span.className = "slider-value";
      span.textContent = clamped.toFixed(fmt);

      const fill = row.querySelector(".slider-fill");
      const handle = row.querySelector(".slider-handle");
      fill.style.width = pct + "%";
      handle.style.left = pct + "%";

      input.replaceWith(span);
      setDirty(hasEditableChanges());
    }

    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
      if (ev.key === "Escape") { input.value = oldText; input.blur(); }
    });
  });
}

function showConfirm(title, body, okLabel = "Confirm", okStyle = "danger") {
  return new Promise((resolve) => {
    el("confirmTitle").textContent = title;
    el("confirmBody").innerHTML = body;
    const dialogCard = el("confirmDialog").querySelector(".modal-card");
    const okBtn = el("confirmOk");
    const secondaryBtn = el("confirmSecondary");
    okBtn.textContent = okLabel;
    okBtn.className = "modal-btn " + okStyle;
    secondaryBtn.classList.add("hidden");
    dialogCard.classList.toggle("diff-card", body.includes("diff-block"));
    el("confirmDialog").classList.remove("hidden");
    function cleanup(result) {
      el("confirmDialog").classList.add("hidden");
      dialogCard.classList.remove("diff-card");
      okBtn.replaceWith(okBtn.cloneNode(true));
      secondaryBtn.replaceWith(secondaryBtn.cloneNode(true));
      el("confirmCancel").replaceWith(el("confirmCancel").cloneNode(true));
      resolve(result);
    }
    el("confirmOk").addEventListener("click", () => cleanup(true), { once: true });
    el("confirmCancel").addEventListener("click", () => cleanup(false), { once: true });
    el("confirmDialog").addEventListener("click", (event) => {
      if (confirmBackdropPointerDown && event.target === el("confirmDialog")) cleanup(false);
      confirmBackdropPointerDown = false;
    }, { once: true });
  });
}

function showChoice(title, body, { okLabel, okStyle = "secondary", secondaryLabel, cancelLabel = "Cancel" }) {
  return new Promise((resolve) => {
    el("confirmTitle").textContent = title;
    el("confirmBody").innerHTML = body;
    const dialogCard = el("confirmDialog").querySelector(".modal-card");
    const okBtn = el("confirmOk");
    const secondaryBtn = el("confirmSecondary");
    const cancelBtn = el("confirmCancel");
    okBtn.textContent = okLabel;
    okBtn.className = "modal-btn " + okStyle;
    secondaryBtn.textContent = secondaryLabel;
    secondaryBtn.className = "modal-btn secondary";
    cancelBtn.textContent = cancelLabel;
    dialogCard.classList.toggle("diff-card", body.includes("diff-block"));
    el("confirmDialog").classList.remove("hidden");
    function cleanup(result) {
      el("confirmDialog").classList.add("hidden");
      dialogCard.classList.remove("diff-card");
      okBtn.replaceWith(okBtn.cloneNode(true));
      secondaryBtn.replaceWith(secondaryBtn.cloneNode(true));
      cancelBtn.replaceWith(cancelBtn.cloneNode(true));
      resolve(result);
    }
    el("confirmOk").addEventListener("click", () => cleanup("ok"), { once: true });
    el("confirmSecondary").addEventListener("click", () => cleanup("secondary"), { once: true });
    el("confirmCancel").addEventListener("click", () => cleanup("cancel"), { once: true });
    el("confirmDialog").addEventListener("click", (event) => {
      if (confirmBackdropPointerDown && event.target === el("confirmDialog")) cleanup("cancel");
      confirmBackdropPointerDown = false;
    }, { once: true });
  });
}

function showCloseDialog(personas) {
  const names = personas.map((n) => `<strong>${escapeHtml(n)}</strong>`).join(", ");
  el("closeDialogPersonas").innerHTML = `Pulse is still running for: ${names}`;
  el("closeDialog").classList.remove("hidden");
}

function hideCloseDialog() {
  el("closeDialog").classList.add("hidden");
}

function wireCloseDialog() {
  el("closeCancelBtn").addEventListener("click", hideCloseDialog);
  el("closeKeepBtn").addEventListener("click", async () => {
    hideCloseDialog();
    await api().close_keep_running();
  });
  el("closeStopBtn").addEventListener("click", async () => {
    hideCloseDialog();
    setNotice("Stopping all personas...");
    await api().stop_all_and_close();
  });
}

window.addEventListener("pywebviewready", refreshAll);
window.addEventListener("DOMContentLoaded", () => {
  wireEvents();
  wireSliders();
  wireCloseDialog();
  if (!window.pywebview) refreshAll();
  setInterval(refreshCurrentStatus, 2000);
});
