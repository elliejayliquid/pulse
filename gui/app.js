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

const state = {
  personas: [],
  current: null,
  prefs: {},
  pendingTransition: null,
  dirty: false,
  canUndo: false,
  originalEditable: null,
  currentTraits: [],
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
        telegram_key: "TELEGRAM_BOT_TOKEN",
        telegram_set: false,
        telegram_enabled: true,
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
  async preview_persona_save() { return { ok: false, error: "Run through pywebview to save." }; },
  async save_persona() { return { ok: false, error: "Run through pywebview to save." }; },
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

let noticeTimer = null;

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
}

function renderTuning(summary) {
  const s = summary || {};
  const sliders = [
    { label: "Temperature",    key: "temperature",        min: 0, max: 2,    fmt: 2, tip: "How creative vs predictable. 0.7–0.85 is the sweet spot. Above 1.0 gets chaotic fast" },
    { label: "Max Response",   key: "max_response_tokens", min: 256, max: 32768, fmt: 0, tip: "Max reply length in tokens (~750 words per 1000). 2048–4096 is a good range — too high and replies get rambly" },
    { label: "Freq Penalty",   key: "frequency_penalty",  min: 0, max: 2,    fmt: 2, tip: "Discourages repeating the same words. 0.3–0.5 works well — above 1.0 causes awkward word avoidance" },
    { label: "Pres Penalty",   key: "presence_penalty",   min: 0, max: 2,    fmt: 2, tip: "Encourages new topics. 0.1–0.3 is natural — too high and the companion jumps topics mid-thought" },
    { label: "Top P",          key: "top_p",              min: 0, max: 1,    fmt: 2, tip: "Word choice diversity. 0.95–1.0 is good. Lower values make responses safer but can feel flat" },
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

  html += `<div class="tuning-extras">
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
    <label class="tuning-inline-label" style="margin-left:auto" title="How many tool calls the model can chain per turn. Default 8 is good — higher lets it do more autonomously but takes longer">
      Max Tool Rounds
      <input id="tuningMaxRounds" class="tuning-inline-input" type="number" min="1" max="32" value="${escapeHtml(rounds)}">
    </label>
  </div>
  <div class="tuning-extras">
    <label class="tuning-inline-label" title="Message pairs kept word-for-word after summarization. 2–4 is good — keeps continuity without eating up context">
      Recent Tail Exchanges
      <input id="tuningTailExchanges" class="tuning-inline-input" type="number" min="1" max="10" value="${escapeHtml(tail)}">
    </label>
  </div>`;

  el("tuningGrid").innerHTML = html;
  wireTuningControls();
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
    telegram_key: status.telegram_key || "TELEGRAM_BOT_TOKEN",
    telegram_set: Boolean(status.telegram_set),
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

  if (active || stopping) {
    btn.textContent = stopping ? "Stopping..." : "■ Stop";
    btn.className = "hero-stop-btn";
    btn.disabled = isBase || stopping;
    btn.title = "";
  } else {
    btn.textContent = "▶ Start";
    btn.className = "hero-start-btn";
    btn.disabled = isBase || missingKey;
    btn.title = missingKey
      ? `API key missing - set ${providerKeyEnv(keyStatus) || "provider.api_key_env"} in .env to start`
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
    <button class="skill-card ${skill.enabled ? "on" : ""}" type="button"
      data-skill="${escapeHtml(skill.name)}" ${editable ? "" : "disabled"}>
      <div class="skill-icon">${escapeHtml(skill.icon)}</div>
      <div class="skill-info">
        <div class="skill-name">${escapeHtml(skill.label)}</div>
        <div class="skill-state">${skill.enabled ? "Enabled" : "Disabled"}</div>
      </div>
      <div class="skill-toggle" aria-hidden="true"></div>
    </button>
  `).join("");
  grid.querySelectorAll(".skill-card").forEach((card) => {
    card.addEventListener("click", () => {
      if (!editable) return;
      const skill = state.current.skills.find((item) => item.name === card.dataset.skill);
      if (!skill) return;
      skill.enabled = !skill.enabled;
      renderSkills(state.current.skills);
      setDirty(hasEditableChanges());
    });
  });
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

function providerKeyLabel(status) {
  if (status.provider_type === "custom") return status.api_key_env || "API Key";
  return PROVIDER_LABELS[status.provider_type] || status.api_key_env || "API Key";
}

function providerDisplayName(status) {
  return PROVIDER_NAMES[status.provider_type] || status.provider_type || "Provider";
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
    el("apiKeyStatus").textContent = normalized.api_key_set ? "Set" : "Missing";
    el("apiKeyStatus").className = normalized.api_key_set ? "set" : "missing";
  }

  el("telegramStatus").textContent = normalized.telegram_set ? "Set" : "Missing";
  el("telegramStatus").className = normalized.telegram_set ? "set" : "missing";
  renderProviderWarnings(normalized);
  renderProcessButton(state.current);
}

function renderIdentity(data) {
  const identity = data.identity || {};
  const summary = data.summary || {};
  el("identityName").value = text(identity.name);
  el("identityUser").value = text(identity.user_name);
  el("identityModel").value = text(identity.model || summary.model_display);
  el("systemPrompt").value = text(identity.system_prompt);
  el("voiceNotes").value = text(identity.voice_notes || identity.relationship_context);
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

function syncLocalServerVisibility() {
  const isLocal = el("providerType").value === "local";
  el("localServerSection").classList.toggle("hidden", !isLocal);
}

function renderProvider(data) {
  const summary = data.summary || {};
  const tts = summary.tts || {};
  el("providerType").value = text(summary.provider_type);
  el("providerModel").value = text(summary.provider_model);
  el("providerBaseUrl").value = text(summary.base_url);
  el("maxContext").value = text(summary.provider_max_context ?? summary.max_context);
  syncBaseUrlVisibility();
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
      voice_notes: el("voiceNotes").value,
      traits: [...state.currentTraits],
    },
    provider: {
      type: el("providerType").value,
      model: el("providerModel").value,
      base_url: el("providerBaseUrl").value,
      max_context: el("maxContext").value,
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
      (state.current?.channels || []).map((channel) => [channel.name, Boolean(channel.enabled)])
    ),
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
    el(id).readOnly = !editable;
  });
  ["providerType", "lsFlashAttn", "tuningReasoning", "tuningEffort", "tuningShowReasoning", "hbDebug"].forEach((id) => {
    el(id).disabled = !editable;
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
  const original = state.originalEditable || { identity: {}, provider: {}, server: {}, model: {}, context_budget: {}, tts: {}, skills: {}, channels: {}, heartbeat: {} };
  const changes = { identity: {}, provider: {}, server: {}, model: {}, context_budget: {}, tts: {}, skills: {}, channels: {}, heartbeat: {} };

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
        : ["max_response_tokens", "max_context", "max_tool_rounds"].includes(key)
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
    if (value !== original.channels?.[key]) {
      changes.channels[key] = value;
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
    || Object.keys(changes.heartbeat).length > 0;
}

function validateHeartbeatFields() {
  const numeric = [
    ["hbInterval", "Heartbeat interval", 1, 1440],
    ["hbMin", "Heartbeat min interval", 1, 1440],
    ["hbMax", "Heartbeat max interval", 1, 1440],
    ["hbQuietStart", "Quiet start", 0, 23],
    ["hbQuietEnd", "Quiet end", 0, 23],
  ];
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
  const hbMin = el("hbMin").value === "" ? null : Number(el("hbMin").value);
  const hbMax = el("hbMax").value === "" ? null : Number(el("hbMax").value);
  if (hbMin !== null && hbMax !== null && hbMin > hbMax) {
    return "Heartbeat min interval cannot be greater than max interval.";
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
    setNotice("You are viewing the base config. Phase 1 is read-only, so nothing can be saved from here yet.");
  }
}

async function loadLogs() {
  if (!state.current) return;
  el("logTail").textContent = await api().get_log_tail(state.current.name, 100);
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
  ["tuningMaxRounds", "tuningTailExchanges"].forEach((id) => {
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
  document.querySelectorAll(".section-header").forEach((header) => {
    header.addEventListener("click", () => header.parentElement.classList.toggle("open"));
  });
  [
    "identityName",
    "identityUser",
    "identityModel",
    "systemPrompt",
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
    el(id).addEventListener("input", () => setDirty(hasEditableChanges()));
  });
  el("providerType").addEventListener("change", () => {
    syncBaseUrlVisibility();
    syncLocalServerVisibility();
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
