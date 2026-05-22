const state = {
  personas: [],
  current: null,
  prefs: {},
  pendingTransition: null,
  dirty: false,
  canUndo: false,
  originalEditable: null,
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
        max_context: "",
        temperature: 0.7,
        max_response_tokens: 2048,
        frequency_penalty: 0.4,
        presence_penalty: 0.2,
        top_p: 1.0,
        reasoning: true,
        reasoning_effort: "high",
        show_reasoning: false,
        max_tool_rounds: 8,
        tts: {},
      },
      skills: [],
      channels: [],
      key_status: { api_key_env: "API_KEY", api_key_set: false, telegram_set: false },
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
  async restore_last_backup() { return { ok: false, error: "Run through pywebview to undo." }; },
  async pick_voice_sample() { return { ok: false, error: "Run through pywebview to pick files." }; },
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

function setNotice(message, kind = "info") {
  const box = el("notice");
  if (!message) {
    box.classList.add("hidden");
    return;
  }
  box.textContent = message;
  box.dataset.kind = kind;
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
    { label: "Temperature",    key: "temperature",        min: 0, max: 2,    fmt: 2 },
    { label: "Max Response",   key: "max_response_tokens", min: 0, max: 8192, fmt: 0 },
    { label: "Freq Penalty",   key: "frequency_penalty",  min: 0, max: 2,    fmt: 2 },
    { label: "Pres Penalty",   key: "presence_penalty",   min: 0, max: 2,    fmt: 2 },
    { label: "Top P",          key: "top_p",              min: 0, max: 1,    fmt: 2 },
  ];

  let html = "";
  for (const sl of sliders) {
    const raw = s[sl.key];
    const val = parseFloat(raw);
    const ok = !isNaN(val);
    const pct = ok ? Math.min(100, Math.max(0, ((val - sl.min) / (sl.max - sl.min)) * 100)) : 0;
    const display = ok ? val.toFixed(sl.fmt) : "default";
    html += `<div class="slider-row" data-min="${sl.min}" data-max="${sl.max}" data-fmt="${sl.fmt}">
      <span class="slider-label">${escapeHtml(sl.label)}</span>
      <div class="slider-track">
        <div class="slider-fill" style="width:${pct}%"></div>
        <div class="slider-handle" style="left:${ok ? pct : 0}%"></div>
      </div>
      <span class="slider-value${ok ? "" : " default"}">${escapeHtml(display)}</span>
    </div>`;
  }

  const reasoning = s.reasoning;
  const effort = text(s.reasoning_effort, "default");
  const showR = s.show_reasoning;
  const rounds = text(s.max_tool_rounds, "default");

  html += `<div class="tuning-extras">
    <div class="check-row">
      <span class="check-box${reasoning ? " checked" : ""}">${reasoning ? "✓" : ""}</span>
      <span class="check-label">Reasoning</span>
    </div>
    <span class="tuning-tag">${escapeHtml(effort)}</span>
    <div class="check-row">
      <span class="check-box${showR ? " checked" : ""}">${showR ? "✓" : ""}</span>
      <span class="check-label">Show Reasoning</span>
    </div>
    <div class="check-row" style="margin-left:auto">
      <span class="check-label" style="color:var(--text-secondary)">Max Tool Rounds</span>
      <span class="tuning-tag">${escapeHtml(rounds)}</span>
    </div>
  </div>`;

  el("tuningGrid").innerHTML = html;
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

function renderProcessButton(data) {
  const btn = el("pulseToggle");
  const isBase = data?.name === "__base__";
  const status = data?.status || {};
  const active = Boolean(status.running) && !status.stale;
  const stopping = status.phase === "stopping";

  btn.disabled = isBase || stopping;

  if (active || stopping) {
    btn.textContent = stopping ? "Stopping..." : "■ Stop";
    btn.className = "hero-stop-btn";
  } else {
    btn.textContent = "▶ Start";
    btn.className = "hero-start-btn";
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
  grid.innerHTML = skills.map((skill) => `
    <div class="skill-card ${skill.enabled ? "on" : ""}">
      <div class="skill-icon">${escapeHtml(skill.icon)}</div>
      <div class="skill-info">
        <div class="skill-name">${escapeHtml(skill.label)}</div>
        <div class="skill-state">${skill.enabled ? "Enabled" : "Disabled"}</div>
      </div>
      <div class="skill-toggle" aria-hidden="true"></div>
    </div>
  `).join("");
  grid.querySelectorAll(".skill-card").forEach((card) => {
    card.style.cursor = "pointer";
    card.addEventListener("click", () => {
      card.classList.toggle("on");
      const state = card.querySelector(".skill-state");
      if (state) state.textContent = card.classList.contains("on") ? "Enabled" : "Disabled";
    });
  });
}

function renderChannels(channels) {
  const node = el("channels");
  if (!channels?.length) {
    node.innerHTML = `<span class="channel off">No channels configured</span>`;
    return;
  }
  node.innerHTML = channels.map((channel) => (
    `<span class="channel ${channel.enabled ? "" : "off"}">${channel.enabled ? "on" : "off"} · ${escapeHtml(channel.label)}</span>`
  )).join("");
}

function renderSecrets(status) {
  el("apiKeyName").textContent = status.api_key_env || "API key env not configured";
  el("apiKeyStatus").textContent = status.api_key_set ? "Set" : "Missing";
  el("apiKeyStatus").className = status.api_key_set ? "set" : "missing";
  el("telegramStatus").textContent = status.telegram_set ? "Set" : "Missing";
  el("telegramStatus").className = status.telegram_set ? "set" : "missing";
}

function renderIdentity(data) {
  const identity = data.identity || {};
  const summary = data.summary || {};
  el("identityName").value = text(identity.name);
  el("identityUser").value = text(identity.user_name);
  el("identityModel").value = text(identity.model || summary.model_display);
  el("systemPrompt").value = text(identity.system_prompt);
  el("voiceNotes").value = text(identity.voice_notes || identity.relationship_context);
  const traits = Array.isArray(identity.traits) ? identity.traits : [];
  el("traits").innerHTML = (traits.length
    ? traits.map((trait) => `<span class="chip">${escapeHtml(trait)}</span>`).join("")
    : `<span class="chip">No traits listed</span>`)
    + `<span class="chip-add">+ add</span>`;
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

function renderProvider(data) {
  const summary = data.summary || {};
  const tts = summary.tts || {};
  el("providerType").value = text(summary.provider_type);
  el("providerModel").value = text(summary.provider_model);
  el("maxContext").value = text(summary.max_context);
  el("ttsVoice").value = text(tts.voice_description);
  el("ttsSample").value = text(tts.voice_sample);
  el("ttsSampleText").value = text(tts.voice_sample_text);
  syncTtsMode();
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
      model: el("identityModel").value,
      voice_notes: el("voiceNotes").value,
    },
    tts: {
      voice_description: el("ttsVoice").value,
      voice_sample: el("ttsSample").value,
      voice_sample_text: el("ttsSampleText").value,
    },
  };
}

function setEditableState(data) {
  const editable = data?.name && data.name !== "__base__";
  ["identityModel", "voiceNotes", "ttsVoice", "ttsSampleText"].forEach((id) => {
    el(id).readOnly = !editable;
  });
  el("ttsSample").readOnly = true;
  el("ttsSamplePicker").disabled = !editable;
  el("ttsSampleClear").disabled = !editable;
  state.originalEditable = editableSnapshot();
  setDirty(false);
  setCanUndo(false);
}

function collectEditableChanges() {
  const current = editableSnapshot();
  const original = state.originalEditable || { identity: {}, tts: {} };
  const changes = { identity: {}, tts: {} };

  for (const [key, value] of Object.entries(current.identity)) {
    if (value !== original.identity?.[key]) {
      changes.identity[key] = value;
    }
  }

  for (const [key, value] of Object.entries(current.tts)) {
    if (value !== original.tts?.[key]) {
      changes.tts[key] = value;
    }
  }

  return changes;
}

function hasEditableChanges() {
  const changes = collectEditableChanges();
  return Object.keys(changes.identity).length > 0 || Object.keys(changes.tts).length > 0;
}

async function loadPersona(name) {
  setNotice("");
  const data = await api().load_persona(name);
  state.current = data;
  renderHero(data);
  renderStatus(data.status || {});
  renderIdentity(data);
  renderProvider(data);
  renderSecrets(data.key_status || {});
  renderTuning(data.summary || {});
  renderRuntime(data.status || {}, data.summary || {});
  renderSkills(data.skills || []);
  renderChannels(data.channels || []);
  renderProcessButton(data);
  setEditableState(data);
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
  document.querySelectorAll(".section-header").forEach((header) => {
    header.addEventListener("click", () => header.parentElement.classList.toggle("open"));
  });
  ["identityModel", "voiceNotes", "ttsVoice", "ttsSampleText"].forEach((id) => {
    el(id).addEventListener("input", () => setDirty(hasEditableChanges()));
  });
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
    if (!state.current || state.current.name === "__base__") return;
    if (el("ttsSample").value && !el("ttsSampleText").value) {
      setNotice("Voice sample is set but transcript is empty — clone mode needs both.", "warning");
      return;
    }
    const changes = collectEditableChanges();
    const preview = await api().preview_persona_save(state.current.name, changes);
    if (!preview.ok) {
      setNotice(preview.error || "Could not preview changes.", "warning");
      return;
    }
    if (!preview.preview.has_changes) {
      setDirty(false);
      setNotice("No changes to save.");
      return;
    }
    const changed = preview.preview.changes.map((item) => item.file).join(", ");
    const diff = preview.preview.diff || "";
    const shownDiff = diff.length > 1200 ? diff.slice(0, 1200) + "\n..." : diff;
    const body = `Saving to <strong>${escapeHtml(changed)}</strong>. A backup will be created first.`
      + (shownDiff ? `<pre class="diff-block">${formatDiff(shownDiff)}</pre>` : "");
    const ok = await showConfirm("Save changes?", body, "Save", "secondary");
    if (!ok) return;
    const result = await api().save_persona(state.current.name, changes);
    if (!result.ok) {
      setNotice(result.error || "Save failed.", "warning");
      return;
    }
    await loadPersona(state.current.name);
    setDirty(false);
    setCanUndo(Boolean(result.changed));
    setNotice(result.changed ? "Saved with backup." : "No changes to save.");
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
    const result = await api().restore_last_backup(state.current.name);
    if (!result.ok) {
      setNotice(result.error || "Undo failed.", "warning");
      return;
    }
    await loadPersona(state.current.name);
    setCanUndo(false);
    setNotice(result.changed ? "Restored latest backup." : "Nothing to restore.");
  });
  el("pulseToggle").addEventListener("click", async () => {
    if (!state.current || state.current.name === "__base__") return;
    const status = state.current.status || {};
    const active = Boolean(status.running) && !status.stale;

    if (active) {
      setNotice("Requesting graceful shutdown...");
      state.pendingTransition = "stopping";
      const result = await api().stop_pulse(state.current.name);
      if (!result.ok) {
        state.pendingTransition = null;
        setNotice(result.error || "Failed to stop Pulse.", "warning");
        return;
      }
      state.current.status = result.status || {};
    } else {
      setNotice("Starting Pulse...");
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

  document.addEventListener("mousedown", (e) => {
    const track = e.target.closest(".slider-track");
    if (!track) return;
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

  document.addEventListener("mouseup", () => { active = null; });

  document.addEventListener("touchstart", (e) => {
    const track = e.target.closest(".slider-track");
    if (!track) return;
    const row = track.closest(".slider-row");
    if (!row) return;
    active = { row, track };
    applySlider(row, pctFromEvent(e, track));
  }, { passive: true });

  document.addEventListener("touchmove", (e) => {
    if (!active) return;
    applySlider(active.row, pctFromEvent(e, active.track));
  }, { passive: true });

  document.addEventListener("touchend", () => { active = null; });
}

function showConfirm(title, body, okLabel = "Confirm", okStyle = "danger") {
  return new Promise((resolve) => {
    el("confirmTitle").textContent = title;
    el("confirmBody").innerHTML = body;
    const okBtn = el("confirmOk");
    okBtn.textContent = okLabel;
    okBtn.className = "modal-btn " + okStyle;
    el("confirmDialog").classList.remove("hidden");
    function cleanup(result) {
      el("confirmDialog").classList.add("hidden");
      okBtn.replaceWith(okBtn.cloneNode(true));
      el("confirmCancel").replaceWith(el("confirmCancel").cloneNode(true));
      resolve(result);
    }
    el("confirmOk").addEventListener("click", () => cleanup(true), { once: true });
    el("confirmCancel").addEventListener("click", () => cleanup(false), { once: true });
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
