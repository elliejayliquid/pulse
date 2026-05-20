const state = {
  personas: [],
  current: null,
  prefs: {},
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
      status: { running: false, phase: "browser fallback" },
      paths: { config: "", persona_dir: "", avatar: "" },
    };
  },
  async get_prefs() { return {}; },
  async save_prefs() { return { ok: true }; },
  async get_log_tail() { return "Run through pywebview to read logs."; },
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
  pill.classList.remove("running", "warning", "stopped");
  if (status?.running) {
    pill.classList.add(status.stale ? "warning" : "running");
    label.textContent = status.stale ? "Unresponsive" : "Running";
  } else {
    pill.classList.add("stopped");
    label.textContent = text(status?.phase, "Stopped");
  }
}

function runtimeRows(status, summary) {
  return [
    ["Phase", text(status?.phase, status?.running ? "running" : "stopped")],
    ["PID", text(status?.pid, "none")],
    ["Provider", text(summary.provider_type, "unknown")],
    ["Provider model", text(summary.provider_model, "not set")],
    ["Last heartbeat", text(status?.last_heartbeat, "unknown")],
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
  el("traits").innerHTML = traits.length
    ? traits.map((trait) => `<span class="chip">${escapeHtml(trait)}</span>`).join("")
    : `<span class="chip">No traits listed</span>`;
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
  el("providerType").value = text(summary.provider_type);
  el("providerModel").value = text(summary.provider_model);
  el("maxContext").value = text(summary.max_context);
  el("ttsVoice").value = text(summary.tts?.voice_description);
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
  el("filePath").textContent = data.paths?.config ? `Read-only: ${data.paths.config}` : "Read-only Phase 1";
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

function escapeHtml(value) {
  return text(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
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
  el("refreshBtn").addEventListener("click", refreshAll);
  el("reloadBtn").addEventListener("click", refreshAll);
  document.querySelectorAll(".section-header").forEach((header) => {
    header.addEventListener("click", () => header.parentElement.classList.toggle("open"));
  });
  el("saveBtn").addEventListener("click", () => setNotice("Saving is intentionally disabled in Phase 1."));
  el("startBtn").addEventListener("click", () => setNotice("Starting Pulse is planned for the process-management phase."));
  el("stopBtn").addEventListener("click", () => setNotice("Stopping Pulse is planned for the process-management phase."));
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

window.addEventListener("pywebviewready", refreshAll);
window.addEventListener("DOMContentLoaded", () => {
  wireEvents();
  wireSliders();
  if (!window.pywebview) refreshAll();
});
