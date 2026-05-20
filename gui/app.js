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
      loadPersona(persona.name);
    });
    menu.appendChild(btn);
  });
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
      <div>
        <div class="skill-name">${escapeHtml(skill.label)}</div>
        <div class="skill-state">${skill.enabled ? "Enabled" : "Disabled"}</div>
      </div>
    </div>
  `).join("");
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
  el("personaPicker").addEventListener("click", () => el("personaMenu").classList.toggle("hidden"));
  el("refreshBtn").addEventListener("click", refreshAll);
  el("reloadBtn").addEventListener("click", refreshAll);
  document.querySelectorAll(".section-header").forEach((header) => {
    header.addEventListener("click", () => header.parentElement.classList.toggle("open"));
  });
  el("saveBtn").addEventListener("click", () => setNotice("Saving is intentionally disabled in Phase 1."));
  el("startBtn").addEventListener("click", () => setNotice("Starting Pulse is planned for the process-management phase."));
  el("stopBtn").addEventListener("click", () => setNotice("Stopping Pulse is planned for the process-management phase."));
}

window.addEventListener("pywebviewready", refreshAll);
window.addEventListener("DOMContentLoaded", () => {
  wireEvents();
  if (!window.pywebview) refreshAll();
});
