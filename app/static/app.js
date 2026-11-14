const STORAGE_KEY = "rag_pipeline_settings_v1";

const DEFAULT_SETTINGS = {
  mode: "hybrid",
  useReranker: true,
  topK: 5,
  candidatePool: 20,
  documentIds: [], // empty = All documents
  rerankMinScore: 0.25,
  rerankRelativeFactor: 0.6,
  rerankMaxChunks: 3,
  llmModel: "gemini-3.5-flash-lite",
  temperature: 0.0,
};

const ABLATION_LABELS = {
  dense_only: "Dense Only (pgvector)",
  sparse_only: "Sparse Only (tsvector)",
  hybrid_rrf: "Hybrid Search (RRF)",
  hybrid_rrf_rerank: "Hybrid (RRF) + Cross-Encoder Rerank",
};

const ABLATION_SHORT_LABELS = {
  dense_only: "Dense Only",
  sparse_only: "Sparse Only",
  hybrid_rrf: "Hybrid RRF",
  hybrid_rrf_rerank: "Hybrid + Rerank",
};

let currentSettings = { ...DEFAULT_SETTINGS, documentIds: [] };
let lastAskResponse = null;
let currentChunkSort = "final_rank";
let showDroppedChunks = false;
let indexedDocuments = [];
let currentEvalReport = null;
let lastUploadedFilename = "";
let pendingDeleteDocId = null;

function syncRequestJson(method, url, payload = null) {
  const xhr = new XMLHttpRequest();
  xhr.open(method, url, false);
  if (payload) {
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(payload));
  } else {
    xhr.send(null);
  }
  if (xhr.status >= 200 && xhr.status < 300) {
    return JSON.parse(xhr.responseText);
  }
  throw new Error(`HTTP ${xhr.status}`);
}

document.addEventListener("DOMContentLoaded", async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const isScreenshot = urlParams.get("screenshot") === "1";
  const vwParam = urlParams.get("vw");
  if (isScreenshot) {
    document.documentElement.dataset.screenshot = "1";
  }
  if (vwParam) {
    document.documentElement.dataset.vw = vwParam;
  }

  initTabs();
  loadSettingsFromStorage();
  initPlaygroundControls();
  initEvalControls();
  initCorpusControls();
  initModals();

  const initialTab = urlParams.get("tab");
  const initialQuery = urlParams.get("q");
  const openSettings = urlParams.get("settings") === "open";
  const openFailed = urlParams.get("failed") === "open";
  const openMobileNav = urlParams.get("mobilenav") === "open";

  if (initialQuery) {
    const qInput = document.getElementById("question-input");
    if (qInput) qInput.value = initialQuery;
  }
  if (openSettings) {
    const sSec = document.getElementById("sec-settings");
    if (sSec) sSec.open = true;
  }
  if (openFailed) {
    const fSec = document.getElementById("failed-questions-expander");
    if (fSec) fSec.open = true;
  }
  if (initialTab) {
    switchToTab(initialTab);
  }
  if (openMobileNav) {
    document.getElementById("top-nav-shell")?.classList.add("mobile-nav-open");
  }

  if (isScreenshot) {
    try {
      indexedDocuments = syncRequestJson("GET", "/api/v1/documents");
      updateCorpusSummaryLine();
      renderDocumentsList();
      renderDocScopeMenu();
    } catch (_) {}
    syncSettingsToUI();
    try {
      currentEvalReport = syncRequestJson("GET", "/api/v1/eval/report");
      renderEvalDashboard(currentEvalReport);
    } catch (_) {}
    try {
      const question = document.getElementById("question-input").value.trim();
      const askPayload = {
        question,
        mode: currentSettings.mode,
        top_k: currentSettings.topK,
        candidate_pool: currentSettings.candidatePool,
        use_reranker: currentSettings.useReranker,
        document_ids: currentSettings.documentIds.length > 0 ? currentSettings.documentIds : null,
        rerank_min_score: currentSettings.rerankMinScore,
        rerank_abstain_score: currentSettings.rerankMinScore,
        rerank_relative_factor: currentSettings.rerankRelativeFactor,
        rerank_max_chunks: currentSettings.rerankMaxChunks,
        llm_model: currentSettings.llmModel,
        temperature: currentSettings.temperature,
      };
      lastAskResponse = syncRequestJson("POST", "/api/v1/ask", askPayload);
      renderAskResponse(lastAskResponse);
    } catch (_) {}
    return;
  }

  await Promise.all([fetchHealth(), loadDocuments()]);
  syncSettingsToUI();
  await Promise.all([runAskQuery(), loadEvalReport()]);
});

/* =====================================================================
   SAFE SANITIZED MARKDOWN RENDERER (A.1)
   ===================================================================== */
function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function normalizeEllipsis(str) {
  return String(str || "").replace(/\.{4,}/g, "...");
}

function cleanQuoteText(quote) {
  let q = normalizeEllipsis(String(quote || "").trim());
  q = q.replace(/^\s*(?:[-*+•]|\d+[.)])\s+/, "");
  return q;
}

function renderInlineMarkdown(escapedLine, allowCitations = false) {
  // 1. Inline triple-backtick code ```lang ... ``` or inline code `...`
  const codePlaceholders = [];
  let out = escapedLine.replace(/```(?:[a-zA-Z0-9_-]+\s+)?([^`]+?)```/g, (_, codeContent) => {
    const token = `\u0000CODE_${codePlaceholders.length}\u0000`;
    codePlaceholders.push(`<code>${codeContent.trim()}</code>`);
    return token;
  });
  out = out.replace(/`([^`\n]+)`/g, (_, codeContent) => {
    const token = `\u0000CODE_${codePlaceholders.length}\u0000`;
    codePlaceholders.push(`<code>${codeContent}</code>`);
    return token;
  });

  // 2. Bold **...**
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // 3. Clickable [n] citations if enabled
  if (allowCitations) {
    out = out.replace(/\[(\d+)\]/g, (_, num) => {
      return `<button type="button" class="cite-badge" onclick="highlightCitation(${num})" title="Jump to Citation [${num}]">[${num}]</button>`;
    });
  }

  // Restore inline code
  out = out.replace(/\u0000CODE_(\d+)\u0000/g, (_, idx) => codePlaceholders[Number(idx)] || "");
  return out;
}

function renderSafeMarkdown(rawText, options = {}) {
  const allowCitations = Boolean(options.allowCitations);
  const normalized = normalizeEllipsis(String(rawText || "").trim());
  if (!normalized) return "";

  // Extract fenced code blocks ```lang ... ``` first
  const fencePlaceholders = [];
  const withoutFences = normalized.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const idx = fencePlaceholders.length;
    const langTag = lang ? `<span class="md-code-lang">${escapeHtml(lang)}</span>` : "";
    fencePlaceholders.push(
      `<div class="md-code-wrap">${langTag}<pre class="md-code-block"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre></div>`
    );
    return `\n@@FENCE_${idx}@@\n`;
  });

  const lines = withoutFences.split("\n");
  const htmlBlocks = [];
  let i = 0;

  while (i < lines.length) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // Fenced code placeholder
    const fenceMatch = trimmed.match(/^@@FENCE_(\d+)@@$/);
    if (fenceMatch) {
      htmlBlocks.push(fencePlaceholders[Number(fenceMatch[1])] || "");
      i++;
      continue;
    }

    // Markdown Table block
    if (trimmed.startsWith("|") && trimmed.endsWith("|") && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const headerCells = trimmed
        .slice(1, -1)
        .split("|")
        .map((c) => renderInlineMarkdown(escapeHtml(c.trim()), allowCitations));
      i += 2; // skip header + separator
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith("|") && lines[i].trim().endsWith("|")) {
        const rowCells = lines[i]
          .trim()
          .slice(1, -1)
          .split("|")
          .map((c) => renderInlineMarkdown(escapeHtml(c.trim()), allowCitations));
        bodyRows.push(`<tr>${rowCells.map((c) => `<td>${c}</td>`).join("")}</tr>`);
        i++;
      }
      htmlBlocks.push(
        `<div class="md-table-wrap"><table class="md-table"><thead><tr>${headerCells
          .map((c) => `<th>${c}</th>`)
          .join("")}</tr></thead><tbody>${bodyRows.join("")}</tbody></table></div>`
      );
      continue;
    }

    // Unordered list (- or * or •)
    if (/^\s*[-*•]\s+/.test(rawLine)) {
      const items = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        const itemText = lines[i].replace(/^\s*[-*•]\s+/, "").trim();
        items.push(`<li>${renderInlineMarkdown(escapeHtml(itemText), allowCitations)}</li>`);
        i++;
      }
      htmlBlocks.push(`<ul class="md-list">${items.join("")}</ul>`);
      continue;
    }

    // Ordered list (1. )
    if (/^\s*\d+[.)]\s+/.test(rawLine)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        const itemText = lines[i].replace(/^\s*\d+[.)]\s+/, "").trim();
        items.push(`<li>${renderInlineMarkdown(escapeHtml(itemText), allowCitations)}</li>`);
        i++;
      }
      htmlBlocks.push(`<ol class="md-list">${items.join("")}</ol>`);
      continue;
    }

    // Subheading inside chunk
    if (/^#{1,4}\s+/.test(trimmed)) {
      const hText = trimmed.replace(/^#{1,4}\s+/, "");
      htmlBlocks.push(`<div class="md-subheading">${renderInlineMarkdown(escapeHtml(hText), allowCitations)}</div>`);
      i++;
      continue;
    }

    // Standard paragraph (collect consecutive normal lines)
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^@@FENCE_\d+@@$/.test(lines[i].trim()) &&
      !/^\s*[-*•]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !(lines[i].trim().startsWith("|") && lines[i].trim().endsWith("|"))
    ) {
      paraLines.push(renderInlineMarkdown(escapeHtml(lines[i].trim()), allowCitations));
      i++;
    }
    htmlBlocks.push(`<p class="md-para">${paraLines.join(" ")}</p>`);
  }

  return htmlBlocks.join("");
}

/* =====================================================================
   TABS, SLIDING NAV PILL, SCROLL STATE & REVEAL
   ===================================================================== */
function updateNavActivePill(animate = true) {
  const pill = document.getElementById("nav-active-pill");
  const activeBtn = document.querySelector(".top-nav-menu .tab-btn.active");
  if (!pill) return;

  if (!activeBtn) {
    pill.classList.remove("is-visible");
    return;
  }

  const left = activeBtn.offsetLeft;
  const width = activeBtn.offsetWidth;
  if (!animate) {
    const prevTransition = pill.style.transition;
    pill.style.transition = "none";
    pill.style.width = `${width}px`;
    pill.style.transform = `translateX(${left}px)`;
    pill.classList.add("is-visible");
    void pill.offsetWidth;
    pill.style.transition = prevTransition;
  } else {
    pill.style.width = `${width}px`;
    pill.style.transform = `translateX(${left}px)`;
    pill.classList.add("is-visible");
  }
}

function initScrollHeaderAndReveal() {
  const navShell = document.getElementById("top-nav-shell");
  if (navShell) {
    const onScroll = () => {
      navShell.classList.toggle("is-scrolled", window.scrollY > 8);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  const revealEls = document.querySelectorAll(".reveal-on-scroll");
  if (!revealEls.length) return;

  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || typeof IntersectionObserver === "undefined") {
    revealEls.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08 }
  );
  revealEls.forEach((el) => observer.observe(el));
}

function switchToTab(tabId) {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  buttons.forEach((b) => {
    const isMatch = b.dataset.tab === tabId;
    b.classList.toggle("active", isMatch);
    if (isMatch) {
      b.setAttribute("aria-current", "page");
    } else {
      b.removeAttribute("aria-current");
    }
  });

  panels.forEach((p) => {
    p.classList.toggle("active", p.id === tabId);
  });

  requestAnimationFrame(() => updateNavActivePill(true));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      switchToTab(btn.dataset.tab);
    });
  });

  document.querySelectorAll("[data-goto-tab]").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      const targetTab = el.dataset.gotoTab;
      if (targetTab) switchToTab(targetTab);
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const targetTab = el.dataset.gotoTab;
        if (targetTab) switchToTab(targetTab);
      }
    });
  });

  const brandHome = document.getElementById("brand-home-link");
  if (brandHome) {
    brandHome.addEventListener("click", () => switchToTab("tab-home"));
  }

  window.addEventListener("resize", () => updateNavActivePill(false));
  initScrollHeaderAndReveal();
}

function shortModelName(fullId) {
  if (!fullId) return "MiniLM-L-6-v2";
  const parts = String(fullId).split("/");
  const last = parts[parts.length - 1];
  return last.replace(/^ms-marco-/, "");
}

async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const docsEl = document.getElementById("status-docs-count");
    const chunksEl = document.getElementById("status-chunks-count");
    const llmEl = document.getElementById("status-llm-model");
    const rerankShortEl = document.getElementById("status-reranker-short");

    if (docsEl) docsEl.textContent = `${data.documents_count} docs`;
    if (chunksEl) chunksEl.textContent = `${data.chunks_count} chunks`;
    if (llmEl && data.generation_model) llmEl.textContent = data.generation_model;

    const fullReranker = data.reranker_model || "cross-encoder/ms-marco-MiniLM-L-6-v2";
    const shortReranker = shortModelName(fullReranker);
    if (rerankShortEl) rerankShortEl.textContent = shortReranker;

    const helper = document.getElementById("reranker-model-helper");
    if (helper) {
      helper.setAttribute("aria-label", `Full model ID: ${fullReranker}`);
      helper.innerHTML = `
        <code>${escapeHtml(shortReranker)}</code>
        <span class="tooltip-popup" role="tooltip">
          Full ID: <code>${escapeHtml(fullReranker)}</code>
        </span>
      `;
    }
  } catch (err) {
    console.error("Failed to fetch health:", err);
  }
}

/* =====================================================================
   SETTINGS PERSISTENCE (localStorage) & DIFF DETECTION
   ===================================================================== */
function loadSettingsFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    currentSettings = {
      mode: parsed.mode || DEFAULT_SETTINGS.mode,
      useReranker:
        typeof parsed.useReranker === "boolean"
          ? parsed.useReranker
          : DEFAULT_SETTINGS.useReranker,
      topK: clampInt(parsed.topK, 1, 10, DEFAULT_SETTINGS.topK),
      candidatePool: clampInt(parsed.candidatePool, 5, 50, DEFAULT_SETTINGS.candidatePool),
      documentIds: Array.isArray(parsed.documentIds) ? parsed.documentIds : [],
      rerankMinScore: clampFloat(parsed.rerankMinScore, 0.05, 0.7, DEFAULT_SETTINGS.rerankMinScore),
      rerankRelativeFactor: clampFloat(
        parsed.rerankRelativeFactor,
        0.2,
        0.95,
        DEFAULT_SETTINGS.rerankRelativeFactor
      ),
      rerankMaxChunks: clampInt(parsed.rerankMaxChunks, 1, 6, DEFAULT_SETTINGS.rerankMaxChunks),
      llmModel: parsed.llmModel || DEFAULT_SETTINGS.llmModel,
      temperature: clampFloat(parsed.temperature, 0.0, 1.0, DEFAULT_SETTINGS.temperature),
    };
    if (currentSettings.candidatePool < currentSettings.topK) {
      currentSettings.candidatePool = Math.max(5, currentSettings.topK);
    }
  } catch (err) {
    currentSettings = { ...DEFAULT_SETTINGS, documentIds: [] };
  }
}

function saveSettingsToStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(currentSettings));
  } catch (err) {
    console.warn("Unable to persist settings in localStorage:", err);
  }
}

function clampInt(val, min, max, fallback) {
  const n = parseInt(val, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function clampFloat(val, min, max, fallback) {
  const n = parseFloat(val);
  if (Number.isNaN(n)) return fallback;
  return Math.max(min, Math.min(max, Number(n.toFixed(2))));
}

function syncSettingsToUI() {
  document.getElementById("mode-select").value = currentSettings.mode;

  const rerankToggle = document.getElementById("reranker-toggle");
  rerankToggle.checked = currentSettings.useReranker;
  document.getElementById("reranker-state-label").textContent = currentSettings.useReranker
    ? "Enabled"
    : "Disabled";

  const topkSlider = document.getElementById("topk-slider");
  topkSlider.value = String(currentSettings.topK);
  document.getElementById("topk-val").textContent = String(currentSettings.topK);

  const poolSlider = document.getElementById("pool-slider");
  const poolMin = Math.max(5, currentSettings.topK);
  poolSlider.min = String(poolMin);
  poolSlider.value = String(currentSettings.candidatePool);
  document.getElementById("pool-val").textContent = String(currentSettings.candidatePool);
  const poolMinTick = document.getElementById("pool-min-tick");
  if (poolMinTick) poolMinTick.textContent = String(poolMin);

  document.getElementById("rerank-min-slider").value = String(currentSettings.rerankMinScore);
  document.getElementById("rerank-min-val").textContent = currentSettings.rerankMinScore.toFixed(2);

  document.getElementById("rerank-ratio-slider").value = String(
    currentSettings.rerankRelativeFactor
  );
  document.getElementById("rerank-ratio-val").textContent =
    currentSettings.rerankRelativeFactor.toFixed(2);

  document.getElementById("rerank-max-slider").value = String(currentSettings.rerankMaxChunks);
  document.getElementById("rerank-max-val").textContent = String(currentSettings.rerankMaxChunks);

  document.getElementById("llm-model-select").value = currentSettings.llmModel;
  document.getElementById("temperature-slider").value = String(currentSettings.temperature);
  document.getElementById("temp-val").textContent = currentSettings.temperature.toFixed(2);

  const statusLlm = document.getElementById("status-llm-model");
  if (statusLlm) statusLlm.textContent = currentSettings.llmModel;

  renderDocScopeMenu();
  updateSettingsDiffIndicators();
}

function updateSettingsDiffIndicators() {
  const diffs = [];

  const checkDiff = (key, isDiff, label) => {
    const badge = document.getElementById(`diff-${key}`);
    const card = document.querySelector(`[data-setting-card="${key}"]`);
    if (badge) badge.classList.toggle("hidden", !isDiff);
    if (card) card.classList.toggle("modified", isDiff);
    if (isDiff) diffs.push(label);
  };

  checkDiff(
    "mode",
    currentSettings.mode !== DEFAULT_SETTINGS.mode,
    `Mode (${currentSettings.mode})`
  );
  checkDiff(
    "useReranker",
    currentSettings.useReranker !== DEFAULT_SETTINGS.useReranker,
    `Reranker (Off)`
  );
  checkDiff(
    "documentIds",
    currentSettings.documentIds.length > 0,
    `Scope (${currentSettings.documentIds.length} docs)`
  );
  checkDiff(
    "topK",
    currentSettings.topK !== DEFAULT_SETTINGS.topK,
    `Top-K (${currentSettings.topK})`
  );
  checkDiff(
    "candidatePool",
    currentSettings.candidatePool !== DEFAULT_SETTINGS.candidatePool,
    `Pool (${currentSettings.candidatePool})`
  );
  checkDiff(
    "rerankMinScore",
    Math.abs(currentSettings.rerankMinScore - DEFAULT_SETTINGS.rerankMinScore) > 1e-4,
    `Min Score (${currentSettings.rerankMinScore.toFixed(2)})`
  );
  checkDiff(
    "rerankRelativeFactor",
    Math.abs(currentSettings.rerankRelativeFactor - DEFAULT_SETTINGS.rerankRelativeFactor) > 1e-4,
    `Ratio vs Top (${currentSettings.rerankRelativeFactor.toFixed(2)})`
  );
  checkDiff(
    "rerankMaxChunks",
    currentSettings.rerankMaxChunks !== DEFAULT_SETTINGS.rerankMaxChunks,
    `Max Chunks (${currentSettings.rerankMaxChunks})`
  );

  const llmOrTempDiff =
    currentSettings.llmModel !== DEFAULT_SETTINGS.llmModel ||
    Math.abs(currentSettings.temperature - DEFAULT_SETTINGS.temperature) > 1e-4;
  checkDiff(
    "llmModelTemp",
    llmOrTempDiff,
    `LLM (${currentSettings.llmModel}, temp=${currentSettings.temperature.toFixed(2)})`
  );

  const summaryPill = document.getElementById("settings-diff-summary");
  const listText = document.getElementById("modified-settings-list");
  if (diffs.length === 0) {
    summaryPill.textContent = "Defaults active";
    summaryPill.classList.remove("has-diffs");
    listText.textContent = "All pipeline parameters match calibrated defaults.";
  } else {
    summaryPill.textContent = `${diffs.length} modified`;
    summaryPill.classList.add("has-diffs");
    listText.innerHTML = `<strong>Modified from defaults:</strong> ${diffs
      .map((d) => `<code>${escapeHtml(d)}</code>`)
      .join(" ")}`;
  }
}

/* =====================================================================
   PLAYGROUND CONTROLS & MULTI-SELECT DOCUMENT SCOPE
   ===================================================================== */
function initPlaygroundControls() {
  const questionInput = document.getElementById("question-input");
  questionInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runAskQuery();
    }
  });

  document.getElementById("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runAskQuery();
  });

  // B: Clicking a sample chip fills the query box and focuses it (does NOT auto-run)
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.dataset.preset;
      questionInput.focus();
      questionInput.setSelectionRange(questionInput.value.length, questionInput.value.length);
    });
  });

  // Mode
  document.getElementById("mode-select").addEventListener("change", (e) => {
    currentSettings.mode = e.target.value;
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  // Reranker switch
  document.getElementById("reranker-toggle").addEventListener("change", (e) => {
    currentSettings.useReranker = e.target.checked;
    document.getElementById("reranker-state-label").textContent = currentSettings.useReranker
      ? "Enabled"
      : "Disabled";
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  // Top-K & Candidate Pool
  const topkSlider = document.getElementById("topk-slider");
  const poolSlider = document.getElementById("pool-slider");

  topkSlider.addEventListener("input", () => {
    const newTopK = parseInt(topkSlider.value, 10);
    currentSettings.topK = newTopK;
    document.getElementById("topk-val").textContent = String(newTopK);
    const newMin = Math.max(5, newTopK);
    poolSlider.min = String(newMin);
    const poolMinTick = document.getElementById("pool-min-tick");
    if (poolMinTick) poolMinTick.textContent = String(newMin);
    if (currentSettings.candidatePool < newTopK) {
      currentSettings.candidatePool = newTopK;
      poolSlider.value = String(newTopK);
      document.getElementById("pool-val").textContent = String(newTopK);
    }
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  poolSlider.addEventListener("input", () => {
    let newPool = parseInt(poolSlider.value, 10);
    if (newPool < currentSettings.topK) {
      newPool = currentSettings.topK;
      poolSlider.value = String(newPool);
    }
    currentSettings.candidatePool = newPool;
    document.getElementById("pool-val").textContent = String(newPool);
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  // Rerank threshold sliders
  const minSlider = document.getElementById("rerank-min-slider");
  minSlider.addEventListener("input", () => {
    currentSettings.rerankMinScore = parseFloat(minSlider.value);
    document.getElementById("rerank-min-val").textContent =
      currentSettings.rerankMinScore.toFixed(2);
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  const ratioSlider = document.getElementById("rerank-ratio-slider");
  ratioSlider.addEventListener("input", () => {
    currentSettings.rerankRelativeFactor = parseFloat(ratioSlider.value);
    document.getElementById("rerank-ratio-val").textContent =
      currentSettings.rerankRelativeFactor.toFixed(2);
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  const maxSlider = document.getElementById("rerank-max-slider");
  maxSlider.addEventListener("input", () => {
    currentSettings.rerankMaxChunks = parseInt(maxSlider.value, 10);
    document.getElementById("rerank-max-val").textContent = String(
      currentSettings.rerankMaxChunks
    );
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  // LLM Model & Temperature
  document.getElementById("llm-model-select").addEventListener("change", (e) => {
    currentSettings.llmModel = e.target.value;
    const statusLlm = document.getElementById("status-llm-model");
    if (statusLlm) statusLlm.textContent = currentSettings.llmModel;
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  const tempSlider = document.getElementById("temperature-slider");
  tempSlider.addEventListener("input", () => {
    currentSettings.temperature = parseFloat(tempSlider.value);
    document.getElementById("temp-val").textContent = currentSettings.temperature.toFixed(2);
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  // Reset to defaults button
  document.getElementById("reset-settings-btn").addEventListener("click", () => {
    currentSettings = { ...DEFAULT_SETTINGS, documentIds: [] };
    saveSettingsToStorage();
    syncSettingsToUI();
  });

  // Multi-select Document Scope dropdown toggle
  const scopeBtn = document.getElementById("doc-scope-btn");
  const scopeMenu = document.getElementById("doc-scope-menu");
  scopeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isHidden = scopeMenu.classList.toggle("hidden");
    scopeBtn.setAttribute("aria-expanded", String(!isHidden));
  });
  document.addEventListener("click", (e) => {
    const ms = document.getElementById("doc-scope-multiselect");
    if (ms && !ms.contains(e.target)) {
      scopeMenu.classList.add("hidden");
      scopeBtn.setAttribute("aria-expanded", "false");
    }
  });

  // Copy buttons & Metrics Details expander
  document.getElementById("copy-answer-btn").addEventListener("click", async () => {
    if (!lastAskResponse) return;
    await copyTextWithFeedback("copy-answer-btn", lastAskResponse.answer || "", "Copy answer");
  });

  document.getElementById("copy-json-btn").addEventListener("click", async () => {
    if (!lastAskResponse) return;
    await copyTextWithFeedback(
      "copy-json-btn",
      JSON.stringify(lastAskResponse, null, 2),
      "Copy JSON"
    );
  });

  const detailsBtn = document.getElementById("toggle-metrics-details-btn");
  const detailsPanel = document.getElementById("metrics-details-panel");
  if (detailsBtn && detailsPanel) {
    detailsBtn.addEventListener("click", () => {
      const isHidden = detailsPanel.classList.toggle("hidden");
      detailsBtn.setAttribute("aria-expanded", String(!isHidden));
      detailsBtn.textContent = isHidden ? "Details" : "Hide details";
    });
  }

  // Sort toggle for Retrieved Context
  document.querySelectorAll(".sort-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sort-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentChunkSort = btn.dataset.sort;
      if (lastAskResponse) {
        renderChunksSection(lastAskResponse);
      }
    });
  });
}

async function copyTextWithFeedback(btnId, text, originalLabel) {
  const btn = document.getElementById(btnId);
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied!";
    setTimeout(() => {
      btn.textContent = originalLabel;
    }, 1400);
  } catch (err) {
    console.error("Clipboard copy failed:", err);
  }
}

function renderDocScopeMenu() {
  const menu = document.getElementById("doc-scope-menu");
  const summary = document.getElementById("doc-scope-summary");
  if (!menu || !summary) return;

  const validDocIds = new Set(indexedDocuments.map((d) => d.id));
  currentSettings.documentIds = currentSettings.documentIds.filter((id) => validDocIds.has(id));
  const allSelected = currentSettings.documentIds.length === 0;

  if (allSelected) {
    summary.textContent = `All documents (${indexedDocuments.length})`;
  } else if (currentSettings.documentIds.length === 1) {
    const match = indexedDocuments.find((d) => d.id === currentSettings.documentIds[0]);
    summary.textContent = match ? match.title : currentSettings.documentIds[0];
  } else {
    summary.textContent = `${currentSettings.documentIds.length} documents selected`;
  }

  const allOptionHtml = `
    <label class="multiselect-option">
      <input type="checkbox" data-doc-scope="all" ${allSelected ? "checked" : ""} />
      <span><strong>All documents (${indexedDocuments.length})</strong></span>
    </label>
  `;

  const docOptionsHtml = indexedDocuments
    .map((doc) => {
      const checked = !allSelected && currentSettings.documentIds.includes(doc.id);
      return `
        <label class="multiselect-option">
          <input type="checkbox" data-doc-scope="${escapeHtml(doc.id)}" ${checked ? "checked" : ""} />
          <span>${escapeHtml(doc.title)} <small class="muted-text">(${doc.chunk_count} chunks)</small></span>
        </label>
      `;
    })
    .join("");

  menu.innerHTML = allOptionHtml + docOptionsHtml;

  menu.querySelectorAll("input[type='checkbox']").forEach((cb) => {
    cb.addEventListener("change", (e) => {
      const val = e.target.dataset.docScope;
      if (val === "all") {
        currentSettings.documentIds = [];
      } else {
        if (e.target.checked) {
          if (!currentSettings.documentIds.includes(val)) {
            currentSettings.documentIds.push(val);
          }
        } else {
          currentSettings.documentIds = currentSettings.documentIds.filter((id) => id !== val);
        }
      }
      saveSettingsToStorage();
      renderDocScopeMenu();
      updateSettingsDiffIndicators();
    });
  });
}

/* =====================================================================
   EXECUTE RAG QUERY WITH PROGRESS STEPS & STREAMING (A.8)
   ===================================================================== */
function updateStepperStage(stage) {
  const stepRet = document.getElementById("step-retrieving");
  const stepRer = document.getElementById("step-reranking");
  const stepGen = document.getElementById("step-generating");
  const statusText = document.getElementById("stepper-status-text");
  if (!stepRet || !stepRer || !stepGen) return;

  stepRet.classList.remove("active", "done");
  stepRer.classList.remove("active", "done");
  stepGen.classList.remove("active", "done");

  if (stage === "retrieving") {
    stepRet.classList.add("active");
    if (statusText) statusText.textContent = "Searching dense + sparse indexes...";
  } else if (stage === "reranking") {
    stepRet.classList.add("done");
    stepRer.classList.add("active");
    if (statusText) statusText.textContent = "Scoring candidates with MiniLM-L-6-v2...";
  } else if (stage === "generating") {
    stepRet.classList.add("done");
    stepRer.classList.add("done");
    stepGen.classList.add("active");
    if (statusText) statusText.textContent = "Generating grounded answer with citations...";
  }
}

async function runAskQuery() {
  const question = document.getElementById("question-input").value.trim();
  if (!question) return;

  setPlaygroundLoadingState(true);
  updateStepperStage("retrieving");

  const payload = {
    question,
    mode: currentSettings.mode,
    top_k: currentSettings.topK,
    candidate_pool: currentSettings.candidatePool,
    use_reranker: currentSettings.useReranker,
    document_ids: currentSettings.documentIds.length ? currentSettings.documentIds : null,
    rerank_min_score: currentSettings.rerankMinScore,
    rerank_relative_factor: currentSettings.rerankRelativeFactor,
    rerank_max_chunks: currentSettings.rerankMaxChunks,
    llm_model: currentSettings.llmModel,
    temperature: currentSettings.temperature,
  };

  try {
    // Try SSE streaming endpoint first for live step transitions + streamed answer
    const res = await fetch("/api/v1/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok && res.headers.get("content-type")?.includes("text/event-stream") && res.body) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalData = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const chunkStr of parts) {
          const line = chunkStr.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          const evt = JSON.parse(jsonStr);
          if (evt.stage === "retrieving" || evt.stage === "reranking" || evt.stage === "generating") {
            updateStepperStage(evt.stage);
          } else if (evt.stage === "delta" && evt.text) {
            const ansBox = document.getElementById("answer-output");
            document.getElementById("answer-loading").classList.add("hidden");
            ansBox.classList.remove("hidden");
            ansBox.innerHTML = renderSafeMarkdown(evt.text, { allowCitations: true });
          } else if (evt.stage === "done" && evt.response) {
            finalData = evt.response;
          }
        }
      }

      if (finalData) {
        lastAskResponse = finalData;
        setPlaygroundLoadingState(false);
        renderAskResponse(finalData);
        return;
      }
    }

    // Fallback to standard JSON POST /api/v1/ask
    updateStepperStage("reranking");
    const stdRes = await fetch("/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!stdRes.ok) {
      const errJson = await stdRes.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${stdRes.status}`);
    }

    const data = await stdRes.json();
    lastAskResponse = data;
    setPlaygroundLoadingState(false);
    renderAskResponse(data);
  } catch (err) {
    setPlaygroundLoadingState(false);
    renderPlaygroundError(err.message || "Request failed");
  }
}

function setPlaygroundLoadingState(isLoading) {
  const submitBtn = document.getElementById("ask-submit-btn");
  const submitLabel = document.getElementById("ask-submit-label");
  submitBtn.disabled = isLoading;
  submitLabel.textContent = isLoading ? "Running..." : "Run";

  document.getElementById("ask-inline-loading").classList.toggle("hidden", !isLoading);
  document.getElementById("answer-loading").classList.toggle("hidden", !isLoading);
  document.getElementById("citations-loading").classList.toggle("hidden", !isLoading);
  document.getElementById("chunks-loading").classList.toggle("hidden", !isLoading);
  document.getElementById("metrics-loading").classList.toggle("hidden", !isLoading);

  if (isLoading) {
    document.getElementById("answer-empty").classList.add("hidden");
    document.getElementById("answer-error").classList.add("hidden");
    document.getElementById("answer-abstain-box").classList.add("hidden");
    document.getElementById("answer-output").classList.add("hidden");
    document.getElementById("citations-error").classList.add("hidden");
    document.getElementById("chunks-error").classList.add("hidden");
    document.getElementById("metrics-error").classList.add("hidden");
    const statePill = document.getElementById("answer-state-pill");
    statePill.className = "state-pill";
    statePill.textContent = "Running...";
  }
}

function renderPlaygroundError(errorMessage) {
  const statePill = document.getElementById("answer-state-pill");
  statePill.className = "state-pill pill-error";
  statePill.textContent = "Error";

  const ansErr = document.getElementById("answer-error");
  ansErr.textContent = `Pipeline Error: ${errorMessage}`;
  ansErr.classList.remove("hidden");

  const citErr = document.getElementById("citations-error");
  citErr.textContent = `Unable to verify citations: ${errorMessage}`;
  citErr.classList.remove("hidden");

  const chkErr = document.getElementById("chunks-error");
  chkErr.textContent = `Retrieval error: ${errorMessage}`;
  chkErr.classList.remove("hidden");
}

function formatPlainSelectionReason(reasonStr) {
  if (!reasonStr) return "";
  let s = String(reasonStr).trim();
  if (s.toLowerCase().startsWith("dropped:")) {
    s = "Dropped:" + s.slice(8);
  } else if (s.toLowerCase().startsWith("used:")) {
    s = "Used:" + s.slice(5);
  }
  return s;
}

function renderAskResponse(data) {
  const statePill = document.getElementById("answer-state-pill");
  const emptyBox = document.getElementById("answer-empty");
  const errorBox = document.getElementById("answer-error");
  const abstainBox = document.getElementById("answer-abstain-box");
  const answerBox = document.getElementById("answer-output");

  emptyBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  abstainBox.classList.add("hidden");
  answerBox.classList.add("hidden");

  document.getElementById("copy-answer-btn").disabled = false;
  document.getElementById("copy-json-btn").disabled = false;
  document.getElementById("prompt-sent-content").textContent =
    data.prompt_sent || "(No LLM prompt sent)";

  const citSummary =
    data.citation_summary ||
    `Verified ${data.verified_citations_count || 0}/${data.total_citations_count || 0} citations`;

  if (data.error) {
    statePill.className = "state-pill pill-error";
    statePill.textContent = "LLM Error";
    errorBox.innerHTML = `<strong>LLM Generation Error:</strong> ${escapeHtml(data.error)}`;
    errorBox.classList.remove("hidden");
  } else if (data.abstained) {
    statePill.className = "state-pill pill-abstained";
    statePill.textContent = "Not found in documents";
    document.getElementById("abstain-message-text").textContent =
      data.answer || "I don’t have enough information in the provided documents.";
    const topChunk = (data.retrieved_chunks || [])[0];
    const reasonText = topChunk
      ? formatPlainSelectionReason(topChunk.selection_reason)
      : "No matching chunks passed the minimum score threshold.";
    document.getElementById("abstain-reason-detail").textContent = reasonText;
    abstainBox.classList.remove("hidden");
  } else {
    statePill.className = "state-pill pill-grounded";
    statePill.textContent = citSummary;
    answerBox.innerHTML = renderSafeMarkdown(data.answer, { allowCitations: true });
    answerBox.classList.remove("hidden");
  }

  renderCitationsSection(data);
  showDroppedChunks = false;
  renderChunksSection(data);
  renderMetricsSection(data);
}

function renderCitationsSection(data) {
  const citList = document.getElementById("citations-list");
  const countEl = document.getElementById("citation-count");
  const citations = data.citations || [];
  if (countEl) countEl.textContent = String(citations.length);

  if (citations.length === 0) {
    const emptyMsg = data.abstained
      ? "Abstained — no citations emitted because the answer was not found in the indexed documents."
      : "No inline citations emitted for this response.";
    citList.innerHTML = `<div class="empty-state-box">${escapeHtml(emptyMsg)}</div>`;
    return;
  }

  citList.innerHTML = citations
    .map((c) => {
      const sim = Number(c.similarity_score || 0);
      let badgeClass = "sim-red";
      let badgeText = `Unverified (${sim.toFixed(2)})`;
      if (c.verified && sim >= 0.5) {
        badgeClass = "sim-green";
        badgeText = `Verified (${sim.toFixed(2)})`;
      } else if (c.verified && sim >= 0.25) {
        badgeClass = "sim-amber";
        badgeText = `Verified (${sim.toFixed(2)})`;
      }

      const cleanQuote = cleanQuoteText(c.quote);

      return `
        <div class="citation-card" id="citation-card-${c.citation_number}">
          <div class="citation-header">
            <div class="citation-header-left">
              <button type="button" class="cite-badge" onclick="highlightCitation(${c.citation_number})">[${c.citation_number}]</button>
              <strong>${escapeHtml(c.document_title)}</strong>
              <span class="sim-badge ${badgeClass}">${badgeText}</span>
            </div>
            <span class="chunk-id-tag">${escapeHtml(c.chunk_id)}</span>
          </div>
          ${
            c.claim
              ? `<div class="claim-row"><strong>Claim:</strong> ${renderInlineMarkdown(escapeHtml(c.claim), false)}</div>`
              : ""
          }
          <div class="quote-text markdown-body">${renderSafeMarkdown(cleanQuote, { allowCitations: false })}</div>
        </div>
      `;
    })
    .join("");
}

function renderSingleChunkCard(c) {
  const isUsed = Boolean(c.selected) || c.selection_status === "used";
  const cardIdNum = c.context_index != null ? c.context_index : `raw-${c._finalRank}`;
  const statusBadge = isUsed
    ? `<span class="chunk-status-badge used">USED [${c.context_index || c._finalRank}]</span>`
    : `<span class="chunk-status-badge dropped">DROPPED</span>`;

  const denseVal = c.dense_score != null ? Math.max(0, Math.min(1, Number(c.dense_score))) : null;
  const densePct = denseVal != null ? Math.round(denseVal * 100) : 0;
  const denseLabel =
    c.dense_rank != null && denseVal != null ? `#${c.dense_rank} · ${denseVal.toFixed(2)}` : "—";

  const sparseVal = c.sparse_score != null ? Math.max(0, Math.min(1, Number(c.sparse_score))) : null;
  const sparsePct = sparseVal != null ? Math.round(sparseVal * 100) : 0;
  const sparseLabel =
    c.sparse_rank != null && sparseVal != null ? `#${c.sparse_rank} · ${sparseVal.toFixed(2)}` : "—";

  const rrfVal = Number(c.rrf_score || 0);
  const rrfLabel = rrfVal > 0 ? `#${c._finalRank} · ${rrfVal.toFixed(4)}` : "—";

  const rerankVal = c.rerank_score != null ? Math.max(0, Math.min(1, Number(c.rerank_score))) : null;
  const rerankPct = rerankVal != null ? Math.round(rerankVal * 100) : 0;
  const rerankLabel =
    rerankVal != null ? `#${c._finalRank} · ${rerankVal.toFixed(2)}` : "Off";

  const contentId = `chunk-text-${c._finalRank}`;
  const cleanContent = normalizeEllipsis(c.content || "");
  const needsExpand = cleanContent.length > 220;
  const plainReason = formatPlainSelectionReason(c.selection_reason);

  return `
    <div class="chunk-card ${isUsed ? "chunk-used" : "chunk-dropped"}" id="chunk-card-${cardIdNum}">
      <div class="chunk-header">
        <div class="chunk-header-left">
          ${statusBadge}
          <strong>${escapeHtml(c.document_title)}</strong>
          <span class="muted-text">Rank #${c._finalRank} &middot; ${c.token_count || 0} tokens</span>
        </div>
        <span class="chunk-id-tag">${escapeHtml(c.chunk_id)}</span>
      </div>
      ${
        plainReason
          ? `<div class="selection-reason">${escapeHtml(plainReason)}</div>`
          : ""
      }
      ${c.heading ? `<div class="chunk-heading">${escapeHtml(c.heading)}</div>` : ""}
      <div class="chunk-content markdown-body ${needsExpand ? "collapsed" : ""}" id="${contentId}">${renderSafeMarkdown(cleanContent)}</div>
      ${
        needsExpand
          ? `<button type="button" class="expand-text-btn" onclick="toggleChunkText('${contentId}', this)">Show full text</button>`
          : ""
      }
      <div class="score-bars-grid">
        <div class="score-bar-item">
          <div class="score-bar-top"><span>Dense</span><strong>${denseLabel}</strong></div>
          <div class="score-bar-track"><div class="score-bar-fill fill-dense" style="width:${densePct}%"></div></div>
        </div>
        <div class="score-bar-item">
          <div class="score-bar-top"><span>Sparse</span><strong>${sparseLabel}</strong></div>
          <div class="score-bar-track"><div class="score-bar-fill fill-sparse" style="width:${sparsePct}%"></div></div>
        </div>
        <div class="score-bar-item">
          <div class="score-bar-top"><span>Rerank</span><strong>${rerankLabel}</strong></div>
          <div class="score-bar-track"><div class="score-bar-fill fill-rerank" style="width:${rerankPct}%"></div></div>
        </div>
        <div class="score-bar-item score-stat-only">
          <div class="score-bar-top"><span>RRF fusion</span><strong>${rrfLabel}</strong></div>
          <div class="rrf-stat-note">Rank fusion score (k=60)</div>
        </div>
      </div>
    </div>
  `;
}

function renderChunksSection(data) {
  const chunksList = document.getElementById("chunks-list");
  const rawChunks = (data.retrieved_chunks || []).map((c, idx) => ({
    ...c,
    _finalRank: idx + 1,
  }));

  const usedCount = rawChunks.filter((c) => c.selected || c.selection_status === "used").length;
  document.getElementById("chunk-count").textContent = String(rawChunks.length);
  document.getElementById("context-used-summary").textContent =
    `${usedCount} used / ${rawChunks.length} retrieved`;

  if (rawChunks.length === 0) {
    chunksList.innerHTML = `<div class="empty-state-box">No candidate chunks matched the selected document scope or minimum sparse threshold.</div>`;
    return;
  }

  const sortedChunks = [...rawChunks];
  if (currentChunkSort === "rerank_score") {
    sortedChunks.sort((a, b) => {
      const sa = a.rerank_score != null ? Number(a.rerank_score) : Number(a.final_score || 0);
      const sb = b.rerank_score != null ? Number(b.rerank_score) : Number(b.final_score || 0);
      return sb - sa;
    });
  } else {
    sortedChunks.sort((a, b) => a._finalRank - b._finalRank);
  }

  const usedChunks = sortedChunks.filter((c) => Boolean(c.selected) || c.selection_status === "used");
  const droppedChunks = sortedChunks.filter((c) => !(Boolean(c.selected) || c.selection_status === "used"));

  let html = "";
  if (usedChunks.length > 0) {
    html += usedChunks.map(renderSingleChunkCard).join("");
  } else {
    html += `<div class="empty-state-box">0 chunks passed the minimum rerank score threshold (${currentSettings.rerankMinScore.toFixed(2)}).</div>`;
  }

  if (droppedChunks.length > 0) {
    html += `
      <div class="dropped-chunks-collapsible">
        <button
          type="button"
          class="secondary-btn dropped-toggle-btn"
          id="toggle-dropped-chunks-btn"
          aria-expanded="${showDroppedChunks ? "true" : "false"}"
        >
          <span>${showDroppedChunks ? `Hide ${droppedChunks.length} dropped chunks` : `Show ${droppedChunks.length} dropped chunks`}</span>
        </button>
        <div id="dropped-chunks-container" class="chunks-list ${showDroppedChunks ? "" : "hidden"}">
          ${droppedChunks.map(renderSingleChunkCard).join("")}
        </div>
      </div>
    `;
  }

  chunksList.innerHTML = html;

  const toggleBtn = document.getElementById("toggle-dropped-chunks-btn");
  const droppedContainer = document.getElementById("dropped-chunks-container");
  if (toggleBtn && droppedContainer) {
    toggleBtn.addEventListener("click", () => {
      showDroppedChunks = !showDroppedChunks;
      droppedContainer.classList.toggle("hidden", !showDroppedChunks);
      toggleBtn.setAttribute("aria-expanded", String(showDroppedChunks));
      toggleBtn.innerHTML = `<span>${
        showDroppedChunks
          ? `Hide ${droppedChunks.length} dropped chunks`
          : `Show ${droppedChunks.length} dropped chunks`
      }</span>`;
    });
  }
}

window.toggleChunkText = function (contentId, btnEl) {
  const el = document.getElementById(contentId);
  if (!el) return;
  const isCollapsed = el.classList.toggle("collapsed");
  btnEl.textContent = isCollapsed ? "Show full text" : "Collapse text";
};

function renderMetricsSection(data) {
  const retMs = Number(data.retrieval_latency_ms || 0);
  const rerankMs = Number(data.rerank_latency_ms || 0);
  const llmMs = Number(data.generation_latency_ms || 0);
  const totalMs = Number(data.total_latency_ms || 0);
  const pTok = Number(data.prompt_tokens || 0);
  const cTok = Number(data.completion_tokens || 0);
  const totTok = Number(data.total_tokens || pTok + cTok);

  document.getElementById("metric-retrieval-ms").textContent = `${retMs.toFixed(1)} ms`;
  document.getElementById("metric-rerank-inline").textContent = `${rerankMs.toFixed(1)} ms`;
  document.getElementById("metric-llm-ms").textContent = `${llmMs.toFixed(0)} ms`;
  document.getElementById("metric-total-ms").textContent = `${totalMs.toFixed(0)} ms`;
  document.getElementById("metric-tokens-total").textContent = String(totTok);

  const retCard = document.getElementById("metric-retrieval-card");
  const llmCard = document.getElementById("metric-llm-card");
  const totCard = document.getElementById("metric-total-card");
  const tokCard = document.getElementById("metric-tokens-card");
  if (retCard) retCard.textContent = `${retMs.toFixed(1)} ms`;
  if (llmCard) llmCard.textContent = `${llmMs.toFixed(0)} ms`;
  if (totCard) totCard.textContent = `${totalMs.toFixed(0)} ms`;
  if (tokCard) tokCard.textContent = String(totTok);

  document.getElementById("metric-rerank-sub").textContent = `Rerank: ${rerankMs.toFixed(1)} ms`;
  document.getElementById("metric-model-sub").textContent =
    (data.effective_settings && data.effective_settings.llm_model) || currentSettings.llmModel;
  document.getElementById("metric-status-sub").textContent = data.abstained
    ? "Abstained"
    : "Grounded";
  document.getElementById("metric-tokens-sub").textContent =
    `${pTok} prompt / ${cTok} out`;
}

window.highlightCitation = function (num) {
  document.querySelectorAll(".citation-card, .chunk-card").forEach((el) => {
    el.classList.remove("highlight");
  });
  const citSection = document.getElementById("section-citations");
  if (citSection && !citSection.open) citSection.open = true;

  const citCard = document.getElementById(`citation-card-${num}`);
  const chunkCard = document.getElementById(`chunk-card-${num}`);
  if (citCard) {
    citCard.classList.add("highlight");
    citCard.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  if (chunkCard) {
    chunkCard.classList.add("highlight");
  }
};

/* =====================================================================
   TAB 2: EVALUATION HARNESS (A.4 & D)
   ===================================================================== */
function updateEvalRunButtonLabels() {
  const selectEl = document.getElementById("ablation-config-select");
  const runSelectedBtn = document.getElementById("run-selected-eval-btn");
  if (!selectEl || !runSelectedBtn) return;
  const shortName = ABLATION_SHORT_LABELS[selectEl.value] || selectEl.value;
  runSelectedBtn.textContent = `Run selected (${shortName})`;
}

function initEvalControls() {
  const selectEl = document.getElementById("ablation-config-select");
  selectEl.addEventListener("change", () => {
    updateEvalRunButtonLabels();
    if (currentEvalReport) renderEvalDashboard(currentEvalReport);
  });
  updateEvalRunButtonLabels();

  document.getElementById("eval-category-filter").addEventListener("change", () => {
    if (currentEvalReport) renderQuestionsTable(currentEvalReport);
  });
  document.getElementById("eval-search-filter").addEventListener("input", () => {
    if (currentEvalReport) renderQuestionsTable(currentEvalReport);
  });

  // Run selected configuration only
  document.getElementById("run-selected-eval-btn").addEventListener("click", async () => {
    await triggerEvaluationRun(selectEl.value);
  });

  // Run all 4 configurations
  document.getElementById("run-eval-btn").addEventListener("click", async () => {
    await triggerEvaluationRun(null);
  });
}

async function triggerEvaluationRun(selectedConfig) {
  const runAllBtn = document.getElementById("run-eval-btn");
  const runSelBtn = document.getElementById("run-selected-eval-btn");
  const evalLoading = document.getElementById("eval-loading");
  const evalError = document.getElementById("eval-error");

  runAllBtn.disabled = true;
  runSelBtn.disabled = true;
  evalError.classList.add("hidden");
  evalLoading.classList.remove("hidden");

  if (selectedConfig) {
    runSelBtn.textContent = `Running ${ABLATION_SHORT_LABELS[selectedConfig] || selectedConfig}...`;
  } else {
    runAllBtn.textContent = "Running all 4...";
  }

  try {
    const res = await fetch("/api/v1/eval/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selectedConfig ? { config: selectedConfig } : {}),
    });
    if (!res.ok) {
      throw new Error(`Evaluation HTTP ${res.status}`);
    }
    currentEvalReport = await res.json();
    renderEvalDashboard(currentEvalReport);
    showToast(
      selectedConfig
        ? `Completed evaluation for ${ABLATION_LABELS[selectedConfig]}`
        : `Completed evaluation across all 4 configurations`
    );
  } catch (err) {
    evalError.textContent = `Evaluation failed: ${err.message}`;
    evalError.classList.remove("hidden");
  } finally {
    evalLoading.classList.add("hidden");
    runAllBtn.disabled = false;
    runSelBtn.disabled = false;
    runAllBtn.textContent = "Run all 4";
    updateEvalRunButtonLabels();
  }
}

async function loadEvalReport() {
  const evalLoading = document.getElementById("eval-loading");
  const evalError = document.getElementById("eval-error");
  try {
    evalLoading.classList.remove("hidden");
    const res = await fetch("/api/v1/eval/report");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    currentEvalReport = await res.json();
    renderEvalDashboard(currentEvalReport);
  } catch (err) {
    if (evalError) {
      evalError.textContent = `Failed to load evaluation report: ${err.message}`;
      evalError.classList.remove("hidden");
    }
  } finally {
    if (evalLoading) evalLoading.classList.add("hidden");
  }
}

function renderEvalDashboard(report) {
  const selectedKey = document.getElementById("ablation-config-select").value;
  const runs = report.ablation_runs || {};
  const activeRun = runs[selectedKey] || runs.hybrid_rrf_rerank;
  if (!activeRun) return;

  const totalQs = activeRun.total_questions || report.dataset_size || 60;
  const ansQs = activeRun.answerable_questions || 50;
  const trapQs = activeRun.unanswerable_questions || 10;

  const subEl = document.getElementById("eval-subtitle-text");
  if (subEl) {
    subEl.textContent = `Benchmark retrieval and answer accuracy across ${totalQs} test questions (${ansQs} answerable, ${trapQs} unanswerable traps) and 4 ablation configurations.`;
  }

  const m = activeRun.metrics;

  // Group 1: Retrieval (Recall@5, MRR@5, nDCG@5) — 3 cards, no orphan
  document.getElementById("eval-retrieval-kpis").innerHTML = `
    <div class="kpi-box">
      <div class="kpi-label">Recall@5</div>
      <div class="kpi-val">${(m.recall_at_5 * 100).toFixed(1)}%</div>
      <div class="kpi-sub">${ansQs} answerable questions</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">MRR@5</div>
      <div class="kpi-val">${m.mrr_at_5.toFixed(3)}</div>
      <div class="kpi-sub">Mean Reciprocal Rank</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">nDCG@5</div>
      <div class="kpi-val">${m.ndcg_at_5.toFixed(3)}</div>
      <div class="kpi-sub">Rank quality discount</div>
    </div>
  `;

  // Group 2: Answer quality (Accuracy, Faithfulness, Citation recall, Abstention) — 4 cards, no orphan
  document.getElementById("eval-answer-kpis").innerHTML = `
    <div class="kpi-box">
      <div class="kpi-label">Accuracy</div>
      <div class="kpi-val">${(m.answer_accuracy * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Key-fact coverage (${totalQs} QA)</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Faithfulness</div>
      <div class="kpi-val">${(m.faithfulness * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Grounded in cited chunks</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Citation recall</div>
      <div class="kpi-val">${(m.citation_recall * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Ground-truth chunks cited</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Abstention</div>
      <div class="kpi-val">${(m.abstention_accuracy * 100).toFixed(1)}%</div>
      <div class="kpi-sub">${trapQs} unanswerable trap queries</div>
    </div>
  `;

  // Compute best values per column across all 4 ablation configs
  const availKeys = Object.keys(ABLATION_LABELS).filter((k) => runs[k]);
  const metricKeysHigherBetter = [
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "answer_accuracy",
    "faithfulness",
    "citation_recall",
    "abstention_accuracy",
  ];
  const bestVals = {};
  metricKeysHigherBetter.forEach((mk) => {
    bestVals[mk] = Math.max(...availKeys.map((k) => Number(runs[k].metrics[mk] ?? -1)));
  });
  const getLat = (rm) => Number(rm.avg_retrieval_latency_ms ?? rm.avg_latency_ms ?? 9999);
  const bestLatency = Math.min(...availKeys.map((k) => getLat(runs[k].metrics)));

  const fmtCell = (val, bestVal, text, lowerBetter = false) => {
    const isBest = lowerBetter
      ? Math.abs(val - bestVal) <= 0.02
      : Math.abs(val - bestVal) <= 0.0005;
    return `<td class="${isBest ? "best-metric-cell" : ""}">${text}</td>`;
  };

  document.getElementById("ablation-tbody").innerHTML = availKeys
    .map((k) => {
      const rm = runs[k].metrics;
      const isSelected = k === selectedKey;
      const lat = getLat(rm);
      return `
        <tr class="${isSelected ? "row-selected" : ""}">
          <td class="sticky-col"><strong>${ABLATION_LABELS[k]}</strong></td>
          ${fmtCell(rm.recall_at_1, bestVals.recall_at_1, `${(rm.recall_at_1 * 100).toFixed(1)}%`)}
          ${fmtCell(rm.recall_at_3, bestVals.recall_at_3, `${(rm.recall_at_3 * 100).toFixed(1)}%`)}
          ${fmtCell(rm.recall_at_5, bestVals.recall_at_5, `${(rm.recall_at_5 * 100).toFixed(1)}%`)}
          ${fmtCell(rm.mrr_at_5, bestVals.mrr_at_5, rm.mrr_at_5.toFixed(3))}
          ${fmtCell(rm.ndcg_at_5, bestVals.ndcg_at_5, rm.ndcg_at_5.toFixed(3))}
          ${fmtCell(rm.answer_accuracy, bestVals.answer_accuracy, `${(rm.answer_accuracy * 100).toFixed(1)}%`)}
          ${fmtCell(rm.faithfulness, bestVals.faithfulness, `${(rm.faithfulness * 100).toFixed(1)}%`)}
          ${fmtCell(rm.citation_recall, bestVals.citation_recall, `${(rm.citation_recall * 100).toFixed(1)}%`)}
          ${fmtCell(rm.abstention_accuracy, bestVals.abstention_accuracy, `${(rm.abstention_accuracy * 100).toFixed(1)}%`)}
          ${fmtCell(lat, bestLatency, `${lat.toFixed(2)} ms`, true)}
        </tr>
      `;
    })
    .join("");

  const catGrid = document.getElementById("category-cards");
  catGrid.innerHTML = Object.entries(activeRun.by_category || {})
    .map(([cat, stats]) => {
      const r5 =
        stats.recall_at_5 != null ? `${(stats.recall_at_5 * 100).toFixed(1)}%` : "N/A (Trap)";
      return `
        <div class="kpi-box">
          <div class="kpi-label">${escapeHtml(cat)} (${stats.count} Qs)</div>
          <div class="kpi-val">${(stats.answer_accuracy * 100).toFixed(1)}%</div>
          <div class="kpi-sub">Answer Acc &bull; Recall@5: ${r5}</div>
        </div>
      `;
    })
    .join("");

  renderFailedQuestionsSection(activeRun, selectedKey);
  renderQuestionsTable(report);
}

function renderFailedQuestionsSection(activeRun, selectedKey) {
  const tbody = document.getElementById("failed-questions-tbody");
  const countBadge = document.getElementById("failed-questions-count");
  const cfgLabel = document.getElementById("failed-config-label");
  if (!tbody) return;

  const configName = ABLATION_LABELS[selectedKey] || selectedKey;
  if (cfgLabel) cfgLabel.textContent = configName;

  const questions = activeRun.questions || [];
  const failed = questions.filter((q) => {
    if (q.should_abstain) {
      return !q.abstained || q.answer_accuracy < 1.0;
    }
    return q.recall_at_5 < 1.0 || q.answer_accuracy < 1.0;
  });

  if (countBadge) countBadge.textContent = String(failed.length);

  if (failed.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-table-cell">All questions passed for ${escapeHtml(configName)}.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = failed
    .map((q) => {
      const expectedDesc = q.should_abstain
        ? "Should abstain (unanswerable trap)"
        : q.expected_answer
        ? `${q.expected_answer} (Key facts: ${(q.key_facts || []).join(", ")})`
        : `Chunks: ${(q.expected_chunk_ids || []).join(", ")} | Facts: ${(q.key_facts || []).join(", ")}`;

      const gotDesc = q.abstained
        ? `Abstained (${(q.answer_accuracy * 100).toFixed(0)}% acc)`
        : `${q.answer || "—"} (${(q.answer_accuracy * 100).toFixed(0)}% fact match, Recall@5 ${(q.recall_at_5 * 100).toFixed(0)}%)`;

      return `
        <tr>
          <td><strong>${escapeHtml(q.id)}</strong></td>
          <td>${escapeHtml(q.question)}</td>
          <td><div class="failed-cell-text">${escapeHtml(expectedDesc)}</div></td>
          <td><div class="failed-cell-text">${escapeHtml(gotDesc)}</div></td>
          <td><span class="badge-cat">${escapeHtml(ABLATION_SHORT_LABELS[selectedKey] || selectedKey)}</span></td>
          <td>
            <button type="button" class="secondary-btn btn-xs" onclick="tryQuestionInPlayground(${JSON.stringify(q.question).replace(/"/g, "&quot;")})">
              Test Live
            </button>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderQuestionsTable(report) {
  const selectedKey = document.getElementById("ablation-config-select").value;
  const activeRun = (report.ablation_runs || {})[selectedKey];
  if (!activeRun) return;

  const catFilter = document.getElementById("eval-category-filter").value;
  const textFilter = document
    .getElementById("eval-search-filter")
    .value.trim()
    .toLowerCase();

  const filtered = (activeRun.questions || []).filter((q) => {
    if (catFilter !== "all" && q.category !== catFilter) return false;
    if (textFilter) {
      const hay = `${q.id} ${q.question} ${(q.expected_chunk_ids || []).join(" ")}`.toLowerCase();
      if (!hay.includes(textFilter)) return false;
    }
    return true;
  });

  document.getElementById("questions-tbody").innerHTML = filtered
    .map((q) => {
      const expStr = q.expected_chunk_ids.length
        ? q.expected_chunk_ids.map((id) => `<code>${escapeHtml(id)}</code>`).join("<br/>")
        : `<span class="muted-text">None (Should Abstain)</span>`;
      const topRet = q.retrieved_chunk_ids[0]
        ? `<code>${escapeHtml(q.retrieved_chunk_ids[0])}</code>`
        : "—";
      return `
        <tr>
          <td><strong>${escapeHtml(q.id)}</strong></td>
          <td><span class="badge-cat">${escapeHtml(q.category)}</span></td>
          <td>${escapeHtml(q.question)}</td>
          <td>${expStr}</td>
          <td>${topRet}</td>
          <td>${q.should_abstain ? "—" : `${(q.recall_at_5 * 100).toFixed(0)}%`}</td>
          <td>${q.should_abstain ? "—" : q.mrr_at_5.toFixed(2)}</td>
          <td><strong>${(q.answer_accuracy * 100).toFixed(0)}%</strong></td>
          <td>
            <button type="button" class="secondary-btn btn-xs" onclick="tryQuestionInPlayground(${JSON.stringify(q.question).replace(/"/g, "&quot;")})">
              Test Live
            </button>
          </td>
        </tr>
      `;
    })
    .join("");
}

window.tryQuestionInPlayground = function (questionText) {
  switchToTab("tab-playground");
  const qInput = document.getElementById("question-input");
  qInput.value = questionText;
  runAskQuery();
};

/* =====================================================================
   TAB 3: CORPUS & DOCUMENT INGESTION (A.5)
   ===================================================================== */
function initCorpusControls() {
  document.getElementById("refresh-docs-btn").addEventListener("click", loadDocuments);

  const searchInput = document.getElementById("doc-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      renderDocumentsList();
    });
  }

  const fileInput = document.getElementById("file-upload-input");
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      lastUploadedFilename = file.name;
      const reader = new FileReader();
      reader.onload = () => {
        const contentText = String(reader.result || "");
        const baseName = file.name.replace(/\.[^.]+$/, "");
        const cleanSlug =
          "doc_" +
          baseName
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "_")
            .replace(/^_+|_+$/g, "");
        const idEl = document.getElementById("ingest-id");
        const sourceEl = document.getElementById("ingest-source");
        if (!idEl.value.trim()) idEl.value = cleanSlug;
        if (!sourceEl.value.trim()) sourceEl.value = `upload/${file.name}`;
        document.getElementById("ingest-content").value = contentText;
        const statusEl = document.getElementById("uploaded-file-status");
        if (statusEl) {
          statusEl.textContent = `Loaded "${file.name}" (${file.size} bytes) — click "Index document" below.`;
        }
      };
      reader.readAsText(file);
    });
  }

  document.getElementById("ingest-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("ingest-submit-btn");
    const feedback = document.getElementById("ingest-feedback");
    const progBox = document.getElementById("ingest-progress-box");
    const progFill = document.getElementById("ingest-progress-fill");
    const progLabel = document.getElementById("ingest-progress-label");

    btn.disabled = true;
    btn.textContent = "Indexing...";
    feedback.classList.add("hidden");
    progBox.classList.remove("hidden");
    progFill.style.width = "35%";
    progLabel.textContent = "Chunking & computing 768-d embeddings...";

    const rawId = document.getElementById("ingest-id").value.trim();
    const rawTitle = document.getElementById("ingest-title").value.trim();
    const rawSource = document.getElementById("ingest-source").value.trim();

    const doc = {
      id: rawId,
      document_id: rawId,
      title: rawTitle,
      source: rawSource,
      content: document.getElementById("ingest-content").value,
      metadata: {
        ingested_via: "web_ui",
        filename: lastUploadedFilename || rawSource || "uploaded.md",
      },
    };

    try {
      setTimeout(() => {
        progFill.style.width = "75%";
        progLabel.textContent = "Upserting into hybrid index...";
      }, 120);

      const res = await fetch("/api/v1/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: [doc] }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      progFill.style.width = "100%";
      progLabel.textContent = `Indexed ${data.chunks_created} chunks.`;

      feedback.textContent = `Indexed ${data.documents_ingested} document (${data.chunks_created} chunks created): ${data.document_ids.join(", ")}`;
      feedback.classList.remove("hidden");
      showToast(`Indexed document (${data.chunks_created} chunks created)`);

      await loadDocuments();
      await fetchHealth();
    } catch (err) {
      feedback.textContent = `Error: ${err.message}`;
      feedback.classList.remove("hidden");
    } finally {
      setTimeout(() => {
        progBox.classList.add("hidden");
        progFill.style.width = "0%";
      }, 600);
      btn.disabled = false;
      btn.textContent = "Index document";
    }
  });
}

async function loadDocuments() {
  const loadingEl = document.getElementById("docs-loading");
  const errorEl = document.getElementById("docs-error");
  try {
    if (loadingEl) loadingEl.classList.remove("hidden");
    if (errorEl) errorEl.classList.add("hidden");
    const res = await fetch("/api/v1/documents");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    indexedDocuments = await res.json();
    updateCorpusSummaryLine();
    renderDocumentsList();
    renderDocScopeMenu();
  } catch (err) {
    if (errorEl) {
      errorEl.textContent = `Failed to load documents: ${err.message}`;
      errorEl.classList.remove("hidden");
    }
  } finally {
    if (loadingEl) loadingEl.classList.add("hidden");
  }
}

function updateCorpusSummaryLine() {
  const totalDocs = indexedDocuments.length;
  const totalChunks = indexedDocuments.reduce((acc, d) => acc + Number(d.chunk_count || 0), 0);
  const summaryEl = document.getElementById("corpus-summary-line");
  if (summaryEl) {
    summaryEl.textContent = `${totalDocs} document${totalDocs === 1 ? "" : "s"} · ${totalChunks} chunks`;
  }
  const statusDocs = document.getElementById("status-docs-count");
  const statusChunks = document.getElementById("status-chunks-count");
  if (statusDocs) statusDocs.textContent = `${totalDocs} docs`;
  if (statusChunks) statusChunks.textContent = `${totalChunks} chunks`;
}

function renderDocumentsList() {
  const container = document.getElementById("documents-list");
  if (!container) return;

  const query = (document.getElementById("doc-search-input")?.value || "").trim().toLowerCase();
  const filtered = indexedDocuments.filter((d) => {
    if (!query) return true;
    const hay = `${d.title} ${d.id} ${d.source}`.toLowerCase();
    return hay.includes(query);
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty-state-box">No documents match "${escapeHtml(query)}".</div>`;
    return;
  }

  // Identical card layout for every doc: 1. Title, 2. ID chip, 3. Source & chunk count, 4. Actions
  container.innerHTML = filtered
    .map(
      (d) => `
      <div class="doc-card" data-doc-id="${escapeHtml(d.id)}">
        <div class="doc-card-title">${escapeHtml(d.title)}</div>
        <div class="doc-card-id-row">
          <span class="chunk-id-tag">${escapeHtml(d.id)}</span>
        </div>
        <div class="doc-card-meta">
          <span>Source: <code>${escapeHtml(d.source)}</code></span>
          <span class="status-sep">&middot;</span>
          <span><strong>${d.chunk_count}</strong> chunks</span>
        </div>
        <div class="doc-card-actions">
          <button
            type="button"
            class="secondary-btn btn-xs"
            onclick="openDocumentChunksModal(${JSON.stringify(d.id).replace(/"/g, "&quot;")}, ${JSON.stringify(d.title).replace(/"/g, "&quot;")})"
          >
            View chunks
          </button>
          <button
            type="button"
            class="secondary-btn btn-xs doc-delete-btn"
            onclick="promptDeleteDocument(${JSON.stringify(d.id).replace(/"/g, "&quot;")}, ${JSON.stringify(d.title).replace(/"/g, "&quot;")})"
          >
            Delete
          </button>
        </div>
      </div>
    `
    )
    .join("");
}

/* =====================================================================
   MODALS (VIEW CHUNKS & DELETE CONFIRM) & TOASTS (A.5)
   ===================================================================== */
function initModals() {
  const chunksModal = document.getElementById("doc-chunks-modal");
  const closeChunksBtn = document.getElementById("close-chunks-modal-btn");
  if (closeChunksBtn && chunksModal) {
    closeChunksBtn.addEventListener("click", () => chunksModal.classList.add("hidden"));
    chunksModal.addEventListener("click", (e) => {
      if (e.target === chunksModal) chunksModal.classList.add("hidden");
    });
  }

  const delModal = document.getElementById("delete-confirm-modal");
  const cancelDelBtn = document.getElementById("cancel-delete-btn");
  const confirmDelBtn = document.getElementById("confirm-delete-btn");
  if (cancelDelBtn && delModal) {
    cancelDelBtn.addEventListener("click", () => {
      pendingDeleteDocId = null;
      delModal.classList.add("hidden");
    });
    delModal.addEventListener("click", (e) => {
      if (e.target === delModal) {
        pendingDeleteDocId = null;
        delModal.classList.add("hidden");
      }
    });
  }
  if (confirmDelBtn && delModal) {
    confirmDelBtn.addEventListener("click", async () => {
      if (!pendingDeleteDocId) return;
      const docId = pendingDeleteDocId;
      confirmDelBtn.disabled = true;
      confirmDelBtn.textContent = "Deleting...";
      try {
        const res = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}`, {
          method: "DELETE",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        delModal.classList.add("hidden");
        showToast(`Deleted document "${docId}"`);
        await loadDocuments();
        await fetchHealth();
      } catch (err) {
        showToast(`Failed to delete document: ${err.message}`);
      } finally {
        pendingDeleteDocId = null;
        confirmDelBtn.disabled = false;
        confirmDelBtn.textContent = "Delete document";
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      chunksModal?.classList.add("hidden");
      delModal?.classList.add("hidden");
    }
  });
}

window.openDocumentChunksModal = async function (docId, docTitle) {
  const modal = document.getElementById("doc-chunks-modal");
  const titleEl = document.getElementById("doc-chunks-modal-title");
  const subEl = document.getElementById("doc-chunks-modal-sub");
  const bodyEl = document.getElementById("doc-chunks-modal-body");
  if (!modal || !bodyEl) return;

  titleEl.textContent = docTitle || docId;
  subEl.textContent = `ID: ${docId}`;
  bodyEl.innerHTML = `<div class="skeleton-block"><div class="skeleton-card"></div></div>`;
  modal.classList.remove("hidden");

  try {
    const res = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}/chunks`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const chunks = await res.json();
    subEl.textContent = `ID: ${docId} · ${chunks.length} chunks indexed`;
    if (chunks.length === 0) {
      bodyEl.innerHTML = `<div class="empty-state-box">No chunks found for this document.</div>`;
      return;
    }
    bodyEl.innerHTML = chunks
      .map(
        (c) => `
        <div class="chunk-card">
          <div class="chunk-header">
            <span class="chunk-id-tag">${escapeHtml(c.chunk_id)}</span>
            <span class="muted-text">${c.token_count || 0} tokens</span>
          </div>
          ${c.heading ? `<div class="chunk-heading">${escapeHtml(c.heading)}</div>` : ""}
          <div class="chunk-content markdown-body">${renderSafeMarkdown(c.content)}</div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    bodyEl.innerHTML = `<div class="error-callout">Failed to load chunks: ${escapeHtml(err.message)}</div>`;
  }
};

window.promptDeleteDocument = function (docId, docTitle) {
  pendingDeleteDocId = docId;
  const modal = document.getElementById("delete-confirm-modal");
  const desc = document.getElementById("delete-modal-desc");
  if (desc) {
    desc.innerHTML = `Are you sure you want to delete <strong>${escapeHtml(docTitle)}</strong> (<code>${escapeHtml(docId)}</code>) and all of its indexed chunks?`;
  }
  modal?.classList.remove("hidden");
};

function showToast(message) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast-item";
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("toast-exit");
    setTimeout(() => toast.remove(), 260);
  }, 3200);
}
