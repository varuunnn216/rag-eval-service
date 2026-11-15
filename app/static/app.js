const STORAGE_KEY = "rag_pipeline_settings_v1";

const DEFAULT_SETTINGS = {
  mode: "hybrid",
  useReranker: true,
  queryExpansion: true,
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

function clampArrayLen(n) {
  const num = Number(n);
  if (!Number.isFinite(num) || num <= 0) return 0;
  return Math.floor(num);
}

function isMarkdownPipeRow(line) {
  const t = String(line || "").trim();
  return t.startsWith("|") && t.indexOf("|", 1) !== -1;
}

function isMarkdownSeparatorRow(line) {
  const t = String(line || "").trim();
  return /^\|?(?:\s*:?-{2,}:?\s*\|)+\s*(?::?-{2,}:?\s*\|?)?$/.test(t) && t.includes("-");
}

function parseMarkdownTableCells(line, allowCitations) {
  let t = String(line || "").trim();
  if (t.startsWith("|")) t = t.slice(1);
  if (t.endsWith("|")) t = t.slice(0, -1);
  return t.split("|").map((c) => renderInlineMarkdown(escapeHtml(c.trim()), allowCitations));
}

function padTableRowCells(cells, targetLen) {
  const len = clampArrayLen(targetLen);
  const out = cells.slice();
  while (out.length < len) {
    out.push("");
  }
  return out;
}

function parseListLine(line) {
  const m = String(line || "").match(/^(\s*)(?:([-*•])|(\d+[.)]))\s+(.*)$/);
  if (!m) return null;
  return {
    indent: m[1].replace(/\t/g, "  ").length,
    type: m[3] ? "ol" : "ul",
    text: m[4].trim(),
  };
}

function renderListBlock(lines, startIdx, allowCitations) {
  let i = startIdx;
  const first = parseListLine(lines[i]);
  if (!first) return { html: "", nextIdx: i };

  function buildList(baseIndent, listType) {
    const items = [];
    while (i < lines.length) {
      const parsed = parseListLine(lines[i]);
      if (!parsed) break;
      if (parsed.indent < baseIndent) break;
      if (parsed.indent > baseIndent) {
        if (items.length > 0) {
          const sub = buildList(parsed.indent, parsed.type);
          items[items.length - 1] = items[items.length - 1].replace(/<\/li>$/, `${sub}</li>`);
          continue;
        }
      }
      if (parsed.type !== listType && parsed.indent === baseIndent) {
        break;
      }
      const content = renderInlineMarkdown(escapeHtml(parsed.text), allowCitations);
      i++;
      let nestedHtml = "";
      if (i < lines.length) {
        const nextParsed = parseListLine(lines[i]);
        if (nextParsed && nextParsed.indent > baseIndent) {
          nestedHtml = buildList(nextParsed.indent, nextParsed.type);
        }
      }
      items.push(`<li>${content}${nestedHtml}</li>`);
    }
    const tag = listType === "ol" ? "ol" : "ul";
    return `<${tag} class="md-list">${items.join("")}</${tag}>`;
  }

  const html = buildList(first.indent, first.type);
  return { html, nextIdx: i };
}

function truncateChunkPreview(rawText, maxChars = 320) {
  const text = normalizeEllipsis(String(rawText || "").trim());
  if (!text || text.length <= maxChars) {
    return { previewText: text, truncated: false };
  }

  const lines = text.split("\n");
  const out = [];
  let chars = 0;
  let inFence = false;
  let inTable = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    const isFenceToggle = trimmed.startsWith("```");
    const isTableLine = isMarkdownPipeRow(trimmed);

    if (isFenceToggle) {
      if (!inFence && chars > maxChars * 0.7 && out.length > 0) {
        break;
      }
      inFence = !inFence;
      out.push(line);
      chars += line.length + 1;
      if (!inFence && chars >= maxChars) break;
      continue;
    }

    if (inFence) {
      out.push(line);
      chars += line.length + 1;
      if (chars >= maxChars * 1.8) {
        out.push("```");
        inFence = false;
        break;
      }
      continue;
    }

    if (isTableLine) {
      if (!inTable && chars > maxChars * 0.75 && out.length > 0) {
        break;
      }
      inTable = true;
      out.push(line);
      chars += line.length + 1;
      const nextIsTable = i + 1 < lines.length && isMarkdownPipeRow(lines[i + 1]);
      if (chars >= maxChars && out.length >= 3 && !isMarkdownSeparatorRow(trimmed)) {
        break;
      }
      if (!nextIsTable) {
        inTable = false;
        if (chars >= maxChars) break;
      }
      continue;
    }

    inTable = false;
    if (chars + line.length + 1 > maxChars && out.length > 0) {
      break;
    }
    out.push(line);
    chars += line.length + 1;
    if (chars >= maxChars) break;
  }

  if (inFence) {
    out.push("```");
  }

  const previewText = out.join("\n").trimEnd();
  return {
    previewText: previewText || lines[0] || "",
    truncated: out.length < lines.length,
  };
}

function renderSafeMarkdown(rawText, options = {}) {
  const allowCitations = Boolean(options.allowCitations);
  try {
    const normalized = normalizeEllipsis(String(rawText || "").trim());
    if (!normalized) return "";

    // Extract closed fenced code blocks ```lang ... ``` first
    const fencePlaceholders = [];
    let withoutFences = normalized.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const idx = fencePlaceholders.length;
      const langTag = lang ? `<span class="md-code-lang">${escapeHtml(lang)}</span>` : "";
      fencePlaceholders.push(
        `<div class="md-code-wrap">${langTag}<pre class="md-code-block"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre></div>`
      );
      return `\n@@FENCE_${idx}@@\n`;
    });

    // Handle an unclosed / cut-off fenced code block at EOF
    withoutFences = withoutFences.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*)$/, (_, lang, code) => {
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
      const prevI = i;
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

      // Markdown Table block (with or without separator row, including ragged/empty/single rows)
      if (isMarkdownPipeRow(trimmed)) {
        const hasSeparator = i + 1 < lines.length && isMarkdownSeparatorRow(lines[i + 1]);
        if (hasSeparator) {
          const headerCells = parseMarkdownTableCells(trimmed, allowCitations);
          i += 2; // skip header + separator
          const rawBodyRows = [];
          while (i < lines.length && isMarkdownPipeRow(lines[i])) {
            if (!isMarkdownSeparatorRow(lines[i])) {
              rawBodyRows.push(parseMarkdownTableCells(lines[i], allowCitations));
            }
            i++;
          }
          const colCount = Math.max(
            headerCells.length,
            ...rawBodyRows.map((r) => r.length),
            1
          );
          const paddedHeaders = padTableRowCells(headerCells, colCount);
          const bodyHtml = rawBodyRows
            .map((row) => {
              const padded = padTableRowCells(row, colCount);
              return `<tr>${padded.map((c) => `<td>${c}</td>`).join("")}</tr>`;
            })
            .join("");
          htmlBlocks.push(
            `<div class="md-table-wrap"><table class="md-table"><thead><tr>${paddedHeaders
              .map((c) => `<th>${c}</th>`)
              .join("")}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`
          );
          continue;
        } else {
          // Ragged / missing separator row / single table row (e.g. citation quote)
          const rawRows = [];
          while (i < lines.length && isMarkdownPipeRow(lines[i])) {
            if (!isMarkdownSeparatorRow(lines[i])) {
              rawRows.push(parseMarkdownTableCells(lines[i], allowCitations));
            }
            i++;
          }
          if (rawRows.length > 0) {
            const colCount = Math.max(...rawRows.map((r) => r.length), 1);
            const bodyHtml = rawRows
              .map((row) => {
                const padded = padTableRowCells(row, colCount);
                return `<tr>${padded.map((c) => `<td>${c}</td>`).join("")}</tr>`;
              })
              .join("");
            htmlBlocks.push(
              `<div class="md-table-wrap"><table class="md-table"><tbody>${bodyHtml}</tbody></table></div>`
            );
            continue;
          }
        }
      }

      // Lists (unordered or ordered, including nested lists)
      if (parseListLine(rawLine)) {
        const { html, nextIdx } = renderListBlock(lines, i, allowCitations);
        if (html) htmlBlocks.push(html);
        i = nextIdx > i ? nextIdx : i + 1;
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
        !parseListLine(lines[i]) &&
        !/^#{1,4}\s+/.test(lines[i].trim()) &&
        !isMarkdownPipeRow(lines[i])
      ) {
        paraLines.push(renderInlineMarkdown(escapeHtml(lines[i].trim()), allowCitations));
        i++;
      }

      // Guaranteed progress invariant: never stall on an unrecognized line
      if (i === prevI) {
        paraLines.push(renderInlineMarkdown(escapeHtml(trimmed), allowCitations));
        i++;
      }

      if (paraLines.length > 0) {
        htmlBlocks.push(`<p class="md-para">${paraLines.join(" ")}</p>`);
      }
    }

    return htmlBlocks.join("");
  } catch (err) {
    if (typeof console !== "undefined" && console.error) {
      console.error("renderSafeMarkdown fallback triggered:", err);
    }
    return `<p class="md-para">${escapeHtml(String(rawText || ""))}</p>`;
  }
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

  const qeToggle = document.getElementById("query-expansion-toggle");
  if (qeToggle) {
    const qeOn = currentSettings.queryExpansion !== false;
    qeToggle.checked = qeOn;
    const qeLabel = document.getElementById("query-expansion-state-label");
    if (qeLabel) qeLabel.textContent = qeOn ? "On" : "Off";
  }

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
    "queryExpansion",
    (currentSettings.queryExpansion !== false) !== DEFAULT_SETTINGS.queryExpansion,
    `Query expansion (Off)`
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

  // Query expansion switch
  const qeToggleEl = document.getElementById("query-expansion-toggle");
  if (qeToggleEl) {
    qeToggleEl.addEventListener("change", (e) => {
      currentSettings.queryExpansion = e.target.checked;
      const qeLabel = document.getElementById("query-expansion-state-label");
      if (qeLabel) qeLabel.textContent = currentSettings.queryExpansion ? "On" : "Off";
      saveSettingsToStorage();
      updateSettingsDiffIndicators();
    });
  }

  // Show retrieved chunks anyway link on abstention card
  const showAbstainChunksBtn = document.getElementById("show-abstain-chunks-btn");
  if (showAbstainChunksBtn) {
    showAbstainChunksBtn.addEventListener("click", () => {
      const chunksSec = document.getElementById("section-chunks");
      if (chunksSec) {
        chunksSec.open = true;
        chunksSec.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

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

let activeAskController = null;
let activeAskRequestId = 0;
const chunkTextStates = new Map();

function clearPlaygroundResults() {
  lastAskResponse = null;
  chunkTextStates.clear();

  const answerEmpty = document.getElementById("answer-empty");
  const answerErr = document.getElementById("answer-error");
  const answerAbstain = document.getElementById("answer-abstain-box");
  const answerOut = document.getElementById("answer-output");
  if (answerEmpty) answerEmpty.classList.add("hidden");
  if (answerErr) {
    answerErr.classList.add("hidden");
    answerErr.innerHTML = "";
  }
  if (answerAbstain) answerAbstain.classList.add("hidden");
  if (answerOut) {
    answerOut.classList.add("hidden");
    answerOut.innerHTML = "";
  }

  const citErr = document.getElementById("citations-error");
  const citList = document.getElementById("citations-list");
  const citCount = document.getElementById("citation-count");
  if (citErr) {
    citErr.classList.add("hidden");
    citErr.innerHTML = "";
  }
  if (citList) citList.innerHTML = "";
  if (citCount) citCount.textContent = "0";

  const chkErr = document.getElementById("chunks-error");
  const chkList = document.getElementById("chunks-list");
  const chkCount = document.getElementById("chunk-count");
  const chkSummary = document.getElementById("context-used-summary");
  if (chkErr) {
    chkErr.classList.add("hidden");
    chkErr.innerHTML = "";
  }
  if (chkList) chkList.innerHTML = "";
  if (chkCount) chkCount.textContent = "0";
  if (chkSummary) chkSummary.textContent = "0 used / 0 retrieved";

  const metErr = document.getElementById("metrics-error");
  if (metErr) {
    metErr.classList.add("hidden");
    metErr.innerHTML = "";
  }
  [
    "metric-retrieval-ms",
    "metric-rerank-inline",
    "metric-llm-ms",
    "metric-total-ms",
    "metric-tokens-total",
    "metric-retrieval-card",
    "metric-llm-card",
    "metric-total-card",
    "metric-tokens-card",
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "—";
  });
  const promptEl = document.getElementById("prompt-sent-content");
  if (promptEl) promptEl.textContent = "";
  const eqRow = document.getElementById("expanded-queries-row");
  const eqList = document.getElementById("expanded-queries-list");
  if (eqRow) eqRow.classList.add("hidden");
  if (eqList) eqList.textContent = "";
  const fbRow = document.getElementById("fallback-used-row");
  const fbLabel = document.getElementById("fallback-used-label");
  if (fbRow) fbRow.classList.add("hidden");
  if (fbLabel) fbLabel.textContent = "";

  const copyAnsBtn = document.getElementById("copy-answer-btn");
  const copyJsonBtn = document.getElementById("copy-json-btn");
  if (copyAnsBtn) copyAnsBtn.disabled = true;
  if (copyJsonBtn) copyJsonBtn.disabled = true;
}

function renderSectionError(el, friendlyTitle, err, retryFnCall = "retryAskQuery()") {
  console.error(friendlyTitle, err);
  if (!el) return;
  const detailText = err && (err.stack || err.message) ? String(err.stack || err.message) : String(err || "Unknown error");
  el.innerHTML = `
    <div class="section-error-row">
      <span>${escapeHtml(friendlyTitle)}</span>
      <button type="button" class="secondary-btn error-retry-btn" onclick="${retryFnCall}">Retry</button>
    </div>
    <details class="error-details">
      <summary>Details</summary>
      <pre>${escapeHtml(detailText)}</pre>
    </details>
  `;
  el.classList.remove("hidden");
}

window.retryAskQuery = function () {
  runAskQuery();
};

window.retryAskSection = function (sectionName) {
  if (!lastAskResponse) {
    runAskQuery();
    return;
  }
  if (sectionName === "answer") {
    const el = document.getElementById("answer-error");
    if (el) el.classList.add("hidden");
    try {
      renderAnswerSection(lastAskResponse);
    } catch (err) {
      renderSectionError(el, "Could not render the answer.", err, "retryAskSection('answer')");
    }
  } else if (sectionName === "citations") {
    const el = document.getElementById("citations-error");
    if (el) el.classList.add("hidden");
    try {
      renderCitationsSection(lastAskResponse);
    } catch (err) {
      renderSectionError(el, "Could not render citations.", err, "retryAskSection('citations')");
    }
  } else if (sectionName === "chunks") {
    const el = document.getElementById("chunks-error");
    if (el) el.classList.add("hidden");
    try {
      renderChunksSection(lastAskResponse);
    } catch (err) {
      renderSectionError(el, "Could not render retrieved context.", err, "retryAskSection('chunks')");
    }
  } else if (sectionName === "metrics") {
    const el = document.getElementById("metrics-error");
    if (el) el.classList.add("hidden");
    try {
      renderMetricsSection(lastAskResponse);
    } catch (err) {
      renderSectionError(el, "Could not render run metrics.", err, "retryAskSection('metrics')");
    }
  }
};

async function runAskQuery() {
  const question = document.getElementById("question-input").value.trim();
  if (!question) return;

  if (activeAskController) {
    activeAskController.abort();
  }
  const controller = new AbortController();
  activeAskController = controller;
  const reqId = ++activeAskRequestId;

  clearPlaygroundResults();
  setPlaygroundLoadingState(true);
  updateStepperStage("retrieving");

  const payload = {
    question,
    mode: currentSettings.mode,
    top_k: currentSettings.topK,
    candidate_pool: currentSettings.candidatePool,
    use_reranker: currentSettings.useReranker,
    query_expansion: currentSettings.queryExpansion !== false,
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
      signal: controller.signal,
    });
    if (reqId !== activeAskRequestId) return;

    if (res.ok && res.headers.get("content-type")?.includes("text/event-stream") && res.body) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalData = null;

      while (true) {
        const { value, done } = await reader.read();
        if (reqId !== activeAskRequestId) return;
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
        if (reqId !== activeAskRequestId) return;
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
      signal: controller.signal,
    });
    if (reqId !== activeAskRequestId) return;

    if (!stdRes.ok) {
      const errJson = await stdRes.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${stdRes.status}`);
    }

    const data = await stdRes.json();
    if (reqId !== activeAskRequestId) return;
    lastAskResponse = data;
    setPlaygroundLoadingState(false);
    renderAskResponse(data);
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (reqId !== activeAskRequestId) return;
    setPlaygroundLoadingState(false);
    renderPlaygroundError(err);
  } finally {
    if (activeAskController === controller) {
      activeAskController = null;
    }
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
    const statePill = document.getElementById("answer-state-pill");
    statePill.className = "state-pill";
    statePill.textContent = "Running...";
  }
}

function renderPlaygroundError(err) {
  const statePill = document.getElementById("answer-state-pill");
  statePill.className = "state-pill pill-error";
  statePill.textContent = "Error";

  renderSectionError(
    document.getElementById("answer-error"),
    "Something went wrong while running your question.",
    err,
    "retryAskQuery()"
  );
  renderSectionError(
    document.getElementById("citations-error"),
    "Unable to verify citations for this request.",
    err,
    "retryAskQuery()"
  );
  renderSectionError(
    document.getElementById("chunks-error"),
    "Could not retrieve document context.",
    err,
    "retryAskQuery()"
  );
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

function renderAnswerSection(data) {
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
  const promptDrawer = document.getElementById("prompt-drawer");
  if (promptDrawer) {
    promptDrawer.classList.toggle("hidden", Boolean(window.__demoMode));
  }
  document.getElementById("prompt-sent-content").textContent =
    window.__demoMode ? "" : (data.prompt_sent || "(No LLM prompt sent)");

  const citSummary =
    data.citation_summary ||
    `Verified ${data.verified_citations_count || 0}/${data.total_citations_count || 0} citations`;

  if (data.error) {
    statePill.className = "state-pill pill-error";
    statePill.textContent = "LLM Error";
    renderSectionError(
      errorBox,
      "The language model could not generate an answer.",
      new Error(String(data.error)),
      "retryAskQuery()"
    );
  } else if (data.abstained) {
    statePill.className = "state-pill pill-abstained";
    statePill.textContent = "Not found in documents";
    document.getElementById("abstain-message-text").textContent =
      data.answer || "I don’t have enough information in the provided documents.";
    let reasonText = data.abstain_reason || "";
    if (!reasonText) {
      const chunks = data.retrieved_chunks || [];
      const usedChunks = chunks.filter((c) => Boolean(c.selected) || c.selection_status === "used");
      if (usedChunks.length > 0) {
        reasonText = `The model found no supporting evidence in the ${usedChunks.length} chunks it was given`;
      } else {
        const minSc = Number(
          (data.effective_settings && data.effective_settings.rerank_min_score) ??
            currentSettings.rerankMinScore
        );
        const bestSc = chunks.reduce(
          (mx, c) => Math.max(mx, Number(c.rerank_score ?? c.final_score ?? 0)),
          0
        );
        reasonText = `No relevant passage found (best rerank score ${bestSc.toFixed(2)}, below the ${minSc.toFixed(2)} minimum)`;
      }
    }
    document.getElementById("abstain-reason-detail").textContent = reasonText;
    abstainBox.classList.remove("hidden");
  } else {
    statePill.className = "state-pill pill-grounded";
    statePill.textContent = citSummary;
    answerBox.innerHTML = renderSafeMarkdown(data.answer, { allowCitations: true });
    answerBox.classList.remove("hidden");
  }
}

function renderAskResponse(data) {
  chunkTextStates.clear();

  try {
    renderAnswerSection(data);
  } catch (err) {
    const statePill = document.getElementById("answer-state-pill");
    if (statePill) {
      statePill.className = "state-pill pill-error";
      statePill.textContent = "Error";
    }
    renderSectionError(
      document.getElementById("answer-error"),
      "Could not display the generated answer.",
      err,
      "retryAskSection('answer')"
    );
  }

  try {
    renderCitationsSection(data);
  } catch (err) {
    renderSectionError(
      document.getElementById("citations-error"),
      "Could not verify or display citations for this answer.",
      err,
      "retryAskSection('citations')"
    );
  }

  showDroppedChunks = false;
  try {
    renderChunksSection(data);
  } catch (err) {
    renderSectionError(
      document.getElementById("chunks-error"),
      "Could not display retrieved document chunks.",
      err,
      "retryAskSection('chunks')"
    );
  }

  try {
    renderMetricsSection(data);
  } catch (err) {
    renderSectionError(
      document.getElementById("metrics-error"),
      "Could not display run metrics.",
      err,
      "retryAskSection('metrics')"
    );
  }
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
  const { previewText, truncated: needsExpand } = truncateChunkPreview(cleanContent, 320);
  const previewHtml = renderSafeMarkdown(previewText);
  const fullHtml = needsExpand ? renderSafeMarkdown(cleanContent) : previewHtml;
  chunkTextStates.set(contentId, { previewHtml, fullHtml, expanded: false });
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
      <div class="chunk-content markdown-body" id="${contentId}">${previewHtml}</div>
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
  const state = chunkTextStates.get(contentId);
  if (!el) return;
  if (state) {
    state.expanded = !state.expanded;
    el.innerHTML = state.expanded ? state.fullHtml : state.previewHtml;
    btnEl.textContent = state.expanded ? "Collapse text" : "Show full text";
    return;
  }
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

  const eqRow = document.getElementById("expanded-queries-row");
  const eqList = document.getElementById("expanded-queries-list");
  if (eqRow && eqList) {
    const rewrites = Array.isArray(data.expanded_queries) ? data.expanded_queries : [];
    if (rewrites.length > 0) {
      eqList.textContent = rewrites.join(" · ");
      eqRow.classList.remove("hidden");
    } else {
      eqList.textContent = "";
      eqRow.classList.add("hidden");
    }
  }

  const fbRow = document.getElementById("fallback-used-row");
  const fbLabel = document.getElementById("fallback-used-label");
  if (fbRow && fbLabel) {
    if (data.fallback_used === "lexical_evidence") {
      fbLabel.textContent = "lexical evidence";
      fbRow.classList.remove("hidden");
    } else {
      fbLabel.textContent = "";
      fbRow.classList.add("hidden");
    }
  }
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
    const qSetEl = document.getElementById("eval-question-set-select");
    const qDocEl = document.getElementById("eval-question-doc-select");
    const question_set = qSetEl ? qSetEl.value : "golden_only";
    const doc_id = question_set === "one_document" && qDocEl ? qDocEl.value : null;
    const payload = { question_set };
    if (selectedConfig) payload.config = selectedConfig;
    if (doc_id) payload.doc_id = doc_id;

    const res = await fetch("/api/v1/eval/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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

  const badgeEl = document.getElementById("eval-dataset-badge");
  if (badgeEl) {
    const qSetLabel = report.question_set || "golden_only";
    const totalQ = report.dataset_size || (activeRun.questions || []).length;
    const synCount = report.synthetic_count || 0;
    const synTag = synCount > 0 ? `<span class="badge badge-synthetic">Includes ${synCount} synthetic</span>` : `<span class="badge badge-golden">Golden only</span>`;
    badgeEl.innerHTML = `<span>Used <strong>${totalQ}</strong> approved questions (${escapeHtml(qSetLabel)})</span> ${synTag}`;
  }

  const typeGrid = document.getElementById("type-cards");
  if (typeGrid) {
    typeGrid.innerHTML = Object.entries(activeRun.by_type || {})
      .map(([qt, stats]) => {
        const r5 = stats.recall_at_5 != null ? `${(stats.recall_at_5 * 100).toFixed(1)}%` : "N/A (Trap)";
        return `
          <div class="kpi-box">
            <div class="kpi-label">${escapeHtml(qt)} (${stats.count} Qs)</div>
            <div class="kpi-val">${(stats.answer_accuracy * 100).toFixed(1)}%</div>
            <div class="kpi-sub">Answer Acc &bull; Recall@5: ${r5}</div>
          </div>
        `;
      })
      .join("");
  }

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
            <button type="button" class="secondary-btn btn-xs" onclick="openQuestionBankRow(${JSON.stringify(q.id).replace(/"/g, "&quot;")})">
              Question bank
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
            <button type="button" class="secondary-btn btn-xs" onclick="openQuestionBankRow(${JSON.stringify(q.id).replace(/"/g, "&quot;")})">
              Question bank
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
            class="secondary-btn btn-xs doc-generate-btn"
            data-gen-doc-id="${escapeHtml(d.id)}"
            ${window.__demoMode ? "disabled title='Disabled in DEMO_MODE'" : ""}
            onclick="startDocQuestionGeneration(${JSON.stringify(d.id).replace(/"/g, "&quot;")}, this)"
          >
            Generate Questions
          </button>
          <button
            type="button"
            class="secondary-btn btn-xs doc-delete-btn"
            onclick="promptDeleteDocument(${JSON.stringify(d.id).replace(/"/g, "&quot;")}, ${JSON.stringify(d.title).replace(/"/g, "&quot;")})"
          >
            Delete
          </button>
        </div>
        <div class="doc-gen-progress muted-text hidden" id="doc-gen-status-${escapeHtml(d.id)}"></div>
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


/* =====================================================================
   QUESTION BANK, GENERATION JOBS & EVAL SET SWITCHING
   ===================================================================== */
window.__demoMode = false;
let qbQuestions = [];
let qbActiveId = null;

window.startDocQuestionGeneration = async function (docId, btnEl) {
  if (window.__demoMode) {
    showToast("Question generation is disabled in DEMO_MODE.");
    return;
  }
  const statusEl = document.getElementById(`doc-gen-status-${docId}`);
  if (btnEl) btnEl.disabled = true;
  if (statusEl) {
    statusEl.classList.remove("hidden");
    statusEl.textContent = "Starting generation...";
  }
  try {
    const res = await fetch("/api/v1/eval/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, per_chunk: 2, include_unanswerable: true }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const job = await res.json();
    const poll = async () => {
      const stRes = await fetch(`/api/v1/eval/generate/status?job_id=${encodeURIComponent(job.job_id)}`);
      if (!stRes.ok) return;
      const st = await stRes.json();
      if (statusEl) statusEl.textContent = st.message || "Generating...";
      if (st.status === "running") {
        setTimeout(poll, 900);
      } else {
        if (btnEl) btnEl.disabled = false;
        showToast(st.status === "completed" ? `Generated questions for ${docId}: ${st.message}` : `Generation failed: ${st.error || "error"}`);
        await loadQuestionBank();
      }
    };
    setTimeout(poll, 500);
  } catch (err) {
    if (btnEl) btnEl.disabled = false;
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
    showToast(`Generation failed: ${err.message}`);
  }
};

function switchEvaluateSubtab(view) {
  const benchView = document.getElementById("eval-benchmark-view");
  const bankView = document.getElementById("eval-question-bank-view");
  const benchBtn = document.getElementById("eval-subtab-benchmark");
  const bankBtn = document.getElementById("eval-subtab-bank");
  const isBank = view === "bank";
  benchView?.classList.toggle("hidden", isBank);
  bankView?.classList.toggle("hidden", !isBank);
  benchBtn?.classList.toggle("active", !isBank);
  bankBtn?.classList.toggle("active", isBank);
  benchBtn?.setAttribute("aria-selected", String(!isBank));
  bankBtn?.setAttribute("aria-selected", String(isBank));
  if (isBank) loadQuestionBank();
}

window.openQuestionBankRow = async function (qid) {
  switchEvaluateSubtab("bank");
  await loadQuestionBank();
  const q = qbQuestions.find((x) => x.id === qid);
  if (q) openQuestionPreview(q);
};

async function loadQuestionBank() {
  try {
    const res = await fetch("/api/v1/eval/questions");
    if (!res.ok) return;
    const data = await res.json();
    qbQuestions = data.questions || [];
    window.__demoMode = Boolean(data.demo_mode);

    const demoBadge = document.getElementById("qb-demo-badge");
    const approveAllBtn = document.getElementById("qb-approve-visible-btn");
    const addBtn = document.getElementById("qb-add-toggle-btn");
    demoBadge?.classList.toggle("hidden", !window.__demoMode);
    if (approveAllBtn) approveAllBtn.disabled = window.__demoMode;
    if (addBtn) addBtn.disabled = window.__demoMode;

    const c = data.counts || {};
    const draftPill = document.getElementById("qb-draft-pill");
    if (draftPill) draftPill.textContent = String(c.draft || 0);
    const countsBar = document.getElementById("qb-counts-bar");
    if (countsBar) {
      countsBar.textContent = `Total: ${c.total || 0} · Golden: ${c.golden || 0} · Synthetic: ${c.synthetic || 0} · Approved: ${c.approved || 0} · Draft: ${c.draft || 0} · Rejected: ${c.rejected || 0}` + (c.needs_review ? ` · Needs review: ${c.needs_review}` : "");
    }
    populateQuestionBankDocFilters();
    renderQuestionBank();
  } catch (_) {}
}

function populateQuestionBankDocFilters() {
  const docIds = Array.from(new Set([
    ...indexedDocuments.map((d) => d.id),
    ...qbQuestions.map((q) => q.doc_id).filter(Boolean),
  ])).sort();

  const evalDocSel = document.getElementById("eval-question-doc-select");
  if (evalDocSel) {
    const prev = evalDocSel.value;
    evalDocSel.innerHTML = docIds.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
    if (prev && docIds.includes(prev)) evalDocSel.value = prev;
  }

  const filterDoc = document.getElementById("qb-filter-doc");
  if (filterDoc) {
    const prev = filterDoc.value;
    filterDoc.innerHTML = `<option value="all">All Docs</option>` + docIds.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
    if (prev && (prev === "all" || docIds.includes(prev))) filterDoc.value = prev;
  }

  const newDoc = document.getElementById("qb-new-doc");
  if (newDoc) {
    const prev = newDoc.value;
    newDoc.innerHTML = docIds.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
    if (prev && docIds.includes(prev)) newDoc.value = prev;
    if (newDoc.value) loadChunksIntoSelect(newDoc.value, "qb-new-chunks", []);
  }
}

function getFilteredQuestionBank() {
  const fDoc = document.getElementById("qb-filter-doc")?.value || "all";
  const fType = document.getElementById("qb-filter-type")?.value || "all";
  const fSrc = document.getElementById("qb-filter-source")?.value || "all";
  const fStatus = document.getElementById("qb-filter-status")?.value || "all";

  return qbQuestions.filter((q) => {
    if (fDoc !== "all" && q.doc_id !== fDoc) return false;
    if (fType !== "all" && q.type !== fType) return false;
    if (fSrc !== "all" && q.source !== fSrc) return false;
    if (fStatus === "needs_review" && !q.needs_review) return false;
    if (fStatus !== "all" && fStatus !== "needs_review" && q.status !== fStatus) return false;
    return true;
  });
}

function renderQuestionBank() {
  const tbody = document.getElementById("qb-tbody");
  if (!tbody) return;
  const rows = getFilteredQuestionBank();
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-table-cell">No questions match current filters.</td></tr>`;
    return;
  }
  const ro = window.__demoMode ? "disabled" : "";
  tbody.innerHTML = rows.map((q) => {
    const srcCls = q.source === "golden" ? "badge-golden" : "badge-synthetic";
    const statusLabel = q.document_missing ? "doc missing" : (q.needs_review ? "needs review" : q.status);
    return `
      <tr data-qb-id="${escapeHtml(q.id)}">
        <td>${escapeHtml(q.question)}</td>
        <td><span class="badge-cat">${escapeHtml(q.type || "single_chunk")}</span></td>
        <td><code>${escapeHtml(q.doc_id || "—")}</code></td>
        <td><span class="badge ${srcCls}">${escapeHtml(q.source)}</span></td>
        <td><span class="badge badge-status-${escapeHtml(q.status)}">${escapeHtml(statusLabel)}</span></td>
        <td class="qb-row-actions">
          <button type="button" class="secondary-btn btn-xs" onclick="openQuestionPreviewById(${JSON.stringify(q.id).replace(/"/g, "&quot;")})">Preview / Edit</button>
          <button type="button" class="secondary-btn btn-xs" ${ro} onclick="setQuestionStatus(${JSON.stringify(q.id).replace(/"/g, "&quot;")}, 'approved')">Approve</button>
          <button type="button" class="secondary-btn btn-xs" ${ro} onclick="setQuestionStatus(${JSON.stringify(q.id).replace(/"/g, "&quot;")}, 'rejected')">Reject</button>
        </td>
      </tr>
    `;
  }).join("");
}

async function loadChunksIntoSelect(docId, selectId, selectedIds) {
  const sel = document.getElementById(selectId);
  if (!sel || !docId) return;
  try {
    const res = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}/chunks`);
    if (!res.ok) return;
    const chunks = await res.json();
    const selSet = new Set(selectedIds || []);
    sel.innerHTML = chunks.map((c) => `<option value="${escapeHtml(c.chunk_id)}" ${selSet.has(c.chunk_id) ? "selected" : ""}>${escapeHtml(c.chunk_id)} — ${escapeHtml(c.heading || "")}</option>`).join("");
  } catch (_) {}
}

window.openQuestionPreviewById = function (qid) {
  const q = qbQuestions.find((x) => x.id === qid);
  if (q) openQuestionPreview(q);
};

async function openQuestionPreview(q) {
  qbActiveId = q.id;
  const panel = document.getElementById("qb-preview-panel");
  panel?.classList.remove("hidden");
  const idEl = document.getElementById("qb-preview-id");
  if (idEl) idEl.textContent = `${q.id} (${q.source} · ${q.status})`;

  const qInput = document.getElementById("qb-edit-question");
  const aInput = document.getElementById("qb-edit-answer");
  const fInput = document.getElementById("qb-edit-facts");
  if (qInput) qInput.value = q.question || "";
  if (aInput) aInput.value = q.expected_answer || "";
  if (fInput) fInput.value = (q.key_facts || []).join(", ");

  const goldEl = document.getElementById("qb-preview-gold-chunks");
  if (goldEl) {
    const refs = q.gold_refs || [];
    if (!refs.length) {
      goldEl.innerHTML = `<div class="muted-text">No gold chunks (unanswerable trap).</div>`;
    } else {
      let chunkContents = {};
      if (q.doc_id) {
        try {
          const r = await fetch(`/api/v1/documents/${encodeURIComponent(q.doc_id)}/chunks`);
          if (r.ok) {
            const list = await r.json();
            list.forEach((c) => { chunkContents[c.chunk_id] = c.content; });
          }
        } catch (_) {}
      }
      goldEl.innerHTML = refs.map((r) => `
        <div class="qb-gold-card">
          <div class="qb-gold-meta"><code>${escapeHtml(r.chunk_id || "")}</code> · ${escapeHtml(r.heading_path || "")}</div>
          <div class="qb-gold-text">${escapeHtml(chunkContents[r.chunk_id] || r.quote_snippet || "")}</div>
        </div>
      `).join("");
    }
  }
  if (q.doc_id) {
    await loadChunksIntoSelect(q.doc_id, "qb-edit-chunks", q.expected_chunk_ids || []);
  }
}

window.setQuestionStatus = async function (qid, status) {
  if (window.__demoMode) return;
  const res = await fetch(`/api/v1/eval/questions/${encodeURIComponent(qid)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (res.ok) {
    showToast(`Question ${qid} marked ${status}`);
    await loadQuestionBank();
  }
};

function initQuestionBank() {
  document.getElementById("eval-subtab-benchmark")?.addEventListener("click", () => switchEvaluateSubtab("benchmark"));
  document.getElementById("eval-subtab-bank")?.addEventListener("click", () => switchEvaluateSubtab("bank"));

  const qSetSel = document.getElementById("eval-question-set-select");
  const qDocSel = document.getElementById("eval-question-doc-select");
  qSetSel?.addEventListener("change", () => {
    qDocSel?.classList.toggle("hidden", qSetSel.value !== "one_document");
  });

  ["qb-filter-doc", "qb-filter-type", "qb-filter-source", "qb-filter-status"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", renderQuestionBank);
  });

  document.getElementById("qb-preview-close")?.addEventListener("click", () => {
    document.getElementById("qb-preview-panel")?.classList.add("hidden");
  });

  document.getElementById("qb-approve-visible-btn")?.addEventListener("click", async () => {
    if (window.__demoMode) return;
    const ids = getFilteredQuestionBank().filter((q) => q.status !== "approved").map((q) => q.id);
    if (!ids.length) {
      showToast("No unapproved visible questions.");
      return;
    }
    const res = await fetch("/api/v1/eval/questions/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, status: "approved" }),
    });
    if (res.ok) {
      showToast(`Approved ${ids.length} visible questions.`);
      await loadQuestionBank();
    }
  });

  document.getElementById("qb-add-toggle-btn")?.addEventListener("click", () => {
    document.getElementById("qb-add-form")?.classList.toggle("hidden");
  });
  document.getElementById("qb-add-cancel-btn")?.addEventListener("click", () => {
    document.getElementById("qb-add-form")?.classList.add("hidden");
  });
  document.getElementById("qb-new-doc")?.addEventListener("change", (e) => {
    loadChunksIntoSelect(e.target.value, "qb-new-chunks", []);
  });

  document.getElementById("qb-add-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (window.__demoMode) return;
    const q = document.getElementById("qb-new-question").value.trim();
    const type = document.getElementById("qb-new-type").value;
    const doc_id = document.getElementById("qb-new-doc").value;
    const chunksSel = document.getElementById("qb-new-chunks");
    const expected_chunk_ids = Array.from(chunksSel?.selectedOptions || []).map((o) => o.value);
    const expected_answer = document.getElementById("qb-new-answer").value.trim();
    const key_facts = document.getElementById("qb-new-facts").value.split(",").map((s) => s.trim()).filter(Boolean);
    const res = await fetch("/api/v1/eval/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, type, doc_id, expected_chunk_ids, expected_answer, key_facts, status: "approved" }),
    });
    if (res.ok) {
      document.getElementById("qb-add-form").classList.add("hidden");
      document.getElementById("qb-new-question").value = "";
      showToast("Added manual question to bank.");
      await loadQuestionBank();
    }
  });

  document.getElementById("qb-edit-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!qbActiveId || window.__demoMode) return;
    const question = document.getElementById("qb-edit-question").value.trim();
    const expected_answer = document.getElementById("qb-edit-answer").value.trim();
    const key_facts = document.getElementById("qb-edit-facts").value.split(",").map((s) => s.trim()).filter(Boolean);
    const chunksSel = document.getElementById("qb-edit-chunks");
    const expected_chunk_ids = Array.from(chunksSel?.selectedOptions || []).map((o) => o.value);
    const res = await fetch(`/api/v1/eval/questions/${encodeURIComponent(qbActiveId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, expected_answer, key_facts, expected_chunk_ids }),
    });
    if (res.ok) {
      showToast("Saved question updates.");
      await loadQuestionBank();
    }
  });

  document.getElementById("qb-edit-approve")?.addEventListener("click", () => qbActiveId && setQuestionStatus(qbActiveId, "approved"));
  document.getElementById("qb-edit-reject")?.addEventListener("click", () => qbActiveId && setQuestionStatus(qbActiveId, "rejected"));
  loadQuestionBank();
}

document.addEventListener("DOMContentLoaded", initQuestionBank);
