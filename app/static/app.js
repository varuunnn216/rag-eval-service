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

let currentSettings = { ...DEFAULT_SETTINGS, documentIds: [] };
let lastAskResponse = null;
let currentChunkSort = "final_rank";
let indexedDocuments = [];
let currentEvalReport = null;
let lastUploadedFilename = "";

document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  loadSettingsFromStorage();
  initPlaygroundControls();
  initEvalControls();
  initCorpusControls();

  await Promise.all([fetchHealth(), loadDocuments()]);
  syncSettingsToUI();
  runAskQuery();
  loadEvalReport();
});

/* =====================================================================
   TABS & SYSTEM HEALTH
   ===================================================================== */
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const docsPill = document.getElementById("pill-docs");
    const modelsPill = document.getElementById("pill-models");
    if (docsPill) {
      docsPill.textContent = `${data.documents_count} Docs`;
    }
    if (modelsPill) {
      modelsPill.textContent = `${data.generation_model}`;
    }
    const helper = document.getElementById("reranker-model-helper");
    if (helper && data.reranker_model) {
      helper.innerHTML = `Model: <code>${escapeHtml(data.reranker_model)}</code>`;
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
    : "Disabled (Raw Rank)";

  const topkSlider = document.getElementById("topk-slider");
  topkSlider.value = String(currentSettings.topK);
  document.getElementById("topk-val").textContent = String(currentSettings.topK);

  const poolSlider = document.getElementById("pool-slider");
  poolSlider.min = String(Math.max(5, currentSettings.topK));
  poolSlider.value = String(currentSettings.candidatePool);
  document.getElementById("pool-val").textContent = String(currentSettings.candidatePool);

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

  renderDocScopeMenu();
  updateSettingsDiffIndicators();
}

function updateSettingsDiffIndicators() {
  const diffs = [];

  const checkDiff = (key, isDiff, label) => {
    const badge = document.getElementById(`diff-${key}`);
    const card = document.querySelector(`[data-setting-card="${key}"]`);
    if (badge) badge.classList.toggle("hidden", !isDiff);
    if (card) card.classList.toggle("is-modified", isDiff);
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
    summaryPill.classList.remove("has-diff");
    listText.textContent = "All pipeline parameters match calibrated defaults.";
  } else {
    summaryPill.textContent = `${diffs.length} modified`;
    summaryPill.classList.add("has-diff");
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

  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.dataset.preset;
      runAskQuery();
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
      : "Disabled (Raw Rank)";
    saveSettingsToStorage();
    updateSettingsDiffIndicators();
  });

  // Top-K & Candidate Pool (enforce candidatePool >= topK)
  const topkSlider = document.getElementById("topk-slider");
  const poolSlider = document.getElementById("pool-slider");

  topkSlider.addEventListener("input", () => {
    const newTopK = parseInt(topkSlider.value, 10);
    currentSettings.topK = newTopK;
    document.getElementById("topk-val").textContent = String(newTopK);
    poolSlider.min = String(Math.max(5, newTopK));
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
    if (!document.getElementById("doc-scope-multiselect").contains(e.target)) {
      scopeMenu.classList.add("hidden");
      scopeBtn.setAttribute("aria-expanded", "false");
    }
  });

  // Copy buttons
  document.getElementById("copy-answer-btn").addEventListener("click", async () => {
    if (!lastAskResponse) return;
    await copyTextWithFeedback("copy-answer-btn", lastAskResponse.answer || "", "Copy answer");
  });

  document.getElementById("copy-json-btn").addEventListener("click", async () => {
    if (!lastAskResponse) return;
    await copyTextWithFeedback(
      "copy-json-btn",
      JSON.stringify(lastAskResponse, null, 2),
      "Copy as JSON"
    );
  });

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
   EXECUTE RAG QUERY & RENDER SECTIONS 3, 4, 5, 6
   ===================================================================== */
async function runAskQuery() {
  const question = document.getElementById("question-input").value.trim();
  if (!question) return;

  setPlaygroundLoadingState(true);

  try {
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

    const res = await fetch("/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
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
    // Distinct Amber "Not found in documents" state
    statePill.className = "state-pill pill-abstained";
    statePill.textContent = "Not found in documents";
    document.getElementById("abstain-message-text").textContent =
      data.answer || "I don't have enough information in the provided documents.";
    const topChunk = (data.retrieved_chunks || [])[0];
    const reasonText = topChunk
      ? `Reason: ${topChunk.selection_reason || "Context lacked sufficient evidence"}`
      : "Reason: No matching chunks passed the minimum threshold.";
    document.getElementById("abstain-reason-detail").textContent = reasonText;
    abstainBox.classList.remove("hidden");
  } else {
    statePill.className = "state-pill pill-grounded";
    statePill.textContent = `Grounded • ${citSummary}`;
    answerBox.innerHTML = formatAnswerWithClickableCitations(data.answer);
    answerBox.classList.remove("hidden");
  }

  renderCitationsSection(data, citSummary);
  renderChunksSection(data);
  renderMetricsSection(data);
}

function renderCitationsSection(data, citSummary) {
  const citList = document.getElementById("citations-list");
  document.getElementById("citation-count").textContent = citSummary;

  if (!data.citations || data.citations.length === 0) {
    const emptyMsg = data.abstained
      ? "Abstained (0 citations emitted — query could not be answered from documents)."
      : `No inline citations emitted (${citSummary}).`;
    citList.innerHTML = `<div class="empty-state-box">${escapeHtml(emptyMsg)}</div>`;
    return;
  }

  citList.innerHTML = data.citations
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
              ? `<div class="claim-row"><strong>Claim:</strong> ${escapeHtml(c.claim)}</div>`
              : ""
          }
          <div class="quote-text">&ldquo;${escapeHtml(c.quote)}&rdquo;</div>
        </div>
      `;
    })
    .join("");
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

  // Compute max RRF score for normalizing the RRF bar visually
  const maxRrf = Math.max(0.035, ...sortedChunks.map((c) => Number(c.rrf_score || 0)));

  chunksList.innerHTML = sortedChunks
    .map((c) => {
      const isUsed = Boolean(c.selected) || c.selection_status === "used";
      const cardIdNum = c.context_index != null ? c.context_index : `raw-${c._finalRank}`;
      const statusBadge = isUsed
        ? `<span class="chunk-status-badge used">USED [${c.context_index || c._finalRank}]</span>`
        : `<span class="chunk-status-badge dropped">DROPPED</span>`;

      const denseVal = c.dense_score != null ? Math.max(0, Number(c.dense_score)) : 0;
      const densePct = Math.min(100, Math.round(denseVal * 100));
      const denseLabel =
        c.dense_rank != null ? `#${c.dense_rank} (${denseVal.toFixed(2)})` : "—";

      const sparseVal = c.sparse_score != null ? Math.max(0, Number(c.sparse_score)) : 0;
      const sparsePct = Math.min(100, Math.round(sparseVal * 100));
      const sparseLabel =
        c.sparse_rank != null ? `#${c.sparse_rank} (${sparseVal.toFixed(2)})` : "—";

      const rrfVal = Number(c.rrf_score || 0);
      const rrfPct = rrfVal > 0 ? Math.min(100, Math.round((rrfVal / maxRrf) * 100)) : 0;
      const rrfLabel = rrfVal > 0 ? `#${c._finalRank} (${rrfVal.toFixed(3)})` : "—";

      const rerankVal = c.rerank_score != null ? Math.max(0, Number(c.rerank_score)) : null;
      const rerankPct = rerankVal != null ? Math.min(100, Math.round(rerankVal * 100)) : 0;
      const rerankLabel =
        rerankVal != null ? `#${c._finalRank} (${rerankVal.toFixed(3)})` : "Off";

      const contentId = `chunk-text-${c._finalRank}`;
      const needsExpand = (c.content || "").length > 180;

      return `
        <div class="chunk-card" id="chunk-card-${cardIdNum}">
          <div class="chunk-header">
            <div class="chunk-header-left">
              ${statusBadge}
              <strong>${escapeHtml(c.document_title)}</strong>
              <span class="muted-text" style="font-size:0.78rem;">Rank #${c._finalRank}</span>
            </div>
            <span class="chunk-id-tag">${escapeHtml(c.chunk_id)}</span>
          </div>
          ${
            c.selection_reason
              ? `<div class="selection-reason">${escapeHtml(c.selection_reason)}</div>`
              : ""
          }
          ${c.heading ? `<div class="chunk-heading">${escapeHtml(c.heading)}</div>` : ""}
          <div class="chunk-content ${needsExpand ? "collapsed" : ""}" id="${contentId}">${escapeHtml(c.content)}</div>
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
              <div class="score-bar-top"><span>RRF</span><strong>${rrfLabel}</strong></div>
              <div class="score-bar-track"><div class="score-bar-fill fill-rrf" style="width:${rrfPct}%"></div></div>
            </div>
            <div class="score-bar-item">
              <div class="score-bar-top"><span>Rerank</span><strong>${rerankLabel}</strong></div>
              <div class="score-bar-track"><div class="score-bar-fill fill-rerank" style="width:${rerankPct}%"></div></div>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

window.toggleChunkText = function (contentId, btnEl) {
  const el = document.getElementById(contentId);
  if (!el) return;
  const isCollapsed = el.classList.toggle("collapsed");
  btnEl.textContent = isCollapsed ? "Show full text" : "Collapse text";
};

function renderMetricsSection(data) {
  const retMs = Number(data.retrieval_latency_ms || 0);
  const rerankMs = Number(data.rerank_latency_ms || 0).toFixed(1);
  const llmMs = Number(data.generation_latency_ms || 0);
  const totalMs = Number(data.total_latency_ms || 0);
  const pTok = Number(data.prompt_tokens || 0);
  const cTok = Number(data.completion_tokens || 0);
  const totTok = Number(data.total_tokens || pTok + cTok);

  document.getElementById("metrics-total-pill").textContent = `${totalMs.toFixed(0)} ms • ${totTok} tokens`;
  document.getElementById("metric-retrieval-ms").textContent = `${retMs.toFixed(1)} ms`;
  document.getElementById("metric-llm-ms").textContent = `${llmMs.toFixed(0)} ms`;
  document.getElementById("metric-total-ms").textContent = `${totalMs.toFixed(0)} ms`;
  document.getElementById("metric-tokens-total").textContent = String(totTok);
  document.getElementById("metric-rerank-sub").textContent = `Rerank: ${rerankMs} ms`;
  document.getElementById("metric-model-sub").textContent =
    (data.effective_settings && data.effective_settings.llm_model) || currentSettings.llmModel;
  document.getElementById("metric-status-sub").textContent = data.abstained
    ? "Abstained"
    : "Grounded";
  document.getElementById("metric-tokens-sub").textContent =
    `${pTok} prompt / ${cTok} out`;
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnswerWithClickableCitations(answerText) {
  const escaped = escapeHtml(answerText);
  return escaped.replace(/\[(\d+)\]/g, (match, num) => {
    return `<button type="button" class="cite-badge" onclick="highlightCitation(${num})" title="Scroll to Citation [${num}]">[${num}]</button>`;
  });
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
   TAB 2: 40-QUESTION EVALUATION HARNESS
   ===================================================================== */
function initEvalControls() {
  document.getElementById("ablation-config-select").addEventListener("change", () => {
    if (currentEvalReport) renderEvalDashboard(currentEvalReport);
  });
  document.getElementById("eval-category-filter").addEventListener("change", () => {
    if (currentEvalReport) renderQuestionsTable(currentEvalReport);
  });
  document.getElementById("eval-search-filter").addEventListener("input", () => {
    if (currentEvalReport) renderQuestionsTable(currentEvalReport);
  });
  document.getElementById("run-eval-btn").addEventListener("click", async () => {
    const btn = document.getElementById("run-eval-btn");
    btn.disabled = true;
    btn.textContent = "Evaluating 40 QA Pairs...";
    try {
      const res = await fetch("/api/v1/eval/run", { method: "POST" });
      currentEvalReport = await res.json();
      renderEvalDashboard(currentEvalReport);
    } finally {
      btn.disabled = false;
      btn.textContent = "Re-Run All 40 Questions";
    }
  });
}

async function loadEvalReport() {
  try {
    const res = await fetch("/api/v1/eval/report");
    currentEvalReport = await res.json();
    renderEvalDashboard(currentEvalReport);
  } catch (err) {
    console.error("Failed to load eval report:", err);
  }
}

function renderEvalDashboard(report) {
  const selectedKey = document.getElementById("ablation-config-select").value;
  const runs = report.ablation_runs || {};
  const activeRun = runs[selectedKey] || runs.hybrid_rrf_rerank;
  if (!activeRun) return;

  const m = activeRun.metrics;
  document.getElementById("eval-kpi-grid").innerHTML = `
    <div class="kpi-box">
      <div class="kpi-label">Recall@5</div>
      <div class="kpi-val">${(m.recall_at_5 * 100).toFixed(1)}%</div>
      <div class="kpi-sub">35 answerable questions</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">MRR@5</div>
      <div class="kpi-val">${m.mrr_at_5.toFixed(3)}</div>
      <div class="kpi-sub">Mean Reciprocal Rank</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">NDCG@5</div>
      <div class="kpi-val">${m.ndcg_at_5.toFixed(3)}</div>
      <div class="kpi-sub">Rank quality discount</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Answer Accuracy</div>
      <div class="kpi-val">${(m.answer_accuracy * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Key-fact coverage (40 QA)</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Faithfulness</div>
      <div class="kpi-val">${(m.faithfulness * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Grounded in cited chunks</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Citation Recall</div>
      <div class="kpi-val">${(m.citation_recall * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Ground-truth chunks cited</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-label">Abstention Acc</div>
      <div class="kpi-val">${(m.abstention_accuracy * 100).toFixed(1)}%</div>
      <div class="kpi-sub">5 unanswerable trap queries</div>
    </div>
  `;

  const labels = {
    dense_only: "Dense Only (pgvector)",
    sparse_only: "Sparse Only (tsvector)",
    hybrid_rrf: "Hybrid Search (RRF)",
    hybrid_rrf_rerank: "Hybrid (RRF) + Cross-Encoder Rerank",
  };

  document.getElementById("ablation-tbody").innerHTML = Object.keys(labels)
    .filter((k) => runs[k])
    .map((k) => {
      const rm = runs[k].metrics;
      const isSelected = k === selectedKey;
      return `
        <tr style="${isSelected ? "background: rgba(59, 130, 246, 0.12);" : ""}">
          <td><strong>${labels[k]}</strong></td>
          <td>${(rm.recall_at_1 * 100).toFixed(1)}%</td>
          <td>${(rm.recall_at_3 * 100).toFixed(1)}%</td>
          <td><strong>${(rm.recall_at_5 * 100).toFixed(1)}%</strong></td>
          <td>${rm.mrr_at_5.toFixed(3)}</td>
          <td>${rm.ndcg_at_5.toFixed(3)}</td>
          <td><strong>${(rm.answer_accuracy * 100).toFixed(1)}%</strong></td>
          <td>${(rm.faithfulness * 100).toFixed(1)}%</td>
          <td>${(rm.citation_recall * 100).toFixed(1)}%</td>
          <td>${(rm.abstention_accuracy * 100).toFixed(1)}%</td>
          <td>${rm.avg_latency_ms.toFixed(2)} ms</td>
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
          <div class="kpi-sub">Answer Acc • Recall@5: ${r5}</div>
        </div>
      `;
    })
    .join("");

  renderQuestionsTable(report);
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
          <td>${(q.recall_at_5 * 100).toFixed(0)}%</td>
          <td>${q.mrr_at_5.toFixed(2)}</td>
          <td><strong>${(q.answer_accuracy * 100).toFixed(0)}%</strong></td>
          <td>
            <button class="secondary-btn" onclick="tryQuestionInPlayground(${JSON.stringify(q.question).replace(/"/g, "&quot;")})">
              Test Live
            </button>
          </td>
        </tr>
      `;
    })
    .join("");
}

window.tryQuestionInPlayground = function (questionText) {
  document.querySelector('[data-tab="tab-playground"]').click();
  document.getElementById("question-input").value = questionText;
  runAskQuery();
};

/* =====================================================================
   TAB 3: CORPUS & DOCUMENT INGESTION
   ===================================================================== */
function initCorpusControls() {
  document.getElementById("refresh-docs-btn").addEventListener("click", loadDocuments);

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
        // Only populate id and source fallbacks if empty; leave title blank unless user typed one
        // so backend honors: Form Title -> Markdown # H1 -> Filename
        if (!idEl.value.trim()) idEl.value = cleanSlug;
        if (!sourceEl.value.trim()) sourceEl.value = `upload/${file.name}`;
        document.getElementById("ingest-content").value = contentText;
        const statusEl = document.getElementById("uploaded-file-status");
        if (statusEl) {
          statusEl.textContent = `Loaded "${file.name}" (${file.size} bytes) — click "Chunk, Embed & Index Document" below!`;
        }
      };
      reader.readAsText(file);
    });
  }

  document.getElementById("ingest-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("ingest-submit-btn");
    const feedback = document.getElementById("ingest-feedback");
    btn.disabled = true;
    btn.textContent = "Indexing...";
    feedback.classList.add("hidden");

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
      const res = await fetch("/api/v1/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: [doc] }),
      });
      const data = await res.json();
      feedback.textContent = `Ingested ${data.documents_ingested} document (${data.chunks_created} chunks created): ${data.document_ids.join(", ")}`;
      feedback.classList.remove("hidden");
      await loadDocuments();
      await fetchHealth();
    } catch (err) {
      feedback.textContent = `Error: ${err.message}`;
      feedback.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      btn.textContent = "Chunk, Embed & Index Document";
    }
  });
}

async function loadDocuments() {
  try {
    const res = await fetch("/api/v1/documents");
    indexedDocuments = await res.json();
    const container = document.getElementById("documents-list");
    if (container) {
      container.innerHTML = indexedDocuments
        .map(
          (d) => `
          <div class="doc-card">
            <div class="chunk-header">
              <strong>${escapeHtml(d.title)}</strong>
              <span class="chunk-id-tag">${escapeHtml(d.id)}</span>
            </div>
            <div class="muted-text" style="margin: 4px 0 0;">
              Source: <code>${escapeHtml(d.source)}</code> &bull; Chunks Indexed: <strong>${d.chunk_count}</strong>
            </div>
          </div>
        `
        )
        .join("");
    }
    renderDocScopeMenu();
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}
