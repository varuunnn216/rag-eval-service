let currentEvalReport = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPlaygroundControls();
  initEvalControls();
  initCorpusControls();

  fetchHealth();
  runAskQuery();
  loadEvalReport();
  loadDocuments();
});

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
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
      docsPill.textContent = `${data.documents_count} Docs / ${data.chunks_count} Chunks`;
    }
    if (modelsPill) {
      modelsPill.textContent = `${data.generation_model} • ${data.reranker_model.split("/").pop()}`;
    }
  } catch (err) {
    console.error("Failed to fetch health:", err);
  }
}

function initPlaygroundControls() {
  const topkSlider = document.getElementById("topk-slider");
  const topkVal = document.getElementById("topk-val");
  const poolSlider = document.getElementById("pool-slider");
  const poolVal = document.getElementById("pool-val");

  topkSlider.addEventListener("input", () => {
    topkVal.textContent = topkSlider.value;
  });
  poolSlider.addEventListener("input", () => {
    poolVal.textContent = poolSlider.value;
  });

  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("question-input").value = btn.dataset.preset;
      runAskQuery();
    });
  });

  document.getElementById("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runAskQuery();
  });
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatAnswerWithClickableCitations(answerText) {
  const escaped = escapeHtml(answerText);
  return escaped.replace(/\[(\d+)\]/g, (match, num) => {
    return `<button type="button" class="cite-badge" onclick="highlightCitation(${num})">[${num}]</button>`;
  });
}

window.highlightCitation = function (num) {
  document.querySelectorAll(".citation-card, .chunk-card").forEach((el) => {
    el.classList.remove("highlight");
  });
  const citCard = document.getElementById(`citation-card-${num}`);
  const chunkCard = document.getElementById(`chunk-card-${num}`);
  if (citCard) {
    citCard.classList.add("highlight");
    citCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  if (chunkCard) {
    chunkCard.classList.add("highlight");
  }
};

async function runAskQuery() {
  const question = document.getElementById("question-input").value.trim();
  if (!question) return;

  const mode = document.getElementById("mode-select").value;
  const useReranker = document.getElementById("reranker-select").value === "true";
  const topK = parseInt(document.getElementById("topk-slider").value, 10);
  const candidatePool = parseInt(document.getElementById("pool-slider").value, 10);

  const submitBtn = document.getElementById("ask-submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Running...";

  try {
    const res = await fetch("/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        mode,
        top_k: topK,
        candidate_pool: candidatePool,
        use_reranker: useReranker,
      }),
    });
    const data = await res.json();
    renderAskResponse(data);
  } catch (err) {
    document.getElementById("answer-output").textContent = `Error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run RAG Pipeline";
  }
}

function renderAskResponse(data) {
  const banner = document.getElementById("answer-status-banner");
  banner.classList.remove("hidden", "grounded", "abstained", "error");
  const citSummary = data.citation_summary || `Verified ${data.verified_citations_count || 0}/${data.total_citations_count || 0} citations`;

  if (data.error) {
    banner.classList.add("error");
    banner.textContent = `LLM Generation Error: ${data.error}`;
  } else if (data.abstained) {
    banner.classList.add("abstained");
    banner.textContent =
      `Abstention Guard Triggered • ${citSummary}`;
  } else {
    banner.classList.add("grounded");
    banner.textContent = `Grounded Response • ${citSummary}`;
  }

  document.getElementById("latency-badges").innerHTML = `
    <span class="latency-pill">Retrieval: <strong>${data.retrieval_latency_ms} ms</strong></span>
    <span class="latency-pill">LLM: <strong>${data.generation_latency_ms} ms</strong></span>
    <span class="latency-pill">Total: <strong>${data.total_latency_ms} ms</strong></span>
    <span class="latency-pill">Tokens: <strong>${data.total_tokens || 0}</strong></span>
  `;

  document.getElementById("answer-output").innerHTML =
    formatAnswerWithClickableCitations(data.answer);

  // Render Verified Citations
  const citList = document.getElementById("citations-list");
  document.getElementById("citation-count").textContent = citSummary;
  if (!data.citations || !data.citations.length) {
    citList.innerHTML = `<div class="muted-text">No citations emitted (${escapeHtml(citSummary)}).</div>`;
  } else {
    citList.innerHTML = data.citations
      .map(
        (c) => {
          const statusClass = c.verified ? "used" : "dropped";
          const statusLabel = c.verified
            ? `VERIFIED (sim=${Number(c.similarity_score || 0).toFixed(2)})`
            : `UNVERIFIED (sim=${Number(c.similarity_score || 0).toFixed(2)})`;
          return `
        <div class="citation-card" id="citation-card-${c.citation_number}">
          <div class="citation-header">
            <div>
              <span class="cite-badge">[${c.citation_number}]</span>
              <strong>${escapeHtml(c.document_title)}</strong>
              <span class="chunk-status-badge ${statusClass}">${statusLabel}</span>
            </div>
            <span class="chunk-id-tag">${escapeHtml(c.chunk_id)}</span>
          </div>
          ${c.claim ? `<div class="selection-reason">Claim: ${escapeHtml(c.claim)}</div>` : ""}
          <div class="quote-text">"${escapeHtml(c.quote)}"</div>
        </div>
      `;
        }
      )
      .join("");
  }

  // Render Retrieved Chunks (marked as USED or DROPPED with reason)
  const chunksList = document.getElementById("chunks-list");
  document.getElementById("chunk-count").textContent = (data.retrieved_chunks || []).length;
  chunksList.innerHTML = (data.retrieved_chunks || [])
    .map((c, idx) => {
      const rankNum = c.context_index != null ? c.context_index : idx + 1;
      const isUsed = Boolean(c.selected) || c.selection_status === "used";
      const statusBadge = isUsed
        ? `<span class="chunk-status-badge used">USED [${c.context_index || rankNum}]</span>`
        : `<span class="chunk-status-badge dropped">DROPPED</span>`;
      const denseStr =
        c.dense_rank != null ? `#${c.dense_rank} (${Number(c.dense_score || 0).toFixed(3)})` : "—";
      const sparseStr =
        c.sparse_rank != null ? `#${c.sparse_rank} (${Number(c.sparse_score || 0).toFixed(3)})` : "—";
      const rrfStr = c.rrf_score ? Number(c.rrf_score).toFixed(4) : "—";
      const rerankStr =
        c.rerank_score != null ? Number(c.rerank_score).toFixed(4) : "Off";

      return `
        <div class="chunk-card" id="chunk-card-${rankNum}" style="${isUsed ? "" : "opacity: 0.72;"}">
          <div class="chunk-header">
            <div>
              ${statusBadge}
              <strong>${escapeHtml(c.document_title)}</strong>
            </div>
            <span class="chunk-id-tag">${escapeHtml(c.chunk_id)}</span>
          </div>
          ${c.selection_reason ? `<div class="selection-reason">${escapeHtml(c.selection_reason)}</div>` : ""}
          ${c.heading ? `<div class="chunk-heading">${escapeHtml(c.heading)}</div>` : ""}
          <div class="chunk-content">${escapeHtml(c.content)}</div>
          <div class="score-row">
            <span class="score-chip">Dense: <strong>${denseStr}</strong></span>
            <span class="score-chip">Sparse: <strong>${sparseStr}</strong></span>
            <span class="score-chip">RRF: <strong>${rrfStr}</strong></span>
            <span class="score-chip">Reranker: <strong>${rerankStr}</strong></span>
            <span class="score-chip">Tokens: <strong>${c.token_count}</strong></span>
          </div>
        </div>
      `;
    })
    .join("");
}

/* TAB 2: EVALUATION HARNESS */
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
        <tr style="${isSelected ? "background: rgba(59, 130, 246, 0.10);" : ""}">
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

/* TAB 3: CORPUS & DOCUMENT INGESTION */
let lastUploadedFilename = "";

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
        const baseName = file.name.replace(/\.[^.]+$/, "");
        const cleanSlug =
          "doc_" +
          baseName
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "_")
            .replace(/^_+|_+$/g, "");
        const idEl = document.getElementById("ingest-id");
        const titleEl = document.getElementById("ingest-title");
        const sourceEl = document.getElementById("ingest-source");
        // Honor user-entered values; only populate fallback when empty
        if (!idEl.value.trim()) idEl.value = cleanSlug;
        if (!titleEl.value.trim()) titleEl.value = baseName.replace(/[_-]+/g, " ");
        if (!sourceEl.value.trim()) sourceEl.value = `upload/${file.name}`;
        document.getElementById("ingest-content").value = String(reader.result || "");
        const statusEl = document.getElementById("uploaded-file-status");
        if (statusEl) {
          statusEl.textContent = `Loaded "${file.name}" (${file.size} bytes) — click "Chunk, Embed & Index Document" below to index it!`;
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
    const docs = await res.json();
    const container = document.getElementById("documents-list");
    container.innerHTML = docs
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
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}
