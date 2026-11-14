/**
 * React Bits (@react-bits) Interactive Component Suite (`app/static/reactbits_fx.js`)
 *
 * Implements:
 *  1. GradientASCII Background — Flowing multi-stop color gradient mapped through an
 *     ASCII character density ramp with cursor wake scrambling & click shockwave ripples.
 *  2. ASCIIText Hero & Section Headings — Bitmap-rasterized interactive ASCII glyph
 *     matrix with 3D cursor tilt, chromatic RGB fringe, hover scramble, and click wave,
 *     paired with DecryptedText ASCII cipher reveals on section titles & chunk IDs.
 *  3. TiltedCard + GlareHover — 3D spring perspective cards for the Hero Showcase.
 *  4. SpotlightCard — Radial cursor-tracking spotlight border/surface glow on cards.
 *  5. StarBorder + ClickSpark — Orbiting glowing CTA border & radial ASCII spark bursts.
 *  6. CountUp — Smooth numeric counter animation for Run Metrics & KPI values.
 */

(function () {
  const ASCII_RAMP = " ·.:-=+*xX#$&@";
  const SCRAMBLE_GLYPHS = "01#$@%*+=-:;<>[]{}?/~^";
  const GRADIENT_STOPS = [
    [6, 182, 212],   // #06b6d4 Cyan
    [59, 130, 246],  // #3b82f6 Royal Blue
    [139, 92, 246],  // #8b5cf6 Violet
    [16, 185, 129],  // #10b981 Emerald
  ];

  let fxEnabled = true;
  let pointer = { x: -9999, y: -9999, vx: 0, vy: 0, lastX: 0, lastY: 0 };
  let ripples = [];

  document.addEventListener("DOMContentLoaded", () => {
    initPointerTracking();
    initFXToggle();
    initGradientASCIIBackground();
    initASCIITextHero();
    initASCIISectionHeaders();
    initTiltedCards();
    initSpotlightCards();
    initClickSpark();
  });

  function initPointerTracking() {
    window.addEventListener(
      "pointermove",
      (e) => {
        pointer.vx = e.clientX - pointer.lastX;
        pointer.vy = e.clientY - pointer.lastY;
        pointer.lastX = e.clientX;
        pointer.lastY = e.clientY;
        pointer.x = e.clientX;
        pointer.y = e.clientY;
      },
      { passive: true }
    );

    window.addEventListener(
      "pointerdown",
      (e) => {
        if (!fxEnabled) return;
        ripples.push({
          x: e.clientX,
          y: e.clientY,
          radius: 0,
          maxRadius: 340,
          strength: 1.0,
        });
        if (ripples.length > 5) ripples.shift();
      },
      { passive: true }
    );
  }

  function initFXToggle() {
    const btn = document.getElementById("ascii-fx-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      fxEnabled = !fxEnabled;
      document.body.classList.toggle("ascii-fx-subtle", !fxEnabled);
      btn.textContent = fxEnabled ? "ASCII FX: ON" : "ASCII FX: SUBTLE";
      btn.setAttribute("aria-pressed", String(fxEnabled));
    });
  }

  function interpolateGradient(t) {
    const clamped = Math.max(0, Math.min(1, t));
    const scaled = clamped * (GRADIENT_STOPS.length - 1);
    const idx = Math.floor(scaled);
    const frac = scaled - idx;
    const c1 = GRADIENT_STOPS[idx];
    const c2 = GRADIENT_STOPS[Math.min(idx + 1, GRADIENT_STOPS.length - 1)];
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * frac);
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * frac);
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * frac);
    return `rgb(${r},${g},${b})`;
  }

  /* =====================================================================
     1. GRADIENT ASCII BACKGROUND (@react-bits/GradientASCII)
     ===================================================================== */
  function initGradientASCIIBackground() {
    const canvas = document.getElementById("gradient-ascii-bg");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cellSize = 18;
    let cols = 0;
    let rows = 0;
    let time = 0;

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.ceil(window.innerWidth / cellSize);
      rows = Math.ceil(window.innerHeight / cellSize);
    }

    resize();
    window.addEventListener("resize", resize);

    function renderFrame() {
      if (!fxEnabled) {
        requestAnimationFrame(renderFrame);
        return;
      }

      time += 0.018;
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      ctx.font = "600 12px 'SFMono-Regular', Consolas, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      // Advance click shockwave ripples
      for (let i = ripples.length - 1; i >= 0; i--) {
        ripples[i].radius += 9.5;
        ripples[i].strength *= 0.96;
        if (ripples[i].radius > ripples[i].maxRadius || ripples[i].strength < 0.04) {
          ripples.splice(i, 1);
        }
      }

      for (let r = 0; r < rows; r++) {
        const y = r * cellSize + cellSize * 0.5;
        for (let c = 0; c < cols; c++) {
          const x = c * cellSize + cellSize * 0.5;

          // Multi-harmonic flowing gradient wave value in [0, 1]
          const nx = c * 0.055;
          const ny = r * 0.065;
          const w1 = Math.sin(nx + time * 1.1) * Math.cos(ny - time * 0.85);
          const w2 = Math.sin((nx + ny) * 0.7 - time * 0.6);
          let val = (w1 + w2 + 2.0) * 0.25;

          // Cursor proximity scramble & boost
          const dx = x - pointer.x;
          const dy = y - pointer.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          let cursorBoost = 0;
          if (dist < 170) {
            cursorBoost = (1.0 - dist / 170) * 0.58;
            val = Math.min(1.0, val + cursorBoost);
          }

          // Click shockwave ring contribution
          let rippleBoost = 0;
          for (let k = 0; k < ripples.length; k++) {
            const rp = ripples[k];
            const rDist = Math.abs(Math.hypot(x - rp.x, y - rp.y) - rp.radius);
            if (rDist < 36) {
              rippleBoost += (1.0 - rDist / 36) * rp.strength * 0.65;
            }
          }
          val = Math.min(1.0, val + rippleBoost);

          // Skip dimmest cells to keep background airy and readable
          if (val < 0.22) continue;

          const rampIdx = Math.min(
            ASCII_RAMP.length - 1,
            Math.max(1, Math.floor(val * (ASCII_RAMP.length - 1)))
          );
          let ch = ASCII_RAMP[rampIdx];
          if (cursorBoost > 0.28 || rippleBoost > 0.25) {
            const scrambleIdx = Math.floor((time * 40 + c * 7 + r * 13) % SCRAMBLE_GLYPHS.length);
            ch = SCRAMBLE_GLYPHS[scrambleIdx];
          }

          const gradPos = (c / Math.max(cols, 1) + Math.sin(time * 0.5 + r * 0.04) * 0.25 + 1) % 1;
          ctx.fillStyle = interpolateGradient(gradPos);
          ctx.globalAlpha = Math.min(0.62, 0.08 + val * 0.34 + cursorBoost * 0.35 + rippleBoost * 0.3);
          ctx.fillText(ch, x, y);
        }
      }
      ctx.globalAlpha = 1.0;
      requestAnimationFrame(renderFrame);
    }

    requestAnimationFrame(renderFrame);
  }

  /* =====================================================================
     2. ASCII TEXT HERO CENTERPIECE (@react-bits/ASCIIText)
     ===================================================================== */
  function initASCIITextHero() {
    const canvas = document.getElementById("ascii-hero-canvas");
    const wrapper = document.getElementById("ascii-hero-stage");
    if (!canvas || !wrapper) return;

    const ctx = canvas.getContext("2d");
    const offscreen = document.createElement("canvas");
    const offCtx = offscreen.getContext("2d");

    const HERO_STRINGS = ["RAG // EVAL", "PGVECTOR", "HYBRID RRF"];
    let activeStringIdx = 0;
    let points = [];
    let width = 0;
    let height = 0;
    let hoverLocal = { x: -999, y: -999, active: false };
    let clickWave = 0;
    let tick = 0;

    function buildGlyphMask(text) {
      width = wrapper.clientWidth || 720;
      height = 128;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const stepX = width < 560 ? 5 : 6;
      const stepY = width < 560 ? 7 : 8;
      offscreen.width = Math.ceil(width / stepX);
      offscreen.height = Math.ceil(height / stepY);

      offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
      offCtx.fillStyle = "#ffffff";
      const fontSize = Math.min(Math.floor(offscreen.width * 0.145), offscreen.height - 2);
      offCtx.font = `900 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif`;
      offCtx.textAlign = "center";
      offCtx.textBaseline = "middle";
      offCtx.fillText(text, offscreen.width / 2, offscreen.height / 2);

      const img = offCtx.getImageData(0, 0, offscreen.width, offscreen.height).data;
      points = [];
      for (let y = 0; y < offscreen.height; y++) {
        for (let x = 0; x < offscreen.width; x++) {
          const alpha = img[(y * offscreen.width + x) * 4 + 3];
          if (alpha > 90) {
            points.push({
              x: x * stepX + stepX * 0.5,
              y: y * stepY + stepY * 0.5,
              nx: x / offscreen.width,
              ny: y / offscreen.height,
              density: alpha / 255,
              phase: Math.random() * Math.PI * 2,
            });
          }
        }
      }
    }

    buildGlyphMask(HERO_STRINGS[activeStringIdx]);
    window.addEventListener("resize", () => buildGlyphMask(HERO_STRINGS[activeStringIdx]));

    wrapper.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      hoverLocal.x = e.clientX - rect.left;
      hoverLocal.y = e.clientY - rect.top;
      hoverLocal.active = true;

      // 3D tilt toward cursor
      const rx = ((hoverLocal.y / rect.height) - 0.5) * -10;
      const ry = ((hoverLocal.x / rect.width) - 0.5) * 12;
      canvas.style.transform = `perspective(700px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg)`;
    });

    wrapper.addEventListener("pointerleave", () => {
      hoverLocal.active = false;
      canvas.style.transform = "perspective(700px) rotateX(0deg) rotateY(0deg)";
    });

    wrapper.addEventListener("click", () => {
      clickWave = 1.0;
      activeStringIdx = (activeStringIdx + 1) % HERO_STRINGS.length;
      buildGlyphMask(HERO_STRINGS[activeStringIdx]);
    });

    function animateHero() {
      tick += 0.045;
      if (clickWave > 0.01) clickWave *= 0.92;

      ctx.clearRect(0, 0, width, height);
      ctx.font = "700 8px 'SFMono-Regular', Consolas, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const chars = "@#W$9876543210?!abc;:+=-,._";
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        let dx = p.x - hoverLocal.x;
        let dy = p.y - hoverLocal.y;
        let dist = hoverLocal.active ? Math.sqrt(dx * dx + dy * dy) : 9999;

        let ox = 0;
        let oy = 0;
        let charIdx = Math.floor(((Math.sin(tick + p.nx * 6 + p.phase) + 1) * 0.5) * 10);

        if (dist < 75) {
          const force = (1 - dist / 75) * 8;
          ox = (dx / (dist || 1)) * force;
          oy = (dy / (dist || 1)) * force;
          charIdx = Math.floor((tick * 25 + i) % chars.length);
        }

        const waveOffset = Math.sin(p.nx * 10 - tick * 1.6) * 2.2 + clickWave * Math.sin(i) * 5;
        const ch = chars[charIdx % chars.length];

        // Chromatic fringe when hovered or clicked
        if (dist < 65 || clickWave > 0.15) {
          ctx.fillStyle = "rgba(6, 182, 212, 0.75)";
          ctx.fillText(ch, p.x + ox - 1.4, p.y + oy + waveOffset);
          ctx.fillStyle = "rgba(236, 72, 153, 0.7)";
          ctx.fillText(ch, p.x + ox + 1.4, p.y + oy + waveOffset);
        }

        const tGrad = (p.nx + Math.sin(tick * 0.4) * 0.2 + 1) % 1;
        ctx.fillStyle = interpolateGradient(tGrad);
        ctx.fillText(ch, p.x + ox, p.y + oy + waveOffset);
      }

      requestAnimationFrame(animateHero);
    }

    requestAnimationFrame(animateHero);
  }

  /* =====================================================================
     3. ASCII SECTION HEADERS & DECRYPTED TEXT (@react-bits/DecryptedText)
     ===================================================================== */
  function initASCIISectionHeaders() {
    const targets = document.querySelectorAll("[data-ascii-heading]");
    targets.forEach((el) => {
      const original = el.dataset.asciiHeading || el.textContent.trim();
      el.dataset.asciiHeading = original;
      scrambleRevealElement(el, original, 520);

      const summary = el.closest("summary") || el;
      summary.addEventListener("pointerenter", () => {
        if (!fxEnabled) return;
        scrambleRevealElement(el, original, 360);
      });
    });
  }

  function scrambleRevealElement(el, targetText, durationMs = 420) {
    if (el._scrambleTimer) clearInterval(el._scrambleTimer);
    const totalSteps = Math.max(8, Math.floor(durationMs / 28));
    let step = 0;

    el._scrambleTimer = setInterval(() => {
      step++;
      const progress = step / totalSteps;
      const revealedCount = Math.floor(targetText.length * progress);
      let out = "";
      for (let i = 0; i < targetText.length; i++) {
        const ch = targetText[i];
        if (ch === " " || i < revealedCount) {
          out += ch;
        } else {
          out += SCRAMBLE_GLYPHS[(i * 7 + step * 3) % SCRAMBLE_GLYPHS.length];
        }
      }
      el.textContent = out;
      if (step >= totalSteps) {
        clearInterval(el._scrambleTimer);
        el.textContent = targetText;
      }
    }, 28);
  }

  window.triggerASCIIDecrypt = scrambleRevealElement;

  /* =====================================================================
     4. TILTED CARDS + GLARE HOVER (@react-bits/TiltedCard)
     ===================================================================== */
  function initTiltedCards() {
    const cards = document.querySelectorAll("[data-rb-tilt]");
    cards.forEach((card) => {
      card.addEventListener("pointermove", (e) => {
        if (!fxEnabled) return;
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const rx = ((y / rect.height) - 0.5) * -9;
        const ry = ((x / rect.width) - 0.5) * 9;
        card.style.transform = `perspective(800px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateY(-2px)`;
        card.style.setProperty("--spot-x", `${x.toFixed(0)}px`);
        card.style.setProperty("--spot-y", `${y.toFixed(0)}px`);
      });

      card.addEventListener("pointerleave", () => {
        card.style.transform = "perspective(800px) rotateX(0deg) rotateY(0deg) translateY(0)";
      });
    });
  }

  /* =====================================================================
     5. SPOTLIGHT CARDS (@react-bits/SpotlightCard)
     ===================================================================== */
  function initSpotlightCards() {
    document.addEventListener(
      "pointermove",
      (e) => {
        if (!fxEnabled) return;
        const cards = document.querySelectorAll(
          ".collapsible-section, .citation-card, .chunk-card, .metric-card, .sample-chip, .control-card"
        );
        cards.forEach((card) => {
          const rect = card.getBoundingClientRect();
          if (
            e.clientX >= rect.left - 120 &&
            e.clientX <= rect.right + 120 &&
            e.clientY >= rect.top - 120 &&
            e.clientY <= rect.bottom + 120
          ) {
            card.style.setProperty("--spot-x", `${(e.clientX - rect.left).toFixed(0)}px`);
            card.style.setProperty("--spot-y", `${(e.clientY - rect.top).toFixed(0)}px`);
          }
        });
      },
      { passive: true }
    );
  }

  /* =====================================================================
     6. CLICK SPARK MICRO-INTERACTION (@react-bits/ClickSpark)
     ===================================================================== */
  function initClickSpark() {
    document.addEventListener("click", (e) => {
      if (!fxEnabled) return;
      const target = e.target.closest("button, .sample-chip, .cite-badge, summary");
      if (!target) return;
      spawnASCIISparks(e.clientX, e.clientY);
    });
  }

  function spawnASCIISparks(x, y) {
    const glyphs = ["*", "+", "#", "·", ":", "x", "@"];
    const colors = ["#06b6d4", "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"];
    const count = 9;

    for (let i = 0; i < count; i++) {
      const span = document.createElement("span");
      span.className = "ascii-click-spark";
      span.textContent = glyphs[i % glyphs.length];
      span.style.left = `${x}px`;
      span.style.top = `${y}px`;
      span.style.color = colors[i % colors.length];
      const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.3;
      const dist = 26 + Math.random() * 28;
      span.style.setProperty("--tx", `${(Math.cos(angle) * dist).toFixed(1)}px`);
      span.style.setProperty("--ty", `${(Math.sin(angle) * dist).toFixed(1)}px`);
      document.body.appendChild(span);
      setTimeout(() => span.remove(), 520);
    }
  }

  /* =====================================================================
     7. COUNT UP NUMERIC ANIMATION (@react-bits/CountUp)
     ===================================================================== */
  window.animateCountUp = function (elOrId, endValue, suffix = "", decimals = 2, duration = 420) {
    const el = typeof elOrId === "string" ? document.getElementById(elOrId) : elOrId;
    if (!el) return;
    const target = Number(endValue);
    if (Number.isNaN(target)) {
      el.textContent = `${endValue}${suffix}`;
      return;
    }
    const startTime = performance.now();
    function step(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = target * eased;
      el.textContent = `${current.toFixed(decimals)}${suffix}`;
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = `${target.toFixed(decimals)}${suffix}`;
      }
    }
    requestAnimationFrame(step);
  };
})();
