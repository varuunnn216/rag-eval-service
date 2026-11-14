/**
 * Official React Bits (@react-bits) Vanilla JS / Canvas Runtime
 * Ported directly from https://reactbits.dev/r/{Component}-JS-CSS.json via React Bits MCP:
 *  1. Waves (Interactive Perlin-noise flowing silk wave lines that bend around cursor velocity & ripple on click)
 *  2. PixelCard (Interactive radial twinkling pixel canvas for the 3 main workspace cards on the landing page)
 *  3. GlareHover (Diagonal specular light sheen sweep across cards and interactive panels)
 *  4. Dock (Spring-magnifying floating navigation pill bar with active indicator dots)
 *  5. GradientText (Flowing multi-stop gradient text with glow layer)
 *  6. ShinyText (Metallic light-sweep shimmer across badges)
 *  7. BlurText (Staggered blur-to-sharp word entrance reveal on headers)
 *  8. StarBorder (Orbiting glowing star trail + stardust motes on primary CTAs)
 *  9. ClickSpark (Bursting star & streak sparks on interactive clicks)
 * 10. CountUp (Cubic-out animated numeric counters for metrics & KPIs)
 */

(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const clamp = (val, min, max) => Math.min(max, Math.max(min, val));
  const reduceMotion = () =>
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

  /* =========================================================================
   * 1. REACT BITS: WAVES (From Waves-JS-CSS.json)
   *    Perlin-noise flowing vector wave lines with cursor displacement & click ripples
   * ========================================================================= */
  class Grad {
    constructor(x, y, z) {
      this.x = x;
      this.y = y;
      this.z = z;
    }
    dot2(x, y) {
      return this.x * x + this.y * y;
    }
  }

  class PerlinNoise {
    constructor(seed = 0) {
      this.grad3 = [
        new Grad(1, 1, 0), new Grad(-1, 1, 0), new Grad(1, -1, 0), new Grad(-1, -1, 0),
        new Grad(1, 0, 1), new Grad(-1, 0, 1), new Grad(1, 0, -1), new Grad(-1, 0, -1),
        new Grad(0, 1, 1), new Grad(0, -1, 1), new Grad(0, 1, -1), new Grad(0, -1, -1),
      ];
      this.p = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240,
        21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88,
        237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
        111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216,
        80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,
        3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58,
        17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
        129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
        238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
        184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128,
        195, 78, 66, 215, 61, 156, 180,
      ];
      this.perm = new Array(512);
      this.gradP = new Array(512);
      this.seed(seed);
    }
    seed(seed) {
      if (seed > 0 && seed < 1) seed *= 65536;
      seed = Math.floor(seed);
      if (seed < 256) seed |= seed << 8;
      for (let i = 0; i < 256; i++) {
        let v = i & 1 ? this.p[i] ^ (seed & 255) : this.p[i] ^ ((seed >> 8) & 255);
        this.perm[i] = this.perm[i + 256] = v;
        this.gradP[i] = this.gradP[i + 256] = this.grad3[v % 12];
      }
    }
    fade(t) {
      return t * t * t * (t * (t * 6 - 15) + 10);
    }
    lerp(a, b, t) {
      return (1 - t) * a + t * b;
    }
    perlin2(x, y) {
      let X = Math.floor(x),
        Y = Math.floor(y);
      x -= X;
      y -= Y;
      X &= 255;
      Y &= 255;
      const n00 = this.gradP[X + this.perm[Y]].dot2(x, y);
      const n01 = this.gradP[X + this.perm[Y + 1]].dot2(x, y - 1);
      const n10 = this.gradP[X + 1 + this.perm[Y]].dot2(x - 1, y);
      const n11 = this.gradP[X + 1 + this.perm[Y + 1]].dot2(x - 1, y - 1);
      const u = this.fade(x);
      return this.lerp(this.lerp(n00, n10, u), this.lerp(n01, n11, u), this.fade(y));
    }
  }

  function initWavesBackdrop() {
    const container = document.getElementById("reactbits-waves");
    if (!container) return;
    const canvas = container.querySelector("canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const noise = new PerlinNoise(42);
    const config = {
      xGap: 18,
      yGap: 32,
      waveSpeedX: 0.012,
      waveSpeedY: 0.005,
      waveAmpX: 32,
      waveAmpY: 16,
      friction: 0.91,
      tension: 0.006,
      maxCursorMove: 110,
      interactionRadius: 160,
    };

    let width = 0;
    let height = 0;
    let lines = [];
    let ripples = [];
    const mouse = {
      x: -500,
      y: -500,
      lx: 0,
      ly: 0,
      sx: -500,
      sy: -500,
      v: 0,
      vs: 0,
      a: 0,
      set: false,
    };

    function setSize() {
      const rect = container.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      setLines();
    }

    function setLines() {
      lines = [];
      const oWidth = width + 180;
      const oHeight = height + 60;
      const totalLines = Math.ceil(oWidth / config.xGap);
      const totalPoints = Math.ceil(oHeight / config.yGap);
      const xStart = (width - config.xGap * totalLines) / 2;
      const yStart = (height - config.yGap * totalPoints) / 2;

      for (let i = 0; i <= totalLines; i++) {
        const pts = [];
        for (let j = 0; j <= totalPoints; j++) {
          pts.push({
            x: xStart + config.xGap * i,
            y: yStart + config.yGap * j,
            wave: { x: 0, y: 0 },
            cursor: { x: 0, y: 0, vx: 0, vy: 0 },
          });
        }
        lines.push(pts);
      }
    }

    function movePoints(time) {
      for (let i = 0; i < lines.length; i++) {
        const pts = lines[i];
        for (let j = 0; j < pts.length; j++) {
          const p = pts[j];
          const move =
            noise.perlin2(
              (p.x + time * config.waveSpeedX) * 0.0022,
              (p.y + time * config.waveSpeedY) * 0.0016
            ) * 12;
          p.wave.x = Math.cos(move) * config.waveAmpX;
          p.wave.y = Math.sin(move) * config.waveAmpY;

          const dx = p.x - mouse.sx;
          const dy = p.y - mouse.sy;
          const dist = Math.hypot(dx, dy);
          const l = Math.max(config.interactionRadius, mouse.vs);
          if (dist < l) {
            const s = 1 - dist / l;
            const f = Math.cos(dist * 0.001) * s;
            p.cursor.vx += Math.cos(mouse.a) * f * l * mouse.vs * 0.0006;
            p.cursor.vy += Math.sin(mouse.a) * f * l * mouse.vs * 0.0006;
          }

          for (let r = 0; r < ripples.length; r++) {
            const rp = ripples[r];
            const rdx = p.x - rp.x;
            const rdy = p.y - rp.y;
            const rDist = Math.hypot(rdx, rdy);
            const band = Math.abs(rDist - rp.radius);
            if (band < 48 && rDist > 0.001) {
              const falloff = (1 - band / 48) * rp.alpha;
              const push = Math.sin((band / 48) * Math.PI) * falloff * 1.65;
              p.cursor.vx += (rdx / rDist) * push;
              p.cursor.vy += (rdy / rDist) * push;
            }
          }

          p.cursor.vx += (0 - p.cursor.x) * config.tension;
          p.cursor.vy += (0 - p.cursor.y) * config.tension;
          p.cursor.vx *= config.friction;
          p.cursor.vy *= config.friction;
          p.cursor.x += p.cursor.vx * 2;
          p.cursor.y += p.cursor.vy * 2;
          p.cursor.x = clamp(p.cursor.x, -config.maxCursorMove, config.maxCursorMove);
          p.cursor.y = clamp(p.cursor.y, -config.maxCursorMove, config.maxCursorMove);
        }
      }

      for (let r = ripples.length - 1; r >= 0; r--) {
        ripples[r].radius += 5.5;
        ripples[r].alpha *= 0.962;
        if (ripples[r].alpha < 0.025) ripples.splice(r, 1);
      }
    }

    function moved(point, withCursor = true) {
      return {
        x: point.x + point.wave.x + (withCursor ? point.cursor.x : 0),
        y: point.y + point.wave.y + (withCursor ? point.cursor.y : 0),
      };
    }

    function drawLines() {
      ctx.clearRect(0, 0, width, height);

      const grad = ctx.createLinearGradient(0, 0, width, height);
      grad.addColorStop(0, "rgba(56, 189, 248, 0.22)");
      grad.addColorStop(0.5, "rgba(129, 140, 248, 0.20)");
      grad.addColorStop(1, "rgba(52, 211, 153, 0.16)");

      ctx.beginPath();
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.1;

      for (let i = 0; i < lines.length; i++) {
        const points = lines[i];
        if (!points.length) continue;
        let p1 = moved(points[0], false);
        ctx.moveTo(p1.x, p1.y);
        for (let idx = 0; idx < points.length; idx++) {
          const isLast = idx === points.length - 1;
          p1 = moved(points[idx], !isLast);
          const p2 = moved(points[idx + 1] || points[points.length - 1], !isLast);
          ctx.lineTo(p1.x, p1.y);
          if (isLast) ctx.moveTo(p2.x, p2.y);
        }
      }
      ctx.stroke();

      /* Subtle cursor halo on the wave grid */
      if (mouse.set && mouse.sx > 0 && mouse.sy > 0) {
        const glow = ctx.createRadialGradient(mouse.sx, mouse.sy, 0, mouse.sx, mouse.sy, 220);
        glow.addColorStop(0, "rgba(56, 189, 248, 0.12)");
        glow.addColorStop(0.5, "rgba(139, 92, 246, 0.06)");
        glow.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, width, height);
      }
    }

    function tick(t) {
      mouse.sx += (mouse.x - mouse.sx) * 0.12;
      mouse.sy += (mouse.y - mouse.sy) * 0.12;
      const dx = mouse.x - mouse.lx;
      const dy = mouse.y - mouse.ly;
      const d = Math.hypot(dx, dy);
      mouse.v = d;
      mouse.vs += (d - mouse.vs) * 0.1;
      mouse.vs = Math.min(100, mouse.vs);
      mouse.lx = mouse.x;
      mouse.ly = mouse.y;
      mouse.a = Math.atan2(dy, dx);

      movePoints(t);
      drawLines();
      if (!reduceMotion()) {
        requestAnimationFrame(tick);
      }
    }

    window.addEventListener(
      "pointermove",
      (e) => {
        const rect = container.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
        if (!mouse.set) {
          mouse.sx = mouse.x;
          mouse.sy = mouse.y;
          mouse.lx = mouse.x;
          mouse.ly = mouse.y;
          mouse.set = true;
        }
      },
      { passive: true }
    );

    window.addEventListener(
      "pointerdown",
      (e) => {
        const rect = container.getBoundingClientRect();
        ripples.push({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          radius: 4,
          alpha: 1,
        });
        if (ripples.length > 6) ripples.shift();
      },
      { passive: true }
    );

    window.addEventListener("resize", setSize);
    setSize();
    requestAnimationFrame(tick);
  }

  /* =========================================================================
   * 2. REACT BITS: PIXEL CARD (From PixelCard-JS-CSS.json)
   *    Interactive twinkling pixel canvas inside the 3 main workspace cards
   * ========================================================================= */
  const PIXEL_VARIANTS = {
    blue: {
      gap: 6,
      speed: 45,
      colors: "#e0f2fe,#7dd3fc,#38bdf8,#0284c7",
    },
    purple: {
      gap: 6,
      speed: 45,
      colors: "#f3e8ff,#d8b4fe,#c084fc,#9333ea",
    },
    emerald: {
      gap: 6,
      speed: 45,
      colors: "#d1fae5,#6ee7b7,#34d399,#059669",
    },
  };

  class PixelCell {
    constructor(ctx, x, y, color, speed, delay) {
      this.ctx = ctx;
      this.x = x;
      this.y = y;
      this.color = color;
      this.speed = (Math.random() * 0.8 + 0.15) * speed;
      this.size = 0;
      this.sizeStep = Math.random() * 0.45 + 0.18;
      this.minSize = 0.6;
      this.maxSizeInteger = 2.4;
      this.maxSize = Math.random() * (this.maxSizeInteger - this.minSize) + this.minSize;
      this.delay = delay;
      this.counter = 0;
      this.counterStep = Math.random() * 4 + 2.5;
      this.isIdle = true;
      this.isReverse = false;
      this.isShimmer = false;
      this.boost = 0;
    }

    draw() {
      const centerOffset = this.maxSizeInteger * 0.5 - this.size * 0.5;
      this.ctx.fillStyle = this.color;
      this.ctx.globalAlpha = Math.min(1, 0.35 + (this.size / this.maxSizeInteger) * 0.55 + this.boost * 0.25);
      this.ctx.fillRect(this.x + centerOffset, this.y + centerOffset, this.size, this.size);
      this.ctx.globalAlpha = 1;
      this.boost *= 0.92;
    }

    appear() {
      this.isIdle = false;
      if (this.counter <= this.delay) {
        this.counter += this.counterStep;
        return;
      }
      if (this.size >= this.maxSize) {
        this.isShimmer = true;
      }
      if (this.isShimmer) {
        this.shimmer();
      } else {
        this.size += this.sizeStep;
      }
      this.draw();
    }

    disappear() {
      this.isShimmer = false;
      this.counter = 0;
      if (this.size <= 0) {
        this.isIdle = true;
        return;
      } else {
        this.size -= 0.12;
      }
      this.draw();
    }

    shimmer() {
      if (this.size >= this.maxSize) {
        this.isReverse = true;
      } else if (this.size <= this.minSize) {
        this.isReverse = false;
      }
      if (this.isReverse) {
        this.size -= this.speed;
      } else {
        this.size += this.speed;
      }
    }

    pulse(strength = 1) {
      this.isIdle = false;
      this.isShimmer = true;
      this.size = Math.min(this.maxSizeInteger, Math.max(this.size, this.maxSize * 0.95));
      this.boost = Math.max(this.boost, strength);
    }
  }

  function getEffectiveSpeed(value, reduced) {
    const throttle = 0.001;
    if (value <= 0 || reduced) return 0;
    if (value >= 100) return 100 * throttle;
    return value * throttle;
  }

  function initPixelCards() {
    const cards = document.querySelectorAll("[data-rb-pixel-card]");
    const reduced = reduceMotion();

    cards.forEach((card) => {
      if (card.dataset.rbPixelBound === "1") return;
      card.dataset.rbPixelBound = "1";

      const canvas = card.querySelector(".pixel-canvas");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const variantName = card.dataset.pixelVariant || "blue";
      const cfg = PIXEL_VARIANTS[variantName] || PIXEL_VARIANTS.blue;
      const colorsArr = cfg.colors.split(",");
      const effSpeed = getEffectiveSpeed(cfg.speed, reduced);

      let pixels = [];
      let rafId = null;
      let activeMethod = "disappear";
      let width = 0;
      let height = 0;

      function buildPixels(originX, originY) {
        const rect = card.getBoundingClientRect();
        width = Math.max(1, Math.floor(rect.width));
        height = Math.max(1, Math.floor(rect.height));
        canvas.width = width;
        canvas.height = height;

        const cx = originX ?? width / 2;
        const cy = originY ?? height / 2;

        pixels = [];
        for (let x = 0; x < width; x += cfg.gap) {
          for (let y = 0; y < height; y += cfg.gap) {
            const color = colorsArr[Math.floor(Math.random() * colorsArr.length)];
            const dist = Math.hypot(x - cx, y - cy);
            const delay = reduced ? 0 : dist * 0.65;
            pixels.push(new PixelCell(ctx, x, y, color, effSpeed, delay));
          }
        }
      }

      function step() {
        ctx.clearRect(0, 0, width, height);
        let allIdle = true;
        for (let i = 0; i < pixels.length; i++) {
          const p = pixels[i];
          p[activeMethod]();
          if (!p.isIdle) allIdle = false;
        }
        if (!allIdle) {
          rafId = requestAnimationFrame(step);
        } else {
          rafId = null;
        }
      }

      function startAnimation(method, originX, originY) {
        activeMethod = method;
        if (method === "appear" && typeof originX === "number") {
          for (let i = 0; i < pixels.length; i++) {
            const p = pixels[i];
            p.delay = reduced ? 0 : Math.hypot(p.x - originX, p.y - originY) * 0.55;
            p.counter = 0;
          }
        }
        if (!rafId) rafId = requestAnimationFrame(step);
      }

      card.addEventListener("pointerenter", (e) => {
        const rect = card.getBoundingClientRect();
        if (!pixels.length || Math.abs(rect.width - width) > 4) {
          buildPixels(e.clientX - rect.left, e.clientY - rect.top);
        }
        startAnimation("appear", e.clientX - rect.left, e.clientY - rect.top);
      });

      card.addEventListener("pointerleave", () => {
        startAnimation("disappear");
      });

      card.addEventListener("pointerdown", (e) => {
        const rect = card.getBoundingClientRect();
        const ox = e.clientX - rect.left;
        const oy = e.clientY - rect.top;
        const radius = Math.max(width, height) * 0.45;
        for (let i = 0; i < pixels.length; i++) {
          const p = pixels[i];
          const d = Math.hypot(p.x - ox, p.y - oy);
          if (d <= radius) {
            p.pulse(1 - d / radius);
          }
        }
        startAnimation("appear", ox, oy);
      });

      const ro = new ResizeObserver(() => buildPixels());
      ro.observe(card);
      buildPixels();
    });
  }

  /* =========================================================================
   * 3. REACT BITS: GLARE HOVER (From GlareHover-JS-CSS.json)
   *    Glossy diagonal specular light sweep across cards on hover
   * ========================================================================= */
  function initGlareHover(root = document) {
    const targets = root.querySelectorAll(
      "[data-rb-glare], .collapsible-section, .kpi-card, .citation-card, .chunk-card, .corpus-card"
    );
    targets.forEach((el) => {
      if (el.dataset.rbGlareBound === "1") return;
      el.dataset.rbGlareBound = "1";
      el.classList.add("rb-glare-hover");
    });
  }

  /* =========================================================================
   * 4. REACT BITS: DOCK NAVIGATION (From Dock-JS-CSS.json)
   *    Smooth proximity-scaled floating dock items
   * ========================================================================= */
  function initDockNav() {
    const dock = document.querySelector("[data-rb-dock]");
    if (!dock) return;
    const items = Array.from(dock.querySelectorAll(".rb-dock-item"));
    if (!items.length) return;

    dock.addEventListener("pointermove", (e) => {
      if (reduceMotion()) return;
      items.forEach((item) => {
        const rect = item.getBoundingClientRect();
        const center = rect.left + rect.width / 2;
        const dist = Math.abs(e.clientX - center);
        const factor = Math.max(0, 1 - dist / 150);
        const scale = 1 + factor * 0.08;
        const lift = -factor * 3;
        item.style.transform = `translateY(${lift.toFixed(2)}px) scale(${scale.toFixed(3)})`;
      });
    });

    dock.addEventListener("pointerleave", () => {
      items.forEach((item) => {
        item.style.transform = "";
      });
    });
  }

  /* =========================================================================
   * 5. REACT BITS: GRADIENT TEXT (From GradientText-JS-CSS.json)
   * ========================================================================= */
  function initGradientText(root = document) {
    const els = root.querySelectorAll("[data-rb-gradient-text]");
    els.forEach((el) => {
      if (el.dataset.rbGradBound === "1") return;
      el.dataset.rbGradBound = "1";

      const text = el.textContent;
      const colors = (el.dataset.colors || "#38BDF8,#818CF8,#C084FC,#34D399").split(",");
      const speed = parseFloat(el.dataset.speed || "6");
      const glow = parseFloat(el.dataset.glow || "0.38");

      const palette = [...colors, colors[0]].join(", ");
      const gradient = `linear-gradient(90deg, ${palette})`;

      el.classList.add("gradient-text");
      el.style.setProperty("--gt-gradient", gradient);
      el.style.setProperty("--gt-glow", String(glow));
      el.textContent = "";

      const inner = document.createElement("span");
      inner.className = "gradient-text__inner";

      const glowSpan = document.createElement("span");
      glowSpan.className = "gradient-text__glow";
      glowSpan.setAttribute("aria-hidden", "true");
      glowSpan.textContent = text;

      const contentSpan = document.createElement("span");
      contentSpan.className = "gradient-text__content";
      contentSpan.textContent = text;

      inner.appendChild(glowSpan);
      inner.appendChild(contentSpan);
      el.appendChild(inner);

      let elapsed = 0;
      let lastTime = null;
      const animDuration = speed * 1000;

      function step(time) {
        if (lastTime === null) lastTime = time;
        const dt = time - lastTime;
        lastTime = time;
        elapsed += dt;

        const fullCycle = animDuration * 2;
        const cycleTime = elapsed % fullCycle;
        const p =
          cycleTime < animDuration
            ? (cycleTime / animDuration) * 100
            : 100 - ((cycleTime - animDuration) / animDuration) * 100;

        const pos = `${p.toFixed(2)}% 50%`;
        contentSpan.style.backgroundPosition = pos;
        glowSpan.style.backgroundPosition = pos;
        contentSpan.style.backgroundSize = "300% 100%";
        glowSpan.style.backgroundSize = "300% 100%";

        if (!reduceMotion()) {
          requestAnimationFrame(step);
        }
      }
      requestAnimationFrame(step);
    });
  }

  /* =========================================================================
   * 6. REACT BITS: SHINY TEXT (From ShinyText-JS-CSS.json)
   * ========================================================================= */
  function initShinyText(root = document) {
    const els = root.querySelectorAll("[data-rb-shiny]");
    els.forEach((el) => {
      if (el.dataset.rbShinyBound === "1") return;
      el.dataset.rbShinyBound = "1";
      el.classList.add("rb-shiny-text");
    });
  }

  /* =========================================================================
   * 7. REACT BITS: BLUR TEXT (From BlurText-JS-CSS.json)
   * ========================================================================= */
  function initBlurText(root = document) {
    const els = root.querySelectorAll("[data-rb-blur-text]");
    els.forEach((el) => {
      if (el.dataset.rbBlurBound === "1") return;
      el.dataset.rbBlurBound = "1";

      const text = el.textContent.trim();
      const words = text.split(/\s+/);
      el.textContent = "";
      el.classList.add("rb-blur-text");

      words.forEach((word, idx) => {
        const span = document.createElement("span");
        span.className = "rb-blur-word";
        span.style.animationDelay = `${idx * 65}ms`;
        span.textContent = word;
        el.appendChild(span);
        if (idx < words.length - 1) {
          el.appendChild(document.createTextNode(" "));
        }
      });
    });
  }

  /* =========================================================================
   * 8. REACT BITS: STAR BORDER (From StarBorder-JS-CSS.json)
   * ========================================================================= */
  const STAR_BORDER_PAD = 28;

  function parseStarColor(input) {
    const probe = document.createElement("span");
    probe.style.color = input;
    probe.style.display = "none";
    document.body.appendChild(probe);
    const computed = getComputedStyle(probe).color;
    document.body.removeChild(probe);
    const match = computed.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/);
    return match ? [+match[1], +match[2], +match[3]] : [96, 165, 250];
  }

  function pointOnRoundedRect(s, w, h, r) {
    const R = Math.max(0, Math.min(r, w / 2, h / 2));
    const top = w - 2 * R;
    const right = h - 2 * R;
    const arc = 0.5 * Math.PI * R;
    const total = 2 * (top + right) + 4 * arc;
    if (total <= 0) return { x: w / 2, y: h / 2, nx: 0, ny: -1 };
    let d = ((s % total) + total) % total;

    if (d < top) return { x: R + d, y: 0, nx: 0, ny: -1 };
    d -= top;
    if (d < arc) {
      const a = -Math.PI / 2 + d / R;
      return { x: w - R + Math.cos(a) * R, y: R + Math.sin(a) * R, nx: Math.cos(a), ny: Math.sin(a) };
    }
    d -= arc;
    if (d < right) return { x: w, y: R + d, nx: 1, ny: 0 };
    d -= right;
    if (d < arc) {
      const a = d / R;
      return { x: w - R + Math.cos(a) * R, y: h - R + Math.sin(a) * R, nx: Math.cos(a), ny: Math.sin(a) };
    }
    d -= arc;
    if (d < top) return { x: w - R - d, y: h, nx: 0, ny: 1 };
    d -= top;
    if (d < arc) {
      const a = Math.PI / 2 + d / R;
      return { x: R + Math.cos(a) * R, y: h - R + Math.sin(a) * R, nx: Math.cos(a), ny: Math.sin(a) };
    }
    d -= arc;
    if (d < right) return { x: 0, y: h - R - d, nx: -1, ny: 0 };
    d -= right;
    const a = Math.PI + (R > 0 ? d / R : 0);
    return { x: R + Math.cos(a) * R, y: R + Math.sin(a) * R, nx: Math.cos(a), ny: Math.sin(a) };
  }

  function attachStarBorder(el, color = "#60a5fa", speedSeconds = 6.5) {
    if (!el || el.dataset.rbStarBound === "1") return;
    el.dataset.rbStarBound = "1";
    el.classList.add("rb-star-border");

    const canvas = document.createElement("canvas");
    canvas.className = "rb-star-border__canvas";
    canvas.setAttribute("aria-hidden", "true");
    el.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const [r, g, b] = parseStarColor(color);
    const hr = Math.min(255, Math.round(r + (255 - r) * 0.72));
    const hg = Math.min(255, Math.round(g + (255 - g) * 0.72));
    const hb = Math.min(255, Math.round(b + (255 - b) * 0.72));

    let width = 0;
    let height = 0;
    let radius = 10;
    let perimeter = 0;
    let dpr = 1;
    let phase = 0;
    let lastTime = performance.now();

    const measure = () => {
      const rect = el.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      const cs = getComputedStyle(el);
      radius = parseFloat(cs.borderTopLeftRadius) || 10;
      const R = Math.max(0, Math.min(radius, width / 2, height / 2));
      perimeter = 2 * (width - 2 * R + (height - 2 * R)) + 2 * Math.PI * R;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round((width + STAR_BORDER_PAD * 2) * dpr));
      canvas.height = Math.max(1, Math.round((height + STAR_BORDER_PAD * 2) * dpr));
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);

    const draw = (now) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      if (!reduceMotion()) {
        phase = (phase + dt / speedSeconds) % 1;
      }

      if (width > 0 && height > 0 && perimeter > 0) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width + STAR_BORDER_PAD * 2, height + STAR_BORDER_PAD * 2);
        ctx.save();
        ctx.translate(STAR_BORDER_PAD, STAR_BORDER_PAD);
        ctx.globalCompositeOperation = "lighter";

        const headDist = phase * perimeter;
        const trailLen = perimeter * 0.24;
        const steps = 32;

        for (let i = 0; i < 2; i++) {
          const offset = i * (perimeter * 0.5);
          const hDist = headDist + offset;
          for (let j = steps - 1; j >= 0; j--) {
            const u0 = j / steps;
            const u1 = (j + 1) / steps;
            const p0 = pointOnRoundedRect(hDist - u0 * trailLen, width, height, radius);
            const p1 = pointOnRoundedRect(hDist - u1 * trailLen, width, height, radius);
            const fade = Math.pow(1 - u0, 1.85);

            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${fade * 0.7})`;
            ctx.lineWidth = 1.6 * Math.pow(1 - u0, 0.45);
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          }

          const head = pointOnRoundedRect(hDist, width, height, radius);
          const bloom = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, 12);
          bloom.addColorStop(0, `rgba(${hr}, ${hg}, ${hb}, 0.9)`);
          bloom.addColorStop(0.45, `rgba(${r}, ${g}, ${b}, 0.35)`);
          bloom.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
          ctx.fillStyle = bloom;
          ctx.beginPath();
          ctx.arc(head.x, head.y, 12, 0, TAU);
          ctx.fill();
        }
        ctx.restore();
      }

      requestAnimationFrame(draw);
    };

    requestAnimationFrame(draw);
  }

  function initStarBorders() {
    document.querySelectorAll(".primary-btn").forEach((btn) => {
      attachStarBorder(btn, "#60a5fa", 6.0);
    });
  }

  /* =========================================================================
   * 9. REACT BITS: CLICK SPARK (From ClickSpark-JS-CSS.json)
   * ========================================================================= */
  function initClickSpark() {
    const canvas = document.createElement("canvas");
    canvas.className = "rb-clickspark-canvas";
    canvas.setAttribute("aria-hidden", "true");
    document.body.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const sparks = [];
    let rafId = 0;
    let dpr = 1;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const colors = [
      [56, 189, 248],
      [192, 132, 252],
      [52, 211, 153],
    ];

    const render = (now) => {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      ctx.save();
      ctx.globalCompositeOperation = "lighter";

      for (let i = sparks.length - 1; i >= 0; i--) {
        const s = sparks[i];
        const t = (now - s.startTime) / s.duration;
        if (t >= 1) {
          sparks.splice(i, 1);
          continue;
        }
        const eased = 1 - Math.pow(1 - t, 3);
        const fade = Math.pow(1 - t, 1.4);
        const dist = eased * s.radius;
        const len = s.size * (1 - eased * 0.55);
        const cos = Math.cos(s.angle);
        const sin = Math.sin(s.angle);
        const hx = s.x + cos * (dist + len);
        const hy = s.y + sin * (dist + len);
        const tx = s.x + cos * dist;
        const ty = s.y + sin * dist;

        const [cr, cg, cb] = s.rgb;
        const grad = ctx.createLinearGradient(tx, ty, hx, hy);
        grad.addColorStop(0, `rgba(${cr}, ${cg}, ${cb}, 0)`);
        grad.addColorStop(0.6, `rgba(${cr}, ${cg}, ${cb}, ${fade * 0.9})`);
        grad.addColorStop(1, `rgba(255, 255, 255, ${fade})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.8 * (1 - t * 0.4);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(hx, hy);
        ctx.stroke();
      }

      ctx.restore();
      if (sparks.length > 0) {
        rafId = requestAnimationFrame(render);
      } else {
        rafId = 0;
      }
    };

    document.addEventListener(
      "pointerdown",
      (e) => {
        if (reduceMotion() || e.button !== 0) return;
        const now = performance.now();
        const count = 9;
        const rgb = colors[Math.floor(Math.random() * colors.length)];
        for (let i = 0; i < count; i++) {
          const base = (TAU * i) / count;
          sparks.push({
            x: e.clientX,
            y: e.clientY,
            angle: base + (Math.random() - 0.5) * 0.35,
            radius: 26 * (0.8 + Math.random() * 0.45),
            size: 11 * (0.8 + Math.random() * 0.4),
            duration: 420 * (0.85 + Math.random() * 0.3),
            startTime: now,
            rgb,
          });
        }
        if (!rafId) rafId = requestAnimationFrame(render);
      },
      { passive: true }
    );
  }

  /* =========================================================================
   * 10. REACT BITS: COUNT UP (Animated numeric counter)
   * ========================================================================= */
  function animateCountUp(el, targetValue, options = {}) {
    if (!el) return;
    const duration = options.duration || 520;
    const decimals = options.decimals ?? 0;
    const suffix = options.suffix || "";
    const prefix = options.prefix || "";

    if (reduceMotion() || Number.isNaN(targetValue)) {
      el.textContent = `${prefix}${Number(targetValue).toFixed(decimals)}${suffix}`;
      return;
    }

    const startTime = performance.now();
    function tick(now) {
      const progress = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = targetValue * eased;
      el.textContent = `${prefix}${val.toFixed(decimals)}${suffix}`;
      if (progress < 1) {
        requestAnimationFrame(tick);
      } else {
        el.textContent = `${prefix}${Number(targetValue).toFixed(decimals)}${suffix}`;
      }
    }
    requestAnimationFrame(tick);
  }

  /* =========================================================================
   * BOOTSTRAP & PUBLIC HOOKS
   * ========================================================================= */
  function initAll() {
    initWavesBackdrop();
    initDockNav();
    initPixelCards();
    initGlareHover(document);
    initGradientText(document);
    initShinyText(document);
    initBlurText(document);
    initStarBorders();
    initClickSpark();
  }

  window.ReactBitsFX = {
    refreshDynamicCards(root = document) {
      initGlareHover(root);
      initPixelCards();
      initGradientText(root);
      initShinyText(root);
    },
    animateCountUp,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
