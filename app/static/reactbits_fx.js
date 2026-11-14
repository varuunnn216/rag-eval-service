/**
 * Official React Bits (@react-bits) Vanilla JS / WebGL2 / Canvas Runtime
 * Ported directly from https://reactbits.dev/r/{Component}-JS-CSS.json via React Bits MCP:
 *  1. Aurora (WebGL2 3-layer OKLCH simplex-noise northern lights shader with pointer reach)
 *  2. GradientText (Flowing multi-stop radial/linear gradient text with glow layer)
 *  3. ShinyText (Metallic light-sweep shimmer across badges & subtitles)
 *  4. BlurText (Staggered blur-to-sharp word entrance reveal on headers)
 *  5. DecryptedText (Cipher glyph scramble reveal on monospace IDs & badges)
 *  6. SpotlightCard (Proximity-aware radial cursor spotlight + glowing rim edge + fractal grain)
 *  7. StarBorder (Orbiting glowing star trail + stardust motes + click pulse wave on buttons)
 *  8. ClickSpark (Bursting star & streak sparks on interactive clicks)
 *  9. CountUp (Cubic-out animated numeric counters for metrics & KPIs)
 */

(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const clamp = (val, min, max) => Math.min(max, Math.max(min, val));
  const reduceMotion = () =>
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

  /* =========================================================================
   * 1. REACT BITS: AURORA (WebGL2 Shader from Aurora-JS-CSS.json)
   * ========================================================================= */
  const MAX_STOPS = 5;
  const MAX_LAYERS = 3;
  const INTRO_SECONDS = 2.0;

  const AURORA_VERTEX = `#version 300 es
in vec2 aPosition;
void main() {
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

  const AURORA_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uStops[${MAX_STOPS}];
uniform int uStopCount;
uniform float uAmplitude;
uniform float uBlend;
uniform float uHeight;
uniform float uScale;
uniform float uGlow;
uniform float uBrightness;
uniform int uLayers;
uniform float uFlip;
uniform vec3 uPointer;
uniform float uReveal;
uniform float uOpacity;
out vec4 outColor;

vec3 permute(vec3 x) {
  return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float smootherstep(float edge0, float edge1, float x) {
  float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

vec3 oklabToLinear(vec3 c) {
  float l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  float m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  float s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  l = l * l * l;
  m = m * m * m;
  s = s * s * s;
  return vec3(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );
}

vec3 toSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 ramp(float t) {
  float x = clamp(t, 0.0, 1.0) * float(uStopCount - 1);
  int i = int(min(floor(x), float(uStopCount - 2)));
  float f = x - float(i);
  f = f * f * (3.0 - 2.0 * f);
  vec3 lch = mix(uStops[i], uStops[i + 1], f);
  return toSrgb(oklabToLinear(vec3(lch.x, lch.y * cos(lch.z), lch.y * sin(lch.z))));
}

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  float y = 1.0 - frag.y / uResolution.y;
  y = mix(y, 1.0 - y, uFlip);
  float u = frag.x / uResolution.x;
  float X = (frag.x - uResolution.x * 0.5) / uResolution.y;
  float t = uTime;
  float reach = uPointer.z * exp(-pow((X - uPointer.x) / 0.38, 2.0));

  vec3 glow = vec3(0.0);
  float cover = 0.0;
  for (int k = 0; k < ${MAX_LAYERS}; k++) {
    if (k >= uLayers) break;
    float layer = float(k);
    float seed = layer * 13.7;
    float pace = t * (1.0 - layer * 0.18);
    float warp = snoise(vec2(X * 0.45 * uScale + pace * 0.02 + seed, pace * 0.05 + seed));
    float wave = snoise(vec2(X * 0.75 * uScale + warp * 0.35 - pace * 0.03 + seed, pace * 0.07 - seed)) * 0.72
      + snoise(vec2(X * 1.6 * uScale + warp * 0.5 + pace * 0.04 - seed, pace * 0.1 + seed * 0.5)) * 0.28;
    float hem = uHeight + layer * 0.07 + wave * 0.15 * uAmplitude;
    hem += reach * 0.08 * (layer < 0.5 ? 1.0 : 0.4);
    hem = mix(-0.25, hem, uReveal);

    float soft = (0.04 + uBlend * 0.16) * (1.0 + layer * 0.9);
    float rise = hem - y;
    float low = -soft - uGlow * 0.14;
    float high = 0.26 + soft * 0.6;
    float body = pow(smootherstep(low, high, rise), 1.5 + uGlow * 0.6);
    float lobes = 0.86 + 0.14 * snoise(vec2(X * 0.9 * uScale - pace * 0.025 + seed, y * 0.3 + pace * 0.04));
    float alpha = body * lobes * (layer < 0.5 ? 1.0 : 0.32 / layer);
    alpha = clamp(alpha * (1.0 + reach * 0.25 * smootherstep(0.9, 0.0, abs(y - hem))), 0.0, 1.0);

    vec3 tint = ramp(u + 0.05 * sin(pace * 0.08 + layer * 2.3) + layer * 0.1 + max(rise, 0.0) * 0.05);
    glow = 1.0 - (1.0 - glow) * (1.0 - tint * alpha);
    cover = 1.0 - (1.0 - cover) * (1.0 - alpha);
  }

  float fade = smootherstep(0.0, 0.7, uReveal) * uOpacity;
  float strength = clamp(uBrightness, 0.0, 2.0);
  glow *= fade * min(strength, 1.0);
  cover *= fade * min(strength, 1.0);
  glow = 1.0 - pow(1.0 - glow, vec3(max(strength, 1.0)));
  float dither = (hash(frag) - 0.5) / 255.0;
  outColor = vec4(glow + dither, clamp(cover + dither, 0.0, 1.0));
}`;

  const toOklch = (rgb) => {
    const [r, g, b] = rgb.map((c) =>
      c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    );
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const lab = [
      0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
    ];
    return [lab[0], Math.hypot(lab[1], lab[2]), Math.atan2(lab[2], lab[1])];
  };

  const unwrapHues = (stops) => {
    const known = stops.findIndex((stop) => stop[1] > 0.02);
    const anchor = known === -1 ? 0 : stops[known][2];
    let previous = anchor;
    return stops.map((stop) => {
      let hue = stop[1] > 0.02 ? stop[2] : previous;
      while (hue - previous > Math.PI) hue -= Math.PI * 2;
      while (hue - previous < -Math.PI) hue += Math.PI * 2;
      previous = hue;
      return [stop[0], stop[1], hue];
    });
  };

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
    gl.deleteShader(shader);
    return null;
  }

  function linkProgram(gl) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, AURORA_VERTEX);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, AURORA_FRAGMENT);
    const prog = gl.createProgram();
    if (!vs || !fs || !prog) return null;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, "aPosition");
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      gl.deleteProgram(prog);
      return null;
    }
    const uniforms = {};
    const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(prog, i);
      if (info) {
        uniforms[info.name.replace("[0]", "")] = gl.getUniformLocation(prog, info.name);
      }
    }
    return { program: prog, uniforms };
  }

  function initAuroraBackdrop() {
    const container = document.getElementById("reactbits-aurora");
    if (!container) return;
    const canvas = container.querySelector("canvas");
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      depth: false,
      stencil: false,
    });
    if (!gl) return;

    const linked = linkProgram(gl);
    if (!linked) return;
    const u = linked.uniforms;

    const buffer = gl.createBuffer();
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const stops = ["#3B82F6", "#8B5CF6", "#06B6D4", "#10B981", "#6366F1"];
    const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    const parseColor = (hex) => {
      probe.clearRect(0, 0, 1, 1);
      probe.fillStyle = hex;
      probe.fillRect(0, 0, 1, 1);
      const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
      return [r / 255, g / 255, b / 255];
    };
    const palette = new Float32Array(MAX_STOPS * 3);
    unwrapHues(stops.map((c) => toOklch(parseColor(c)))).forEach((s, idx) =>
      palette.set(s, idx * 3)
    );

    const view = { width: 1, height: 1 };
    const pointer = { x: 0, y: 0, inside: false };
    const state = {
      clock: 0,
      px: 0,
      py: 0.4,
      reach: 0,
      reveal: reduceMotion() ? 1 : 0,
    };
    let last = 0;

    function render() {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(linked.program);
      gl.bindVertexArray(vao);
      gl.uniform2f(u.uResolution, canvas.width, canvas.height);
      gl.uniform1f(u.uTime, state.clock);
      gl.uniform3fv(u.uStops, palette);
      gl.uniform1i(u.uStopCount, stops.length);
      gl.uniform1f(u.uAmplitude, 1.15);
      gl.uniform1f(u.uBlend, 0.58);
      gl.uniform1f(u.uHeight, 0.48);
      gl.uniform1f(u.uScale, 1.05);
      gl.uniform1f(u.uGlow, 0.72);
      gl.uniform1f(u.uBrightness, 1.05);
      gl.uniform1i(u.uLayers, 3);
      gl.uniform1f(u.uFlip, 0);
      gl.uniform3f(u.uPointer, state.px, state.py, state.reach);
      gl.uniform1f(u.uReveal, 1 - Math.pow(1 - state.reveal, 3));
      gl.uniform1f(u.uOpacity, 0.88);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function frame(now) {
      if (document.hidden) {
        requestAnimationFrame(frame);
        return;
      }
      const dt = last ? clamp((now - last) / 1000, 0.001, 0.05) : 1 / 60;
      last = now;
      if (!reduceMotion()) {
        state.clock += dt * 0.85;
      }
      const follow = 1 - Math.exp(-dt / 0.55);
      if (pointer.inside) {
        const tx = (pointer.x - view.width / 2) / Math.max(1, view.height);
        const ty = pointer.y / Math.max(1, view.height);
        state.px += (tx - state.px) * follow;
        state.py += (ty - state.py) * follow;
      }
      state.reach +=
        ((pointer.inside ? 1 : 0) - state.reach) *
        (1 - Math.exp(-dt / (pointer.inside ? 0.8 : 1.3)));
      if (state.reveal < 1) {
        state.reveal = Math.min(1, state.reveal + dt / INTRO_SECONDS);
      }
      render();
      requestAnimationFrame(frame);
    }

    function resize() {
      view.width = Math.max(1, container.clientWidth);
      view.height = Math.max(1, container.clientHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      canvas.width = Math.round(view.width * dpr);
      canvas.height = Math.round(view.height * dpr);
      render();
    }

    window.addEventListener(
      "pointermove",
      (e) => {
        const rect = container.getBoundingClientRect();
        pointer.x = e.clientX - rect.left;
        pointer.y = e.clientY - rect.top;
        pointer.inside =
          pointer.x >= 0 &&
          pointer.y >= 0 &&
          pointer.x <= rect.width &&
          pointer.y <= rect.height;
      },
      { passive: true }
    );

    window.addEventListener("resize", resize);
    resize();
    requestAnimationFrame(frame);
  }

  /* =========================================================================
   * 2. REACT BITS: GRADIENT TEXT (from GradientText-JS-CSS.json)
   * ========================================================================= */
  function initGradientTexts() {
    const nodes = document.querySelectorAll("[data-rb-gradient-text]");
    if (!nodes.length) return;

    const items = [];
    nodes.forEach((node) => {
      if (node.dataset.rbGtReady === "1") return;
      node.dataset.rbGtReady = "1";
      const text = node.textContent;
      const colors = (
        node.dataset.colors || "#60A5FA,#C084FC,#38BDF8,#34D399,#818CF8"
      )
        .split(",")
        .map((s) => s.trim());
      const speed = parseFloat(node.dataset.speed || "6");
      const glow = parseFloat(node.dataset.glow || "0.45");

      node.classList.add("gradient-text");
      node.style.setProperty("--gt-glow", String(glow));
      node.innerHTML = `<span class="gradient-text__inner"><span class="gradient-text__content">${text}</span><span class="gradient-text__glow" aria-hidden="true">${text}</span></span>`;
      items.push({ node, colors, speed, time: Math.random() * 4 });
    });

    if (!items.length) return;
    let last = performance.now();
    function tick(now) {
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
      last = now;
      items.forEach((it) => {
        it.time += dt;
        const cycle = it.time / it.speed;
        const blobs = it.colors.map((color, i) => {
          const spin = cycle * TAU;
          const rate = 0.55 + ((i * 0.618) % 1) * 0.5;
          const x = 50 + 44 * Math.sin(spin * rate + i * 2.1);
          const y = 50 + 40 * Math.sin(spin * (rate * 0.83 + 0.21) + i * 1.3);
          return `radial-gradient(ellipse 58% 145% at ${x.toFixed(1)}% ${y.toFixed(1)}%, ${color} 0%, transparent 100%)`;
        });
        const image = `${blobs.join(", ")}, linear-gradient(125deg, ${it.colors.join(", ")})`;
        it.node.style.setProperty("--gt-gradient", image);
      });
      if (!reduceMotion()) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* =========================================================================
   * 3. REACT BITS: SHINY TEXT (from ShinyText-JS-CSS.json)
   * ========================================================================= */
  function initShinyTexts() {
    document.querySelectorAll("[data-rb-shiny]").forEach((el) => {
      if (el.dataset.rbShinyReady === "1") return;
      el.dataset.rbShinyReady = "1";
      el.classList.add("rb-shiny-text");
    });
  }

  /* =========================================================================
   * 4. REACT BITS: BLUR TEXT (from BlurText-JS-CSS.json)
   * ========================================================================= */
  function initBlurTexts() {
    document.querySelectorAll("[data-rb-blur-text]").forEach((el) => {
      if (el.dataset.rbBlurReady === "1") return;
      el.dataset.rbBlurReady = "1";
      const raw = el.textContent.trim();
      if (!raw) return;
      const words = raw.split(/\s+/);
      el.innerHTML = words
        .map(
          (w, i) =>
            `<span class="rb-blur-word" style="animation-delay:${(i * 65).toFixed(0)}ms">${w}</span>`
        )
        .join(" ");
    });
  }

  /* =========================================================================
   * 5. REACT BITS: DECRYPTED TEXT (from DecryptedText-JS-CSS.json)
   * ========================================================================= */
  const CIPHER_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*+<>";

  function bindDecryptedHover(root = document) {
    root
      .querySelectorAll(".chunk-id-tag, .state-pill, [data-rb-decrypt]")
      .forEach((el) => {
        if (el.dataset.rbDecryptBound === "1") return;
        el.dataset.rbDecryptBound = "1";
        el.addEventListener("pointerenter", () => {
          if (reduceMotion() || el.dataset.rbDecryptRunning === "1") return;
          const original = el.dataset.rbOrig || el.textContent;
          el.dataset.rbOrig = original;
          el.dataset.rbDecryptRunning = "1";
          const chars = Array.from(original);
          let iter = 0;
          const maxIter = 9;
          const timer = setInterval(() => {
            iter++;
            const revealedCount = Math.floor((iter / maxIter) * chars.length);
            el.textContent = chars
              .map((ch, idx) => {
                if (/\s/.test(ch) || idx < revealedCount) return ch;
                return CIPHER_POOL[Math.floor(Math.random() * CIPHER_POOL.length)];
              })
              .join("");
            if (iter >= maxIter) {
              clearInterval(timer);
              el.textContent = original;
              el.dataset.rbDecryptRunning = "0";
            }
          }, 28);
        });
      });
  }

  /* =========================================================================
   * 6. REACT BITS: SPOTLIGHT CARD (from SpotlightCard-JS-CSS.json)
   * ========================================================================= */
  const spotlightCards = new Set();
  const globalPointer = { x: -9999, y: -9999, active: false };
  let spotlightRaf = 0;

  function registerSpotlightCards(root = document) {
    const selectors = [
      ".collapsible-section",
      ".control-card",
      ".citation-card",
      ".chunk-card",
      ".metric-card",
      ".kpi-box",
      ".doc-card",
      ".card",
    ];
    root.querySelectorAll(selectors.join(",")).forEach((card) => {
      if (card.dataset.rbSpotlight === "1") return;
      card.dataset.rbSpotlight = "1";
      card.classList.add("rb-spotlight-card");

      const light = document.createElement("span");
      light.className = "rb-spotlight__light";
      light.setAttribute("aria-hidden", "true");

      const edge = document.createElement("span");
      edge.className = "rb-spotlight__edge";
      edge.setAttribute("aria-hidden", "true");

      card.insertBefore(light, card.firstChild);
      card.appendChild(edge);

      const isSection =
        card.classList.contains("collapsible-section") || card.classList.contains("card");
      spotlightCards.add({
        root: card,
        light,
        edge,
        x: 0,
        y: 0,
        presence: 0,
        size: isSection ? 420 : 250,
        proximity: isSection ? 110 : 75,
        intensity: isSection ? 0.16 : 0.22,
      });
    });
  }

  function stepSpotlights() {
    spotlightRaf = 0;
    let keepAlive = false;

    spotlightCards.forEach((item) => {
      if (!document.body.contains(item.root)) {
        spotlightCards.delete(item);
        return;
      }
      const rect = item.root.getBoundingClientRect();
      if (rect.bottom < -100 || rect.top > window.innerHeight + 100) return;

      let tx = item.x;
      let ty = item.y;
      let target = 0;

      if (globalPointer.active) {
        tx = globalPointer.x - rect.left;
        ty = globalPointer.y - rect.top;
        const gap = Math.hypot(
          Math.max(0, -tx, tx - rect.width),
          Math.max(0, -ty, ty - rect.height)
        );
        if (gap === 0) target = 1;
        else if (item.proximity > 0 && gap < item.proximity) {
          target = Math.pow(1 - gap / item.proximity, 2);
        }
      }

      if (item.presence < 0.02 && target > 0) {
        item.x = tx;
        item.y = ty;
      } else {
        item.x += (tx - item.x) * 0.28;
        item.y += (ty - item.y) * 0.28;
      }
      item.presence += (target - item.presence) * (target > item.presence ? 0.26 : 0.14);
      if (Math.abs(target - item.presence) < 0.003) item.presence = target;

      if (item.presence > 0.002) {
        item.light.style.background = `radial-gradient(circle ${item.size}px at ${item.x.toFixed(1)}px ${item.y.toFixed(1)}px, rgba(96, 165, 250, 0.22), rgba(139, 92, 246, 0.09) 45%, transparent 100%)`;
        item.edge.style.background = `radial-gradient(circle ${(item.size * 0.65).toFixed(1)}px at ${item.x.toFixed(1)}px ${item.y.toFixed(1)}px, rgba(147, 197, 253, 0.85), rgba(167, 139, 250, 0.35) 50%, transparent 100%)`;
        item.light.style.opacity = String(Math.min(1, item.presence * (item.intensity * 4.2)));
        item.edge.style.opacity = String(Math.min(1, item.presence * 0.95));
      } else {
        item.light.style.opacity = "0";
        item.edge.style.opacity = "0";
      }

      if (Math.abs(target - item.presence) > 0.004 || Math.hypot(tx - item.x, ty - item.y) > 0.5) {
        keepAlive = true;
      }
    });

    if (keepAlive) {
      spotlightRaf = requestAnimationFrame(stepSpotlights);
    }
  }

  function wakeSpotlights() {
    if (!spotlightRaf) {
      spotlightRaf = requestAnimationFrame(stepSpotlights);
    }
  }

  /* =========================================================================
   * 7. REACT BITS: STAR BORDER (from StarBorder-JS-CSS.json)
   * ========================================================================= */
  const BLEED = 32;

  function outlineRect(width, height, radius) {
    const r = Math.max(0, Math.min(radius, width / 2, height / 2));
    const arc = (Math.PI * r) / 2;
    const across = Math.max(0, width - 2 * r);
    const down = Math.max(0, height - 2 * r);
    const segments = [
      { length: across, x: r, y: 0, dx: 1, dy: 0, angle: null },
      { length: arc, x: width - r, y: r, dx: 0, dy: 0, angle: -Math.PI / 2 },
      { length: down, x: width, y: r, dx: 0, dy: 1, angle: null },
      { length: arc, x: width - r, y: height - r, dx: 0, dy: 0, angle: 0 },
      { length: across, x: width - r, y: height, dx: -1, dy: 0, angle: null },
      { length: arc, x: r, y: height - r, dx: 0, dy: 0, angle: Math.PI / 2 },
      { length: down, x: 0, y: height - r, dx: 0, dy: -1, angle: null },
      { length: arc, x: r, y: r, dx: 0, dy: 0, angle: Math.PI },
    ];
    return {
      r,
      segments,
      total: Math.max(
        1,
        segments.reduce((s, seg) => s + seg.length, 0)
      ),
    };
  }

  function pointOnOutline(shape, dist, out) {
    let s = ((dist % shape.total) + shape.total) % shape.total;
    for (const seg of shape.segments) {
      if (seg.length > 0 && s <= seg.length) {
        if (seg.angle === null) {
          out.x = seg.x + seg.dx * s;
          out.y = seg.y + seg.dy * s;
          out.nx = seg.dy;
          out.ny = -seg.dx;
        } else {
          const a = seg.angle + s / shape.r;
          out.nx = Math.cos(a);
          out.ny = Math.sin(a);
          out.x = seg.x + out.nx * shape.r;
          out.y = seg.y + out.ny * shape.r;
        }
        return out;
      }
      s -= seg.length;
    }
    return out;
  }

  function attachStarBorder(btn, opts = {}) {
    if (!btn || btn.dataset.rbStarBorder === "1") return;
    btn.dataset.rbStarBorder = "1";
    btn.classList.add("rb-star-border");

    const canvas = document.createElement("canvas");
    canvas.className = "rb-star-border__canvas";
    canvas.setAttribute("aria-hidden", "true");
    btn.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const headRgb = opts.headRgb || [147, 197, 253];
    const tailRgb = opts.tailRgb || [139, 92, 246];
    const duration = opts.duration || 3.4;
    const radius = opts.radius || 10;
    const thickness = 1.6;
    const pt = { x: 0, y: 0, nx: 0, ny: -1 };
    const dust = [];

    let width = 0;
    let height = 0;
    let dpr = 1;
    let phase = Math.random() * 200;
    let boost = 0;
    let hovered = false;
    let last = performance.now();

    function measure() {
      width = btn.offsetWidth;
      height = btn.offsetHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round((width + BLEED * 2) * dpr));
      canvas.height = Math.max(1, Math.round((height + BLEED * 2) * dpr));
    }

    btn.addEventListener("pointerenter", () => {
      hovered = true;
    });
    btn.addEventListener("pointerleave", () => {
      hovered = false;
    });
    new ResizeObserver(measure).observe(btn);
    measure();

    function frame(now) {
      const dt = Math.min(0.05, Math.max(0.001, (now - last) / 1000));
      last = now;

      if (width > 4 && height > 4 && !document.hidden) {
        const shape = outlineRect(
          Math.max(1, width - thickness),
          Math.max(1, height - thickness),
          radius
        );
        const speed = (shape.total / duration) * (hovered ? 1.65 : 1);
        phase = (phase + speed * dt) % shape.total;
        boost += ((hovered ? 1 : 0) - boost) * 0.15;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const offset = (BLEED + thickness / 2) * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, offset, offset);
        ctx.globalCompositeOperation = "lighter";

        const trailLen = shape.total * 0.32;
        const steps = 36;
        pointOnOutline(shape, phase, pt);
        let px = pt.x;
        let py = pt.y;

        for (let i = 1; i <= steps; i++) {
          pointOnOutline(shape, phase - trailLen * (i / steps), pt);
          const u = 1 - i / steps;
          const alpha = u * u * (0.75 + 0.25 * boost);
          const r = Math.round(tailRgb[0] + (headRgb[0] - tailRgb[0]) * u);
          const g = Math.round(tailRgb[1] + (headRgb[1] - tailRgb[1]) * u);
          const b = Math.round(tailRgb[2] + (headRgb[2] - tailRgb[2]) * u);

          ctx.strokeStyle = `rgba(${r},${g},${b},${(alpha * 0.28).toFixed(3)})`;
          ctx.lineWidth = 5;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(pt.x, pt.y);
          ctx.stroke();

          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
          ctx.lineWidth = thickness;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(pt.x, pt.y);
          ctx.stroke();

          px = pt.x;
          py = pt.y;
        }

        // Star head bloom + cross sparkle
        pointOnOutline(shape, phase, pt);
        const bloom = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, 18);
        bloom.addColorStop(0, "rgba(255,255,255,0.95)");
        bloom.addColorStop(0.3, `rgba(${headRgb[0]},${headRgb[1]},${headRgb[2]},0.55)`);
        bloom.addColorStop(1, "rgba(59,130,246,0)");
        ctx.fillStyle = bloom;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 18, 0, TAU);
        ctx.fill();

        if (Math.random() < (hovered ? 0.45 : 0.2) && dust.length < 24) {
          dust.push({
            x: pt.x,
            y: pt.y,
            vx: pt.nx * (6 + Math.random() * 10) + (Math.random() - 0.5) * 6,
            vy: pt.ny * (6 + Math.random() * 10) + (Math.random() - 0.5) * 6,
            age: 0,
            life: 0.4 + Math.random() * 0.35,
          });
        }

        for (let i = dust.length - 1; i >= 0; i--) {
          const m = dust[i];
          m.age += dt;
          m.x += m.vx * dt;
          m.y += m.vy * dt;
          if (m.age >= m.life) {
            dust.splice(i, 1);
            continue;
          }
          const a = 1 - m.age / m.life;
          ctx.fillStyle = `rgba(191,219,254,${(a * 0.85).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(m.x, m.y, 1.2 * a, 0, TAU);
          ctx.fill();
        }
      }

      if (!reduceMotion()) requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

  /* =========================================================================
   * 8. REACT BITS: CLICK SPARK (from ClickSpark-JS-CSS.json)
   * ========================================================================= */
  function initClickSpark() {
    const canvas = document.createElement("canvas");
    canvas.className = "rb-clickspark-canvas";
    canvas.setAttribute("aria-hidden", "true");
    document.body.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let dpr = 1;
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
    }
    window.addEventListener("resize", resize);
    resize();

    const bursts = [];
    const colors = ["#60A5FA", "#A78BFA", "#34D399", "#38BDF8", "#F472B6"];
    let raf = 0;

    function drawStar(x, y, radius, rot) {
      const pinch = radius * 0.2;
      ctx.beginPath();
      ctx.moveTo(x + Math.cos(rot) * radius, y + Math.sin(rot) * radius);
      for (let i = 1; i <= 4; i++) {
        const a = rot + (i * Math.PI) / 2;
        const mid = a - Math.PI / 4;
        ctx.quadraticCurveTo(
          x + Math.cos(mid) * pinch,
          y + Math.sin(mid) * pinch,
          x + Math.cos(a) * radius,
          y + Math.sin(a) * radius
        );
      }
      ctx.closePath();
    }

    function loop(now) {
      raf = 0;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

      for (let b = bursts.length - 1; b >= 0; b--) {
        const burst = bursts[b];
        const p = (now - burst.start) / burst.duration;
        if (p >= 1) {
          bursts.splice(b, 1);
          continue;
        }
        const ease = 1 - Math.pow(1 - p, 3);
        const fade = 1 - p;

        // Expanding ring
        ctx.globalAlpha = fade * 0.45;
        ctx.strokeStyle = burst.ringColor;
        ctx.lineWidth = 1.5 * (1 - p);
        ctx.beginPath();
        ctx.arc(burst.x, burst.y, 4 + 26 * ease, 0, TAU);
        ctx.stroke();

        burst.sparks.forEach((sp) => {
          const dist = sp.radius * ease;
          const x1 = burst.x + Math.cos(sp.angle) * dist;
          const y1 = burst.y + Math.sin(sp.angle) * dist;
          const len = sp.size * Math.pow(1 - p, 1.3);
          const x2 = burst.x + Math.cos(sp.angle) * (dist + len);
          const y2 = burst.y + Math.sin(sp.angle) * (dist + len);

          ctx.globalAlpha = fade;
          ctx.strokeStyle = sp.color;
          ctx.fillStyle = sp.color;
          ctx.lineWidth = 1.8;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();

          if (sp.isStar) {
            drawStar(x2, y2, 3.8 * (1 - p * 0.5), sp.angle + p * 2);
            ctx.fill();
          }
        });
      }

      if (bursts.length > 0) {
        raf = requestAnimationFrame(loop);
      }
    }

    window.addEventListener("pointerdown", (e) => {
      if (reduceMotion() || e.button !== 0) return;
      const target = e.target.closest(
        "button, .sample-chip, .tab-btn, summary, .cite-badge, .sort-btn"
      );
      if (!target) return;

      const count = 10;
      const sparks = Array.from({ length: count }, (_, i) => ({
        angle: (TAU * i) / count + (Math.random() - 0.5) * 0.25,
        radius: 22 + Math.random() * 18,
        size: 8 + Math.random() * 7,
        color: colors[i % colors.length],
        isStar: i % 2 === 0,
      }));
      bursts.push({
        x: e.clientX,
        y: e.clientY,
        start: performance.now(),
        duration: 460,
        ringColor: colors[Math.floor(Math.random() * colors.length)],
        sparks,
      });
      if (!raf) raf = requestAnimationFrame(loop);
    });
  }

  /* =========================================================================
   * 9. REACT BITS: COUNT UP (Animated Numeric Counter)
   * ========================================================================= */
  window.animateCountUp = function (elOrId, targetValue, suffix = "", decimals = 0) {
    const el = typeof elOrId === "string" ? document.getElementById(elOrId) : elOrId;
    if (!el) return;
    const end = Number(targetValue);
    if (Number.isNaN(end) || reduceMotion()) {
      el.textContent = `${Number(targetValue || 0).toFixed(decimals)}${suffix}`;
      return;
    }
    const duration = 520;
    const startTime = performance.now();
    function step(now) {
      const p = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = end * eased;
      el.textContent = `${val.toFixed(decimals)}${suffix}`;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  };

  /* =========================================================================
   * BOOTSTRAP & MUTATION OBSERVER FOR DYNAMIC CARDS
   * ========================================================================= */
  function initAllReactBits() {
    initAuroraBackdrop();
    initGradientTexts();
    initShinyTexts();
    initBlurTexts();
    registerSpotlightCards(document);
    bindDecryptedHover(document);
    initClickSpark();

    attachStarBorder(document.getElementById("ask-submit-btn"), {
      headRgb: [147, 197, 253],
      tailRgb: [99, 102, 241],
      radius: 10,
      duration: 3.0,
    });
    attachStarBorder(document.getElementById("run-eval-btn"), {
      headRgb: [56, 189, 248],
      tailRgb: [139, 92, 246],
      radius: 10,
      duration: 3.6,
    });
    attachStarBorder(document.getElementById("ingest-submit-btn"), {
      headRgb: [52, 211, 153],
      tailRgb: [59, 130, 246],
      radius: 10,
      duration: 3.8,
    });

    window.addEventListener(
      "pointermove",
      (e) => {
        globalPointer.x = e.clientX;
        globalPointer.y = e.clientY;
        globalPointer.active = true;
        wakeSpotlights();
      },
      { passive: true }
    );

    window.addEventListener(
      "scroll",
      () => {
        if (globalPointer.active) wakeSpotlights();
      },
      { passive: true, capture: true }
    );

    // Watch dynamically rendered Citation Cards, Chunk Cards, and KPI Cards
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        m.addedNodes.forEach((node) => {
          if (node.nodeType === 1) {
            registerSpotlightCards(node.parentElement || document);
            bindDecryptedHover(node.parentElement || document);
          }
        });
      }
      wakeSpotlights();
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAllReactBits);
  } else {
    initAllReactBits();
  }
})();
