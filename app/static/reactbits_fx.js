/**
 * Official React Bits (@react-bits) Vanilla JS / WebGL2 / Canvas Runtime
 * Ported directly from https://reactbits.dev/r/{Component}-JS-CSS.json via React Bits MCP:
 *  1. GooeyNav (WebGL2 SDF liquid metaball top menu bar with multi-color droplets & clipped ink overlay)
 *  2. Prism (WebGL2 3D raymarched spectral prism light dispersion backdrop in Coral, Gold, Mint & Rose)
 *  3. Waves (Interactive Perlin-noise vector wave lines in warm coral, gold & mint)
 *  4. PixelCard (Interactive radial twinkling pixel canvas in Coral, Mint & Gold variants)
 *  5. Magnet (Spring-physics magnetic cursor pull on buttons & chips)
 *  6. GlareHover (Diagonal specular light sheen sweep across cards)
 *  7. GradientText (Flowing multi-stop gradient text with glow layer)
 *  8. ShinyText (Metallic light-sweep shimmer across badges)
 *  9. BlurText (Staggered blur-to-sharp word entrance reveal on headers)
 * 10. StarBorder (Orbiting glowing star trail in warm gold/coral/mint on primary CTAs)
 * 11. ClickSpark (Bursting star & streak sparks in Coral, Gold, Mint & Rose on clicks)
 * 12. CountUp (Cubic-out animated numeric counters for metrics & KPIs)
 */

(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const clamp = (val, min, max) => Math.min(max, Math.max(min, val));
  const clamp01 = (val) => Math.min(1, Math.max(0, val));
  const reduceMotion = () =>
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

  /* =========================================================================
   * 1. REACT BITS: GOOEY NAV (From GooeyNav-JS-CSS.json)
   *    WebGL2 SDF liquid metaball navigation pill with multi-hue droplets
   * ========================================================================= */
  const GN_MAX_DROPS = 32;
  const GN_MAX_PILLS = 4;
  const GN_MARGIN = 64;

  const GN_VERTEX = `#version 300 es
in vec2 aPosition;
void main() {
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

  const GN_FRAGMENT = `#version 300 es
precision highp float;

uniform vec2 uSize;
uniform float uRatio;
uniform vec4 uPills[${GN_MAX_PILLS}];
uniform vec4 uHover;
uniform vec4 uDrops[${GN_MAX_DROPS}];
uniform vec2 uTails[${GN_MAX_DROPS}];
uniform vec3 uTints[${GN_MAX_DROPS}];
uniform int uCount;
uniform float uSmooth;
uniform vec3 uPill;
uniform vec4 uHoverColor;
uniform float uShadow;

out vec4 outColor;

float sdPill(vec2 p, vec4 box) {
  vec2 extent = max(box.zw, vec2(0.0));
  float r = min(extent.x, extent.y);
  vec2 q = abs(p - box.xy) - extent + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.0001), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

vec4 blend(vec4 a, vec4 b, float k) {
  float h = clamp(0.5 + 0.5 * (b.w - a.w) / k, 0.0, 1.0);
  return vec4(mix(b.rgb, a.rgb, h), mix(b.w, a.w, h) - k * h * (1.0 - h));
}

vec4 field(vec2 p) {
  vec4 shape = vec4(uPill, 1e5);
  for (int i = 0; i < ${GN_MAX_PILLS}; i++) {
    vec4 pill = uPills[i];
    if (pill.z <= 0.0) continue;
    shape = blend(shape, vec4(uPill, sdPill(p, pill)), uSmooth);
  }
  for (int i = 0; i < ${GN_MAX_DROPS}; i++) {
    if (i >= uCount) break;
    vec4 drop = uDrops[i];
    if (drop.z <= 0.0) continue;
    float d = sdCapsule(p, drop.xy, drop.xy - uTails[i], drop.z);
    if (d > shape.w + uSmooth) continue;
    shape = blend(shape, vec4(uTints[i], d), uSmooth);
  }
  return shape;
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, uSize.y * uRatio - gl_FragCoord.y) / uRatio;
  vec4 shape = field(p);
  float d = shape.w;
  float aa = max(fwidth(d), 0.35);
  float body = clamp(0.5 - d / aa, 0.0, 1.0);

  vec2 grad = vec2(dFdx(d), -dFdy(d));
  vec2 n = length(grad) > 0.0001 ? normalize(grad) : vec2(0.0, 1.0);
  float rim = (1.0 - smoothstep(0.0, 1.6, -d)) * max(0.0, -n.y) * 0.35;
  vec3 fill = mix(shape.rgb, vec3(1.0), rim) * (1.0 - 0.06 * max(0.0, n.y) * (1.0 - smoothstep(0.0, 6.0, -d)));

  float lifted = field(p - vec2(0.0, 3.0)).w;
  float shadow = (1.0 - smoothstep(-3.0, 9.0, lifted)) * uShadow * (1.0 - body);

  float hover = clamp(0.5 - sdPill(p, uHover) / aa, 0.0, 1.0) * uHoverColor.a;

  vec4 color = vec4(0.0, 0.0, 0.0, shadow);
  color = vec4(uHoverColor.rgb * hover, hover) + color * (1.0 - hover);
  color = vec4(fill * body, body) + color * (1.0 - body);
  outColor = color;
}`;

  const gnTravelEase = (t) => {
    const x = clamp01(t);
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  };

  const gnBezier = (a, b, c, d, t) => {
    const u = 1 - t;
    return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d;
  };

  const gnPillPath = (x, y, w, h) => {
    if (w <= 0.5 || h <= 0.5) return "";
    const r = Math.min(w, h) / 2;
    return `M${x + r},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + h - r}A${r},${r} 0 0 1 ${x + w - r},${y + h}H${x + r}A${r},${r} 0 0 1 ${x},${y + h - r}V${y + r}A${r},${r} 0 0 1 ${x + r},${y}Z`;
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

  let gooeyController = null;

  function initGooeyNav() {
    const stage = document.getElementById("gooey-nav-stage");
    const canvas = document.getElementById("gooey-nav-canvas");
    const list = document.getElementById("gooey-nav-list");
    const overlay = document.getElementById("gooey-nav-overlay");
    const fallback = document.getElementById("gooey-nav-fallback");
    if (!stage || !canvas || !list || !overlay || !fallback) return;

    const gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
    });

    let pipeline = null;
    if (gl) {
      const vs = compileShader(gl, gl.VERTEX_SHADER, GN_VERTEX);
      const fs = compileShader(gl, gl.FRAGMENT_SHADER, GN_FRAGMENT);
      const prog = gl.createProgram();
      if (vs && fs && prog) {
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (gl.getProgramParameter(prog, gl.LINK_STATUS)) {
          const uniformNames = [
            "uSize",
            "uRatio",
            "uPills",
            "uHover",
            "uDrops",
            "uTails",
            "uTints",
            "uCount",
            "uSmooth",
            "uPill",
            "uHoverColor",
            "uShadow",
          ];
          const uniforms = Object.fromEntries(
            uniformNames.map((n) => [n, gl.getUniformLocation(prog, n)])
          );
          const buffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
          gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
            gl.STATIC_DRAW
          );
          pipeline = {
            program: prog,
            uniforms,
            buffer,
            position: gl.getAttribLocation(prog, "aPosition"),
          };
        }
      }
    }

    fallback.style.display = pipeline ? "none" : "block";
    canvas.style.display = pipeline ? "block" : "none";

    /* Warm Solar Amber Pill + Multi-Spectral Droplet Tints (Coral, Gold, Mint, Rose) */
    const settings = {
      animationTime: 620,
      particleCount: 16,
      timeVariance: 280,
      spread: 54,
      gooeyness: 0.55,
      wobble: 0.52,
      hoverEffect: true,
      pillRgb: [1.0, 0.87, 0.68, 1.0], // Warm solar champagne #FFDEAD
      tints: [
        [1.0, 0.42, 0.29], // Sunset Coral #FF6B4A
        [1.0, 0.72, 0.0],  // Solar Gold #FFB800
        [0.0, 0.96, 0.83], // Electric Mint #00F5D4
        [1.0, 0.18, 0.58], // Rose Magenta #FF2E93
      ],
      hoverRgb: [1.0, 0.65, 0.35, 0.12],
      shadow: 0.45,
    };

    const view = { width: 0, height: 0 };
    let rects = [];
    let target = -1;
    const pills = new Map();
    const drops = [];
    const hover = { x: 0, y: 0, w: 0, h: 0, alpha: 0, index: -1 };
    const pillData = new Float32Array(GN_MAX_PILLS * 4);
    const dropData = new Float32Array(GN_MAX_DROPS * 4);
    const tailData = new Float32Array(GN_MAX_DROPS * 2);
    const tintData = new Float32Array(GN_MAX_DROPS * 3);
    let raf = 0;
    let last = 0;

    const measure = () => {
      const box = stage.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      view.width = box.width;
      view.height = box.height;
      const outerWidth = box.width + GN_MARGIN * 2;
      const outerHeight = box.height + GN_MARGIN * 2;
      Object.assign(canvas.style, {
        left: `${-GN_MARGIN}px`,
        top: `${-GN_MARGIN}px`,
        width: `${outerWidth}px`,
        height: `${outerHeight}px`,
      });
      const w = Math.max(1, Math.round(outerWidth * ratio));
      const h = Math.max(1, Math.round(outerHeight * ratio));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      rects = Array.from(list.children).map((item) => {
        const r = item.getBoundingClientRect();
        return { x: r.left - box.left, y: r.top - box.top, w: r.width, h: r.height };
      });
    };

    const trace = (boxes, blobs, smooth) => {
      const field = (x, y) => {
        let d = 1e5;
        const merge = (value) => {
          const h = clamp01(0.5 + (0.5 * (value - d)) / smooth);
          d = value + (d - value) * h - smooth * h * (1 - h);
        };
        for (const box of boxes) {
          const r = Math.min(box.hw, box.hh);
          const qx = Math.abs(x - box.cx) - box.hw + r;
          const qy = Math.abs(y - box.cy) - box.hh + r;
          merge(Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r);
        }
        for (const blob of blobs) {
          const px = x - blob.x;
          const py = y - blob.y;
          const bx = -blob.tx;
          const by = -blob.ty;
          const span = bx * bx + by * by;
          const h = span > 0.0001 ? clamp01((px * bx + py * by) / span) : 0;
          const value = Math.hypot(px - bx * h, py - by * h) - blob.r;
          if (value > d + smooth) continue;
          merge(value);
        }
        return d;
      };
      const step = 2;
      const parts = [];
      for (let y = 0; y < view.height; y += 1) {
        const cy = y + 0.5;
        let previous = field(0, cy);
        let start = previous < 0 ? 0 : null;
        for (let x = step; x <= view.width + step; x += step) {
          const value = field(Math.min(x, view.width), cy);
          if (previous < 0 !== value < 0) {
            const cross = x - step + (step * previous) / (previous - value);
            if (value < 0) start = cross;
            else if (start !== null) {
              parts.push(`M${start.toFixed(1)},${y}H${cross.toFixed(1)}V${y + 1}H${start.toFixed(1)}Z`);
              start = null;
            }
          }
          previous = value;
        }
        if (start !== null)
          parts.push(`M${start.toFixed(1)},${y}H${view.width}V${y + 1}H${start.toFixed(1)}Z`);
      }
      return parts.join("");
    };

    const getPill = (index) => {
      if (!pills.has(index)) pills.set(index, { mass: 0, scale: 0, velocity: 0, outgoing: 0 });
      return pills.get(index);
    };

    const shape = (index, amount) => {
      const rect = rects[index];
      if (!rect || amount <= 0.002) return null;
      const h = rect.h * Math.min(1.08, 0.42 + 0.58 * Math.sqrt(Math.max(0, amount)));
      const w = Math.max(h, rect.w * Math.pow(Math.max(0, amount), 0.75));
      return { x: rect.x + rect.w / 2 - w / 2, y: rect.y + rect.h / 2 - h / 2, w, h };
    };

    const plan = (drop, index, now, fresh) => {
      const rect = rects[index];
      if (!rect) return;
      const reach = Math.max(0, settings.spread);
      const ex = rect.x + rect.w / 2 + (Math.random() - 0.5) * rect.w * 0.45;
      const ey = rect.y + rect.h / 2 + (Math.random() - 0.5) * rect.h * 0.2;
      const base = Math.max(0.15, settings.animationTime / 1000);
      const variance = (Math.random() - 0.5) * (Math.max(0, settings.timeVariance) / 1000);
      const duration = Math.max(0.15, base * (0.62 + Math.random() * 0.32) + variance * 0.6);
      const out = Math.random() * TAU;
      const into = Math.random() * TAU;
      const burst = reach * (0.7 + Math.random() * 0.6);
      const swing = reach * (0.55 + Math.random() * 0.55);
      const carry = Math.min(duration / 3, 0.12);
      drop.index = index;
      drop.start = now;
      drop.duration = duration;
      drop.path = [
        drop.x,
        drop.y,
        fresh ? drop.x + Math.cos(out) * burst * 1.6 : drop.x + drop.vx * carry,
        fresh ? drop.y + Math.sin(out) * burst * 1.6 : drop.y + drop.vy * carry,
        ex + Math.cos(into) * swing * 1.5,
        ey + Math.sin(into) * swing * 1.5,
        ex,
        ey,
      ];
    };

    const emit = (from, to, now) => {
      const source = pills.get(from);
      const rect = rects[from];
      if (!source || !rect || !rects[to]) return;
      const available = source.mass - source.outgoing;
      if (available <= 0.002) return;
      const room = GN_MAX_DROPS - drops.length;
      const count = Math.min(room, Math.max(1, Math.round(settings.particleCount * available)));
      if (count <= 0) return;
      const win = Math.max(0.15, settings.animationTime / 1000) * 0.16;
      for (let i = 0; i < count; i++) {
        const mass = available / count;
        source.outgoing += mass;
        drops.push({
          x: rect.x + rect.w / 2 + (Math.random() - 0.5) * rect.w * 0.6,
          y: rect.y + rect.h / 2 + (Math.random() - 0.5) * rect.h * 0.3,
          vx: 0,
          vy: 0,
          mass,
          size: 0.85 + Math.random() * 0.42,
          given: 0,
          source: from,
          index: to,
          leave: now + Math.random() * win * 1000,
          emerged: 0,
          absorbed: 0,
          state: "waiting",
          tint: settings.tints[Math.floor(Math.random() * settings.tints.length)],
        });
      }
    };

    const render = () => {
      const height = rects[target]?.h ?? rects[0]?.h ?? 38;
      const smooth = height * (0.18 + 0.5 * clamp01(settings.gooeyness));
      const boxes = [];
      const blobs = [];
      for (const [index, state] of pills) {
        const box = shape(index, state.scale);
        if (box)
          boxes.push({
            cx: box.x + box.w / 2,
            cy: box.y + box.h / 2,
            hw: box.w / 2,
            hh: box.h / 2,
          });
      }
      for (const drop of drops) {
        if (drop.state === "waiting") continue;
        const radius = 0.26 * height * drop.size * Math.min(drop.emerged, 1 - drop.absorbed);
        if (radius <= 0.05) continue;
        const speed = Math.hypot(drop.vx, drop.vy);
        const len = Math.min(radius * 1.1, speed * 0.018);
        blobs.push({
          x: drop.x,
          y: drop.y,
          r: radius,
          tx: speed > 0.01 ? (drop.vx / speed) * len : 0,
          ty: speed > 0.01 ? (drop.vy / speed) * len : 0,
          tint: drop.tint,
        });
      }

      const settledPill = boxes.length === 1 && !blobs.length;
      if (settledPill) {
        const box = boxes[0];
        overlay.style.clipPath = `path('${gnPillPath(
          box.cx - box.hw,
          box.cy - box.hh,
          box.hw * 2,
          box.hh * 2
        )}')`;
      } else {
        overlay.style.clipPath = `path('${trace(boxes, blobs, smooth) || "M0,0Z"}')`;
      }
      overlay.style.visibility = "visible";

      if (!pipeline || !gl) {
        const box = shape(target, 1);
        if (box) {
          fallback.style.transform = `translate(${box.x.toFixed(2)}px, ${box.y.toFixed(2)}px)`;
          fallback.style.width = `${box.w}px`;
          fallback.style.height = `${box.h}px`;
        }
        fallback.style.opacity = box ? "1" : "0";
        return;
      }

      let slot = 0;
      for (const box of boxes) {
        if (slot >= GN_MAX_PILLS) break;
        pillData[slot * 4] = box.cx + GN_MARGIN;
        pillData[slot * 4 + 1] = box.cy + GN_MARGIN;
        pillData[slot * 4 + 2] = box.hw;
        pillData[slot * 4 + 3] = box.hh;
        slot += 1;
      }
      pillData.fill(0, slot * 4);
      let count = 0;
      for (const blob of blobs) {
        if (count >= GN_MAX_DROPS) break;
        dropData[count * 4] = blob.x + GN_MARGIN;
        dropData[count * 4 + 1] = blob.y + GN_MARGIN;
        dropData[count * 4 + 2] = blob.r;
        dropData[count * 4 + 3] = 0;
        tailData[count * 2] = blob.tx;
        tailData[count * 2 + 1] = blob.ty;
        const tint = blob.tint || settings.pillRgb;
        tintData[count * 3] = tint[0];
        tintData[count * 3 + 1] = tint[1];
        tintData[count * 3 + 2] = tint[2];
        count += 1;
      }

      const { program, uniforms, buffer, position } = pipeline;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      gl.uniform2f(uniforms.uSize, view.width + GN_MARGIN * 2, view.height + GN_MARGIN * 2);
      gl.uniform1f(uniforms.uRatio, canvas.width / Math.max(1, view.width + GN_MARGIN * 2));
      gl.uniform4fv(uniforms.uPills, pillData);
      gl.uniform4f(
        uniforms.uHover,
        hover.x + GN_MARGIN,
        hover.y + GN_MARGIN,
        hover.w / 2,
        hover.h / 2
      );
      gl.uniform4fv(uniforms.uDrops, dropData);
      gl.uniform2fv(uniforms.uTails, tailData);
      gl.uniform3fv(uniforms.uTints, tintData);
      gl.uniform1i(uniforms.uCount, count);
      gl.uniform1f(uniforms.uSmooth, smooth);
      gl.uniform3f(
        uniforms.uPill,
        settings.pillRgb[0],
        settings.pillRgb[1],
        settings.pillRgb[2]
      );
      gl.uniform4f(
        uniforms.uHoverColor,
        settings.hoverRgb[0],
        settings.hoverRgb[1],
        settings.hoverRgb[2],
        settings.hoverRgb[3] * hover.alpha
      );
      gl.uniform1f(uniforms.uShadow, settings.shadow);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    const frame = (now) => {
      raf = 0;
      const dt = Math.min(0.05, (now - last) / 1000 || 0);
      last = now;
      const emergeTime = 0.09;
      const absorbTime = 0.12;

      for (let i = drops.length - 1; i >= 0; i--) {
        const drop = drops[i];
        if (drop.state === "waiting") {
          if (now < drop.leave) continue;
          drop.state = "flying";
          plan(drop, drop.index, now, true);
        }
        if (drop.state === "flying") {
          drop.emerged = Math.min(1, drop.emerged + dt / emergeTime);
          const source = pills.get(drop.source);
          if (source && drop.given < drop.mass) {
            const step = Math.min(drop.mass * (dt / emergeTime), drop.mass - drop.given);
            source.mass = Math.max(0, source.mass - step);
            source.outgoing = Math.max(0, source.outgoing - step);
            drop.given += step;
          }
          const t = clamp01((now - drop.start) / 1000 / drop.duration);
          const eased = gnTravelEase(t);
          const [x0, y0, x1, y1, x2, y2, x3, y3] = drop.path;
          const x = gnBezier(x0, x1, x2, x3, eased);
          const y = gnBezier(y0, y1, y2, y3, eased);
          if (dt > 0) {
            drop.vx = (x - drop.x) / dt;
            drop.vy = (y - drop.y) / dt;
          }
          drop.x = x;
          drop.y = y;
          if (t >= 1) drop.state = "absorbing";
        }
        if (drop.state === "absorbing") {
          const step = Math.min(1 - drop.absorbed, dt / absorbTime);
          drop.absorbed += step;
          drop.vx *= 0.8;
          drop.vy *= 0.8;
          const host = getPill(drop.index);
          host.mass = Math.min(1, host.mass + drop.mass * step);
          if (drop.absorbed >= 1) drops.splice(i, 1);
        }
      }

      const zeta = reduceMotion() ? 1 : 1 - 0.72 * clamp01(settings.wobble);
      const omega = 26;
      const steps = Math.max(1, Math.ceil(dt / (1 / 240)));
      const h = dt / steps;
      for (const [index, state] of pills) {
        for (let n = 0; n < steps; n++) {
          state.velocity +=
            (omega * omega * (state.mass - state.scale) - 2 * zeta * omega * state.velocity) * h;
          state.scale += state.velocity * h;
        }
        if (
          index !== target &&
          state.mass <= 0.002 &&
          state.outgoing <= 0.002 &&
          Math.abs(state.scale) < 0.004
        ) {
          pills.delete(index);
        }
      }

      if (hover.index >= 0 && rects[hover.index]) {
        const rect = rects[hover.index];
        const k = 1 - Math.exp(-dt / 0.06);
        hover.x += (rect.x + rect.w / 2 - hover.x) * k;
        hover.y += (rect.y + rect.h / 2 - hover.y) * k;
        hover.w += (rect.w - hover.w) * k;
        hover.h += (rect.h - hover.h) * k;
      }
      const hoverGoal = settings.hoverEffect && hover.index >= 0 && hover.index !== target ? 1 : 0;
      hover.alpha += (hoverGoal - hover.alpha) * (1 - Math.exp(-dt / 0.08));

      render();

      const host = pills.get(target);
      const resting =
        drops.length === 0 &&
        pills.size === 1 &&
        host &&
        Math.abs(host.mass - host.scale) < 0.0008 &&
        Math.abs(host.velocity) < 0.004;
      const hoverRect = rects[hover.index];
      const hovering =
        Math.abs(hoverGoal - hover.alpha) > 0.003 ||
        (hoverRect &&
          Math.abs(hoverRect.x + hoverRect.w / 2 - hover.x) +
            Math.abs(hoverRect.w - hover.w) >
            0.1);
      if (!resting || hovering) raf = requestAnimationFrame(frame);
      else {
        host.mass = 1;
        host.scale = 1;
        host.velocity = 0;
        hover.alpha = hoverGoal;
        render();
      }
    };

    const wake = () => {
      if (raf) return;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };

    const settle = (index) => {
      pills.clear();
      drops.length = 0;
      target = index;
      const state = getPill(index);
      state.mass = 1;
      state.scale = 1;
      state.velocity = 0;
      render();
    };

    const go = (index, animate = true) => {
      if (index === target) return;
      if (!rects[index] || !animate || reduceMotion() || target < 0) {
        settle(index);
        return;
      }
      const now = performance.now();
      target = index;
      getPill(index);
      for (const drop of drops) {
        drop.index = index;
        if (drop.state === "flying") plan(drop, index, now, false);
        if (drop.state === "absorbing") {
          drop.state = "flying";
          drop.mass *= 1 - drop.absorbed;
          drop.absorbed = 0;
          plan(drop, index, now, false);
        }
      }
      for (const from of [...pills.keys()]) {
        if (from !== index) emit(from, index, now);
      }
      wake();
    };

    list.addEventListener("pointerover", (event) => {
      const item = event.target.closest("[data-gooey-index]");
      if (!item || !list.contains(item)) return;
      const index = Number(item.dataset.gooeyIndex);
      const rect = rects[index];
      if (!rect) return;
      if (hover.alpha < 0.01) {
        hover.x = rect.x + rect.w / 2;
        hover.y = rect.y + rect.h / 2;
        hover.w = rect.w;
        hover.h = rect.h;
      }
      hover.index = index;
      wake();
    });

    list.addEventListener("pointerleave", () => {
      hover.index = -1;
      wake();
    });

    const ro = new ResizeObserver(() => {
      measure();
      if (!drops.length) settle(target < 0 ? 0 : target);
      else render();
    });
    ro.observe(stage);
    ro.observe(list);

    measure();
    settle(0);
    document.fonts?.ready.then(() => {
      measure();
      if (!drops.length) settle(target < 0 ? 0 : target);
    });

    gooeyController = { go };
  }

  /* =========================================================================
   * 2. REACT BITS: PRISM (From Prism-JS-CSS.json) + WARM PERLIN WAVES
   *    WebGL2 3D raymarched spectral prism in Coral, Gold, Mint & Rose
   * ========================================================================= */
  const PRISM_PALETTE_SIZE = 64;
  const PRISM_VERTEX = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

  const PRISM_FRAGMENT = `#version 300 es
precision highp float;
uniform vec2 uResolution;
uniform mat3 uRay;
uniform float uGlow;
uniform vec2 uOffset;
uniform float uColorFreq;
uniform float uCenterShift;
uniform float uInvBaseHalf;
uniform float uInvHeight;
uniform float uMinAxis;
uniform float uPxScale;
uniform sampler2D uPalette;
uniform int uUsePalette;
uniform float uTime;
uniform float uNoise;
uniform float uSaturation;
out vec4 outColor;

const vec4 PHASE_COS = vec4(1.0, 0.5403023058681398, -0.4161468365471424, -0.9899924966004454);
const vec4 PHASE_SIN = vec4(0.0, 0.8414709848078965, 0.9092974268256817, 0.1411200080598672);

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453123);
}

float pyramid(vec3 p) {
  vec3 q = vec3(abs(p.x) * uInvBaseHalf, abs(p.y) * uInvHeight, abs(p.z) * uInvBaseHalf);
  return max((q.x + q.y + q.z - 1.0) * uMinAxis * 0.5773502691896258, -p.y);
}

vec3 grade(vec3 col, vec2 seed) {
  col += (rand(seed) - 0.5) * uNoise;
  col = clamp(col, 0.0, 1.0);
  float L = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = clamp(mix(vec3(L), col, uSaturation), 0.0, 1.0);
  return col;
}

void main() {
  vec2 f = (gl_FragCoord.xy - 0.5 * uResolution - uOffset) * uPxScale;
  vec3 origin = uRay * vec3(f, 0.0);
  vec3 axis = uRay[2];
  float z = 5.0;
  vec4 o = vec4(0.0);
  for (int i = 0; i < 80; i++) {
    vec3 p = origin + axis * z;
    float d = 0.12 + 0.2 * abs(pyramid(p + vec3(0.0, uCenterShift, 0.0)));
    z -= d;
    float phase = (p.y + z) * uColorFreq;
    vec4 band = sin(phase) * PHASE_COS + cos(phase) * PHASE_SIN + 1.0;
    if (uUsePalette == 1) band.rgb = texture(uPalette, vec2(phase * 0.15915494309189535, 0.5)).rgb * 2.0;
    o += band / d;
  }
  vec4 raw = o * o * uGlow / 1e5;
  vec4 e = exp(2.0 * raw);
  vec4 mapped = (e - 1.0) / (e + 1.0);
  vec2 seed = gl_FragCoord.xy + vec2(uTime);
  outColor = vec4(grade(mapped.rgb, seed) * mapped.a * 0.65, mapped.a * 0.65);
}`;

  const prismRotation = (yaw, pitch, roll) => {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cx = Math.cos(pitch), sx = Math.sin(pitch);
    const cz = Math.cos(roll), sz = Math.sin(roll);
    return [
      cy * cz + sy * sx * sz, cx * sz, -sy * cz + cy * sx * sz,
      -cy * sz + sy * sx * cz, cx * cz, sy * sz + cy * sx * cz,
      sy * cx, -sx, cy * cx,
    ];
  };

  const prismMultiply = (a, b) => {
    const out = new Array(9);
    for (let c = 0; c < 3; c++) {
      for (let r = 0; r < 3; r++) {
        out[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
      }
    }
    return out;
  };

  function initPrismBackdrop() {
    const container = document.getElementById("reactbits-prism");
    if (!container) return;
    const canvas = container.querySelector(".prism-canvas");
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      depth: false,
      stencil: false,
    });
    if (!gl) return;

    const vs = compileShader(gl, gl.VERTEX_SHADER, PRISM_VERTEX);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, PRISM_FRAGMENT);
    const prog = gl.createProgram();
    if (!vs || !fs || !prog) return;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;

    const u = {};
    const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(prog, i);
      if (info) u[info.name] = gl.getUniformLocation(prog, info.name);
    }

    const vao = gl.createVertexArray();
    const paletteTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, paletteTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    /* Multi-Spectral Warm Palette: Coral, Solar Gold, Electric Mint, Rose, Warm Amber */
    const stops = [
      [255, 107, 74],  // #FF6B4A Sunset Coral
      [255, 184, 0],   // #FFB800 Solar Gold
      [0, 245, 212],   // #00F5D4 Electric Mint
      [255, 46, 147],  // #FF2E93 Rose Magenta
      [255, 209, 102], // #FFD166 Warm Amber
    ];
    const data = new Uint8Array(PRISM_PALETTE_SIZE * 4);
    for (let i = 0; i < PRISM_PALETTE_SIZE; i++) {
      const position = (i / PRISM_PALETTE_SIZE) * stops.length;
      const index = Math.floor(position);
      const a = stops[index % stops.length];
      const b = stops[(index + 1) % stops.length];
      const t = position - index;
      const k = t * t * (3 - 2 * t);
      for (let c = 0; c < 3; c++) data[i * 4 + c] = Math.round(a[c] + (b[c] - a[c]) * k);
      data[i * 4 + 3] = 255;
    }
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      PRISM_PALETTE_SIZE,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      data
    );

    const view = { width: 1, height: 1, scale: 0.45 };
    const pose = { yaw: 0, pitch: 0, roll: 0 };
    const pointer = { x: 0, y: 0, inside: false };
    let time = 0;
    let last = 0;

    const resize = () => {
      view.width = Math.max(1, container.clientWidth);
      view.height = Math.max(1, container.clientHeight);
      canvas.width = Math.max(1, Math.round(view.width * view.scale));
      canvas.height = Math.max(1, Math.round(view.height * view.scale));
    };

    window.addEventListener(
      "pointermove",
      (e) => {
        const rect = container.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
        const y = ((e.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1;
        pointer.inside = Math.abs(x) <= 1 && Math.abs(y) <= 1;
        pointer.x = clamp(x, -1, 1);
        pointer.y = clamp(y, -1, 1);
      },
      { passive: true }
    );

    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();

    const frame = (now) => {
      const dt = last ? clamp((now - last) / 1000, 0.001, 0.05) : 1 / 60;
      last = now;
      if (!reduceMotion()) time += dt;

      const yaw = pointer.inside ? -pointer.x * 0.22 : 0;
      const pitch = pointer.inside ? pointer.y * 0.22 : 0;
      pose.yaw += (yaw - pose.yaw) * 0.06;
      pose.pitch += (pitch - pose.pitch) * 0.06;

      const t = time * 0.35;
      const tilt = prismRotation(pose.yaw, pose.pitch, pose.roll);
      const ray = prismMultiply(
        tilt,
        prismRotation(t * 0.55, Math.sin(t * 0.6 + 0.6) * 0.5, Math.sin(t * 0.35 + 1.9) * 0.4)
      );

      const height = 3.6;
      const half = 2.8;

      gl.bindVertexArray(vao);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(prog);
      gl.uniform2f(u.uResolution, canvas.width, canvas.height);
      gl.uniformMatrix3fv(u.uRay, false, ray);
      gl.uniform1f(u.uGlow, 0.92);
      gl.uniform2f(u.uOffset, 0, 60 * view.scale);
      gl.uniform1f(u.uColorFreq, 0.95);
      gl.uniform1f(u.uCenterShift, height * 0.25);
      gl.uniform1f(u.uInvBaseHalf, 1 / half);
      gl.uniform1f(u.uInvHeight, 1 / height);
      gl.uniform1f(u.uMinAxis, Math.min(half, height));
      gl.uniform1f(u.uPxScale, 1 / (canvas.height * 0.1 * 3.8));
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, paletteTexture);
      gl.uniform1i(u.uPalette, 0);
      gl.uniform1i(u.uUsePalette, 1);
      gl.uniform1f(u.uTime, time);
      gl.uniform1f(u.uNoise, 0.015);
      gl.uniform1f(u.uSaturation, 1.45);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      if (!reduceMotion()) {
        requestAnimationFrame(frame);
      }
    };

    requestAnimationFrame(frame);
  }

  /* =========================================================================
   * 3. REACT BITS: WAVES (Warm Coral / Gold / Mint Perlin-Noise Silk Lines)
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
    const canvas = document.getElementById("reactbits-waves-canvas");
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const noise = new PerlinNoise(77);
    const config = {
      xGap: 24,
      yGap: 36,
      waveSpeedX: 0.011,
      waveSpeedY: 0.005,
      waveAmpX: 28,
      waveAmpY: 14,
      friction: 0.91,
      tension: 0.006,
      maxCursorMove: 95,
      interactionRadius: 150,
    };

    let width = 0;
    let height = 0;
    let lines = [];
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

      lines = [];
      const oWidth = width + 160;
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

      for (let i = 0; i < lines.length; i++) {
        const pts = lines[i];
        for (let j = 0; j < pts.length; j++) {
          const p = pts[j];
          const move =
            noise.perlin2(
              (p.x + t * config.waveSpeedX) * 0.002,
              (p.y + t * config.waveSpeedY) * 0.0015
            ) * 12;
          p.wave.x = Math.cos(move) * config.waveAmpX;
          p.wave.y = Math.sin(move) * config.waveAmpY;

          const cdx = p.x - mouse.sx;
          const cdy = p.y - mouse.sy;
          const dist = Math.hypot(cdx, cdy);
          const l = Math.max(config.interactionRadius, mouse.vs);
          if (dist < l) {
            const s = 1 - dist / l;
            const f = Math.cos(dist * 0.001) * s;
            p.cursor.vx += Math.cos(mouse.a) * f * l * mouse.vs * 0.00055;
            p.cursor.vy += Math.sin(mouse.a) * f * l * mouse.vs * 0.00055;
          }

          p.cursor.vx += (0 - p.cursor.x) * config.tension;
          p.cursor.vy += (0 - p.cursor.y) * config.tension;
          p.cursor.vx *= config.friction;
          p.cursor.vy *= config.friction;
          p.cursor.x = clamp(p.cursor.x + p.cursor.vx * 2, -config.maxCursorMove, config.maxCursorMove);
          p.cursor.y = clamp(p.cursor.y + p.cursor.vy * 2, -config.maxCursorMove, config.maxCursorMove);
        }
      }

      ctx.clearRect(0, 0, width, height);
      const grad = ctx.createLinearGradient(0, 0, width, height);
      grad.addColorStop(0, "rgba(255, 107, 74, 0.14)");
      grad.addColorStop(0.5, "rgba(255, 184, 0, 0.12)");
      grad.addColorStop(1, "rgba(0, 245, 212, 0.13)");

      ctx.beginPath();
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.0;

      for (let i = 0; i < lines.length; i++) {
        const points = lines[i];
        if (!points.length) continue;
        ctx.moveTo(points[0].x + points[0].wave.x, points[0].y + points[0].wave.y);
        for (let idx = 0; idx < points.length; idx++) {
          const pt = points[idx];
          ctx.lineTo(pt.x + pt.wave.x + pt.cursor.x, pt.y + pt.wave.y + pt.cursor.y);
        }
      }
      ctx.stroke();

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

    window.addEventListener("resize", setSize);
    setSize();
    requestAnimationFrame(tick);
  }

  /* =========================================================================
   * 4. REACT BITS: PIXEL CARD (Coral, Mint, and Gold Multi-Spectral Variants)
   * ========================================================================= */
  const PIXEL_VARIANTS = {
    coral: {
      gap: 6,
      speed: 48,
      colors: "#fff1f2,#fda4af,#ff6b4a,#f97316,#ffb800",
    },
    mint: {
      gap: 6,
      speed: 48,
      colors: "#ecfdf5,#6ee7b7,#00f5d4,#10b981,#14b8a6",
    },
    gold: {
      gap: 6,
      speed: 48,
      colors: "#fefce8,#fde047,#ffb800,#f59e0b,#ff2e93",
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
      this.maxSizeInteger = 2.5;
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
      this.ctx.globalAlpha = Math.min(
        1,
        0.38 + (this.size / this.maxSizeInteger) * 0.52 + this.boost * 0.25
      );
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

      const variantName = card.dataset.pixelVariant || "coral";
      const cfg = PIXEL_VARIANTS[variantName] || PIXEL_VARIANTS.coral;
      const colorsArr = cfg.colors.split(",");
      const effSpeed = reduced ? 0 : cfg.speed * 0.001;

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
            const delay = reduced ? 0 : dist * 0.6;
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
   * 5. REACT BITS: MAGNET (From Magnet-JS-CSS.json)
   *    Spring-physics magnetic cursor pull on buttons & chips
   * ========================================================================= */
  function initMagnet(root = document) {
    const els = root.querySelectorAll("[data-rb-magnet], .sample-chip");
    els.forEach((el) => {
      if (el.dataset.rbMagnetBound === "1") return;
      el.dataset.rbMagnetBound = "1";

      el.addEventListener("pointermove", (e) => {
        if (reduceMotion()) return;
        const rect = el.getBoundingClientRect();
        const dx = e.clientX - (rect.left + rect.width / 2);
        const dy = e.clientY - (rect.top + rect.height / 2);
        el.style.transform = `translate3d(${(dx * 0.16).toFixed(2)}px, ${(dy * 0.22).toFixed(2)}px, 0)`;
      });

      el.addEventListener("pointerleave", () => {
        el.style.transform = "";
      });
    });
  }

  /* =========================================================================
   * 6. REACT BITS: GLARE HOVER (From GlareHover-JS-CSS.json)
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
   * 7. REACT BITS: GRADIENT TEXT (From GradientText-JS-CSS.json)
   * ========================================================================= */
  function initGradientText(root = document) {
    const els = root.querySelectorAll("[data-rb-gradient-text]");
    els.forEach((el) => {
      if (el.dataset.rbGradBound === "1") return;
      el.dataset.rbGradBound = "1";

      const text = el.textContent;
      const colors = (
        el.dataset.colors || "#FF6B4A,#FFB800,#00F5D4,#FF2E93,#FFD166"
      ).split(",");
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
   * 8. REACT BITS: SHINY TEXT & BLUR TEXT
   * ========================================================================= */
  function initShinyText(root = document) {
    root.querySelectorAll("[data-rb-shiny]").forEach((el) => {
      if (el.dataset.rbShinyBound === "1") return;
      el.dataset.rbShinyBound = "1";
      el.classList.add("rb-shiny-text");
    });
  }

  function initBlurText(root = document) {
    root.querySelectorAll("[data-rb-blur-text]").forEach((el) => {
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
   * 9. REACT BITS: STAR BORDER (Warm Gold / Coral / Mint Orbiting Trails)
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
    return match ? [+match[1], +match[2], +match[3]] : [255, 184, 0];
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

  function attachStarBorder(el, color = "#ffb800", speedSeconds = 6.0) {
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

            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${fade * 0.72})`;
            ctx.lineWidth = 1.6 * Math.pow(1 - u0, 0.45);
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          }

          const head = pointOnRoundedRect(hDist, width, height, radius);
          const bloom = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, 12);
          bloom.addColorStop(0, `rgba(${hr}, ${hg}, ${hb}, 0.92)`);
          bloom.addColorStop(0.45, `rgba(${r}, ${g}, ${b}, 0.38)`);
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
      let color = "#ffb800";
      if (btn.classList.contains("btn-mint")) color = "#00f5d4";
      else if (btn.classList.contains("btn-coral")) color = "#ff6b4a";
      attachStarBorder(btn, color, 6.0);
    });
  }

  /* =========================================================================
   * 10. REACT BITS: CLICK SPARK (Multi-Spectral Coral, Gold, Mint & Rose)
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
      [255, 107, 74],  // Sunset Coral
      [255, 184, 0],   // Solar Gold
      [0, 245, 212],   // Electric Mint
      [255, 46, 147],  // Rose Magenta
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
        ctx.lineWidth = 1.9 * (1 - t * 0.4);
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
        const count = 10;
        for (let i = 0; i < count; i++) {
          const base = (TAU * i) / count;
          const rgb = colors[(i + Math.floor(Math.random() * 2)) % colors.length];
          sparks.push({
            x: e.clientX,
            y: e.clientY,
            angle: base + (Math.random() - 0.5) * 0.35,
            radius: 28 * (0.8 + Math.random() * 0.45),
            size: 12 * (0.8 + Math.random() * 0.4),
            duration: 440 * (0.85 + Math.random() * 0.3),
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
   * 11. REACT BITS: COUNT UP (Animated numeric counter)
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
    initGooeyNav();
    initPrismBackdrop();
    initWavesBackdrop();
    initPixelCards();
    initMagnet(document);
    initGlareHover(document);
    initGradientText(document);
    initShinyText(document);
    initBlurText(document);
    initStarBorders();
    initClickSpark();
  }

  window.ReactBitsFX = {
    syncGooeyIndex(index) {
      gooeyController?.go?.(index, true);
    },
    refreshDynamicCards(root = document) {
      initGlareHover(root);
      initPixelCards();
      initMagnet(root);
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
