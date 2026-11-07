// Landing scene: the dark photo as a WebGL surface. The cursor's light restores colour and lifts the
// scene into terrain from the engine's heightmap; a click raises the whole night and tilts the camera.
const GRID_X = 192;

const VERTEX = `
attribute vec2 a_uv;
attribute float a_height;
attribute vec3 a_normal;
uniform mat4 u_matrix;
uniform vec2 u_mouse;
uniform float u_radius;
uniform float u_aspect;
uniform float u_flood;
uniform float u_amp;
varying vec2 v_uv;
varying float v_height;
varying float v_light;
varying vec3 v_normal;
varying float v_depth;
float lightAt(vec2 uv) {
  vec2 d = (uv - u_mouse) * vec2(u_aspect, 1.0);
  return smoothstep(u_radius, u_radius * 0.15, length(d));
}
void main() {
  float local = lightAt(a_uv);
  float lift = max(local * 0.75, u_flood);
  vec3 position = vec3((a_uv.x - 0.5) * u_aspect, a_height * u_amp * lift, a_uv.y - 0.5);
  gl_Position = u_matrix * vec4(position, 1.0);
  v_uv = a_uv;
  v_height = a_height;
  v_light = local;
  v_normal = a_normal;
  v_depth = gl_Position.z;
}`;

const FRAGMENT = `
precision highp float;
uniform sampler2D u_dark;
uniform sampler2D u_lit;
uniform vec2 u_mouse;
uniform float u_radius;
uniform float u_aspect;
uniform float u_flood;
varying vec2 v_uv;
varying float v_height;
varying float v_light;
varying vec3 v_normal;
varying float v_depth;
void main() {
  vec2 d = (v_uv - u_mouse) * vec2(u_aspect, 1.0);
  float dist = length(d);
  float local = smoothstep(u_radius, u_radius * 0.2, dist);
  float reveal = max(local, u_flood);
  vec3 dark = texture2D(u_dark, v_uv).rgb;
  vec3 lit = texture2D(u_lit, v_uv).rgb;
  vec3 color = mix(dark, lit, reveal);
  float diffuse = max(dot(normalize(v_normal), normalize(vec3(-0.35, 0.9, 0.45))), 0.0);
  color *= mix(1.0, 0.72 + 0.42 * diffuse, u_flood);
  vec3 warm = vec3(1.0, 0.86, 0.62);
  float band = fract(v_height * 14.0);
  float contour = smoothstep(0.05, 0.0, band) + smoothstep(0.95, 1.0, band);
  color += warm * contour * reveal * (0.08 + v_height * 0.55);
  color += warm * 0.10 * smoothstep(u_radius * 1.25, 0.0, dist) * (1.0 - u_flood);
  float ring = smoothstep(0.012, 0.0, abs(dist - u_radius * 0.92)) * (1.0 - u_flood);
  color += warm * ring * 0.22;
  float fog = smoothstep(2.6, 5.0, v_depth) * u_flood;
  gl_FragColor = vec4(mix(color, vec3(0.024, 0.027, 0.035), fog), 1.0);
}`;

export function createHeroScene(canvas, { darkUrl, litUrl, heightUrl, reducedMotion = false, onReady } = {}) {
  const gl = canvas.getContext('webgl', { antialias: true, alpha: false, premultipliedAlpha: false });
  if (!gl) return null;
  const program = link(gl, VERTEX, FRAGMENT);
  const uniform = (name) => gl.getUniformLocation(program, name);
  const state = {
    mouse: [0.5, 0.45], target: [0.5, 0.45], radius: 0, flood: 0, floodTarget: 0,
    yaw: 0, pitch: 1.32, distance: 1, targetYaw: 0, lastMove: -Infinity, drag: null,
    aspect: 1.5, indexCount: 0, visible: true, disposed: false, frame: 0, start: performance.now(),
  };

  Promise.all([loadImage(darkUrl), loadImage(litUrl), loadImage(heightUrl)]).then(([dark, lit, height]) => {
    if (state.disposed) return;
    state.aspect = dark.naturalWidth / dark.naturalHeight;
    buildMesh(height);
    bindTexture(0, dark);
    bindTexture(1, lit);
    onReady?.();
    state.frame = requestAnimationFrame(render);
  }).catch((error) => console.warn('Hero scene unavailable:', error));

  function bindTexture(unit, image) {
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
  }

  function buildMesh(heightImage) {
    const columns = GRID_X;
    const rows = Math.max(2, Math.round(GRID_X / state.aspect));
    const scratch = document.createElement('canvas');
    scratch.width = columns;
    scratch.height = rows;
    const context = scratch.getContext('2d', { willReadFrequently: true });
    context.drawImage(heightImage, 0, 0, columns, rows);
    const pixels = context.getImageData(0, 0, columns, rows).data;
    const raw = new Float32Array(columns * rows);
    let max = 0;
    for (let i = 0; i < raw.length; i += 1) { raw[i] = Math.sqrt(pixels[i * 4] / 255); max = Math.max(max, raw[i]); }
    const heights = new Float32Array(raw.length);
    const at = (x, y) => raw[Math.min(rows - 1, Math.max(0, y)) * columns + Math.min(columns - 1, Math.max(0, x))];
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < columns; x += 1) {
        let sum = 0;
        for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) sum += at(x + dx, y + dy);
        heights[y * columns + x] = max > 0 ? sum / 9 / max : 0;
      }
    }
    const uvs = new Float32Array(columns * rows * 2);
    const normals = new Float32Array(columns * rows * 3);
    const h = (x, y) => heights[Math.min(rows - 1, Math.max(0, y)) * columns + Math.min(columns - 1, Math.max(0, x))] * 0.3;
    const stepX = state.aspect / (columns - 1);
    const stepZ = 1 / (rows - 1);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < columns; x += 1) {
        const i = y * columns + x;
        uvs[i * 2] = x / (columns - 1);
        uvs[i * 2 + 1] = y / (rows - 1);
        const nx = (h(x - 1, y) - h(x + 1, y)) / (2 * stepX);
        const nz = (h(x, y - 1) - h(x, y + 1)) / (2 * stepZ);
        const length = Math.hypot(nx, 1, nz);
        normals.set([nx / length, 1 / length, nz / length], i * 3);
      }
    }
    const indices = new Uint16Array((columns - 1) * (rows - 1) * 6);
    let n = 0;
    for (let y = 0; y < rows - 1; y += 1) {
      for (let x = 0; x < columns - 1; x += 1) {
        const a = y * columns + x;
        indices.set([a, a + columns, a + 1, a + 1, a + columns, a + columns + 1], n);
        n += 6;
      }
    }
    state.indexCount = indices.length;
    gl.useProgram(program);
    attribute('a_uv', uvs, 2);
    attribute('a_height', heights, 1);
    attribute('a_normal', normals, 3);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  }

  function attribute(name, data, size) {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const slot = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(slot);
    gl.vertexAttribPointer(slot, size, gl.FLOAT, false, 0, 0);
  }

  // Distance at which a near-top-down camera covers the viewport with the photo plane.
  function coverDistance(viewAspect, fov) {
    const half = Math.tan(fov / 2);
    return Math.min(0.5 / half, (state.aspect / 2) / (half * viewAspect)) * 0.9;
  }

  function camera(viewAspect) {
    const fov = 0.62;
    const flat = coverDistance(viewAspect, fov);
    const f = easeInOut(state.flood);
    const pitch = lerp(1.38, 0.66, f);
    const distance = lerp(flat, flat * 1.55 + 0.25, f);
    const yaw = state.yaw;
    const target = [0, lerp(0, 0.04, f), lerp(0, 0.04, f)];
    const eye = [
      target[0] + Math.sin(yaw) * Math.cos(pitch) * distance,
      target[1] + Math.sin(pitch) * distance,
      target[2] + Math.cos(yaw) * Math.cos(pitch) * distance,
    ];
    return multiply(perspective(fov, viewAspect, 0.02, 20), lookAt(eye, target, [0, 1, 0]));
  }

  function render(now) {
    if (state.disposed) return;
    state.frame = requestAnimationFrame(render);
    if (!state.visible || !state.indexCount) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 1.75);
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const viewAspect = width / Math.max(1, height);

    const idle = now - state.lastMove > 2600;
    if (idle && state.floodTarget === 0 && !reducedMotion) {
      const t = (now - state.start) / 1000;
      state.target = [0.5 + Math.sin(t * 0.33) * 0.26, 0.46 + Math.sin(t * 0.66) * 0.16];
    }
    const ease = reducedMotion ? 1 : (idle ? 0.03 : 0.12);
    state.mouse[0] += (state.target[0] - state.mouse[0]) * ease;
    state.mouse[1] += (state.target[1] - state.mouse[1]) * ease;
    state.radius += ((viewAspect < 1 ? 0.32 : 0.22) - state.radius) * 0.06;
    state.flood += (state.floodTarget - state.flood) * (reducedMotion ? 1 : 0.035);
    if (state.floodTarget === 1 && !state.drag && !reducedMotion && now - state.lastMove > 3000) state.targetYaw += 0.0012;
    if (state.floodTarget === 0) state.targetYaw = (state.mouse[0] - 0.5) * 0.12;
    state.yaw += (state.targetYaw - state.yaw) * 0.06;

    const matrix = camera(viewAspect);
    state.matrix = matrix;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.024, 0.027, 0.035, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    gl.uniformMatrix4fv(uniform('u_matrix'), false, matrix);
    gl.uniform2f(uniform('u_mouse'), state.mouse[0], state.mouse[1]);
    gl.uniform1f(uniform('u_radius'), state.radius);
    gl.uniform1f(uniform('u_aspect'), state.aspect);
    gl.uniform1f(uniform('u_flood'), state.flood);
    gl.uniform1f(uniform('u_amp'), 0.3);
    gl.uniform1i(uniform('u_dark'), 0);
    gl.uniform1i(uniform('u_lit'), 1);
    gl.drawElements(gl.TRIANGLES, state.indexCount, gl.UNSIGNED_SHORT, 0);
  }

  // Projects the pointer onto the ground plane so the light lands where the cursor is.
  function pointerToUv(event) {
    if (!state.matrix) return null;
    const rect = canvas.getBoundingClientRect();
    const ndc = [((event.clientX - rect.left) / rect.width) * 2 - 1, 1 - ((event.clientY - rect.top) / rect.height) * 2];
    const inverse = invert(state.matrix);
    if (!inverse) return null;
    const near = transform(inverse, [ndc[0], ndc[1], -1, 1]);
    const far = transform(inverse, [ndc[0], ndc[1], 1, 1]);
    const t = near[1] / (near[1] - far[1]);
    const x = near[0] + (far[0] - near[0]) * t;
    const z = near[2] + (far[2] - near[2]) * t;
    return [x / state.aspect + 0.5, z + 0.5];
  }

  const onMove = (event) => {
    if (state.drag) {
      state.targetYaw = state.drag.yaw - (event.clientX - state.drag.x) * 0.005;
      state.lastMove = performance.now();
      return;
    }
    if (event.pointerType === 'touch') return;
    const uv = pointerToUv(event);
    if (!uv) return;
    state.target = uv;
    state.lastMove = performance.now();
  };
  const onDown = (event) => {
    if (state.floodTarget !== 1 || event.pointerType === 'touch') return;
    state.drag = { x: event.clientX, yaw: state.targetYaw, moved: false };
  };
  const onUp = () => { state.drag = null; };
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerdown', onDown);
  window.addEventListener('pointerup', onUp);
  const observer = new IntersectionObserver(([entry]) => { state.visible = entry.isIntersecting; });
  observer.observe(canvas);

  return {
    get raised() { return state.floodTarget === 1; },
    setRaised(value) {
      state.floodTarget = value ? 1 : 0;
      if (!value) state.targetYaw = 0;
    },
    dispose() {
      state.disposed = true;
      cancelAnimationFrame(state.frame);
      observer.disconnect();
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load ${url}`));
    image.src = url;
  });
}

function link(gl, vertexSource, fragmentSource) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'Shader failed.');
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Program failed.');
  return program;
}

const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const range = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (near + far) * range, -1, 0, 0, near * far * range * 2, 0];
}

function lookAt(eye, target, up) {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let c = 0; c < 4; c += 1) for (let r = 0; r < 4; r += 1) for (let k = 0; k < 4; k += 1) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}

function transform(m, v) {
  const out = [0, 0, 0, 0];
  for (let r = 0; r < 4; r += 1) out[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
  return [out[0] / out[3], out[1] / out[3], out[2] / out[3]];
}

function invert(m) {
  const inv = new Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (!det) return null;
  return inv.map((value) => value / det);
}

function normalize(v) { const length = Math.hypot(...v) || 1; return v.map((value) => value / length); }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
