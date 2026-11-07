// Light relief: the restored photo draped over the engine's heightmap as an orbitable WebGL terrain.
const GRID = 128;

const VERTEX = `
attribute vec2 a_plane;
attribute float a_height;
attribute vec3 a_normal;
attribute vec2 a_uv;
uniform mat4 u_matrix;
uniform float u_lift;
varying vec2 v_uv;
varying float v_height;
varying vec3 v_normal;
varying float v_depth;
void main() {
  vec3 position = vec3(a_plane.x, a_height * u_lift, a_plane.y);
  gl_Position = u_matrix * vec4(position, 1.0);
  v_uv = a_uv;
  v_height = a_height;
  v_normal = a_normal;
  v_depth = gl_Position.z;
}`;

const FRAGMENT = `
precision highp float;
uniform sampler2D u_photo;
uniform float u_lift;
varying vec2 v_uv;
varying float v_height;
varying vec3 v_normal;
varying float v_depth;
void main() {
  vec3 photo = texture2D(u_photo, v_uv).rgb;
  vec3 light = normalize(vec3(-0.4, 0.9, 0.5));
  float diffuse = max(dot(normalize(v_normal), light), 0.0);
  vec3 color = photo * (0.55 + 0.6 * diffuse);
  float band = fract(v_height * 18.0);
  float contour = smoothstep(0.06, 0.0, band) + smoothstep(0.94, 1.0, band);
  vec3 warm = vec3(1.0, 0.86, 0.62);
  color += warm * contour * (0.12 + v_height * 0.9) * u_lift;
  color += warm * pow(v_height, 2.2) * 0.55 * u_lift;
  float fog = smoothstep(2.2, 4.4, v_depth);
  gl_FragColor = vec4(mix(color, vec3(0.024, 0.027, 0.035), fog), 1.0);
}`;

export function createRelief(canvas, photoUrl, heightUrl, { reducedMotion = false } = {}) {
  let disposed = false;
  let frame = 0;
  let visible = true;
  const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
  if (!gl) return { dispose() {}, supported: false };

  const view = { yaw: -0.55, pitch: 0.62, distance: 2.45, targetYaw: -0.55, targetPitch: 0.62 };
  let lift = reducedMotion ? 1 : 0;
  let liftStart = 0;
  let lastInteraction = 0;
  let drag = null;
  let indexCount = 0;
  let aspect = 1;

  const program = link(gl, VERTEX, FRAGMENT);
  const location = (name) => gl.getAttribLocation(program, name);
  const uniform = (name) => gl.getUniformLocation(program, name);

  Promise.all([loadImage(photoUrl), loadImage(heightUrl)]).then(([photo, heightImage]) => {
    if (disposed) return;
    aspect = photo.naturalWidth / photo.naturalHeight || 1;
    buildMesh(heightImage);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, photo);
    liftStart = performance.now();
    frame = requestAnimationFrame(render);
  }).catch(() => { /* The card keeps its caption; the canvas stays dark. */ });

  // Heights come from a mostly dark illumination map, so a square root lifts mid-tones into visible terrain.
  function buildMesh(heightImage) {
    const scratch = document.createElement('canvas');
    scratch.width = GRID;
    scratch.height = GRID;
    const context = scratch.getContext('2d', { willReadFrequently: true });
    context.drawImage(heightImage, 0, 0, GRID, GRID);
    const pixels = context.getImageData(0, 0, GRID, GRID).data;
    const raw = new Float32Array(GRID * GRID);
    let max = 0;
    for (let i = 0; i < raw.length; i += 1) { raw[i] = Math.sqrt(pixels[i * 4] / 255); max = Math.max(max, raw[i]); }
    const heights = new Float32Array(raw.length);
    for (let y = 0; y < GRID; y += 1) {
      for (let x = 0; x < GRID; x += 1) {
        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const sx = Math.min(GRID - 1, Math.max(0, x + dx));
            const sy = Math.min(GRID - 1, Math.max(0, y + dy));
            sum += raw[sy * GRID + sx];
            count += 1;
          }
        }
        heights[y * GRID + x] = max > 0 ? (sum / count) / max : 0;
      }
    }
    const width = 2;
    const depth = 2 / aspect;
    const amplitude = 0.42;
    const plane = new Float32Array(GRID * GRID * 2);
    const uvs = new Float32Array(GRID * GRID * 2);
    const normals = new Float32Array(GRID * GRID * 3);
    const at = (x, y) => heights[Math.min(GRID - 1, Math.max(0, y)) * GRID + Math.min(GRID - 1, Math.max(0, x))] * amplitude;
    for (let y = 0; y < GRID; y += 1) {
      for (let x = 0; x < GRID; x += 1) {
        const i = y * GRID + x;
        plane[i * 2] = (x / (GRID - 1) - 0.5) * width;
        plane[i * 2 + 1] = (y / (GRID - 1) - 0.5) * depth;
        uvs[i * 2] = x / (GRID - 1);
        uvs[i * 2 + 1] = y / (GRID - 1);
        const nx = (at(x - 1, y) - at(x + 1, y)) / (2 * width / (GRID - 1));
        const nz = (at(x, y - 1) - at(x, y + 1)) / (2 * depth / (GRID - 1));
        const length = Math.hypot(nx, 1, nz);
        normals[i * 3] = nx / length;
        normals[i * 3 + 1] = 1 / length;
        normals[i * 3 + 2] = nz / length;
      }
    }
    const scaled = heights.map((value) => value * amplitude);
    const indices = new Uint16Array((GRID - 1) * (GRID - 1) * 6);
    let n = 0;
    for (let y = 0; y < GRID - 1; y += 1) {
      for (let x = 0; x < GRID - 1; x += 1) {
        const a = y * GRID + x;
        indices.set([a, a + GRID, a + 1, a + 1, a + GRID, a + GRID + 1], n);
        n += 6;
      }
    }
    indexCount = indices.length;
    gl.useProgram(program);
    bindAttribute('a_plane', plane, 2);
    bindAttribute('a_height', scaled, 1);
    bindAttribute('a_normal', normals, 3);
    bindAttribute('a_uv', uvs, 2);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  }

  function bindAttribute(name, data, size) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const slot = location(name);
    gl.enableVertexAttribArray(slot);
    gl.vertexAttribPointer(slot, size, gl.FLOAT, false, 0, 0);
  }

  function render(now) {
    if (disposed) return;
    frame = requestAnimationFrame(render);
    if (!visible || !indexCount) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    if (!reducedMotion) {
      lift = easeOutCubic(Math.min(1, (now - liftStart) / 2200));
      if (!drag && now - lastInteraction > 2500) view.targetYaw += 0.0016;
    }
    view.yaw += (view.targetYaw - view.yaw) * 0.08;
    view.pitch += (view.targetPitch - view.pitch) * 0.08;
    const eye = [
      Math.sin(view.yaw) * Math.cos(view.pitch) * view.distance,
      Math.sin(view.pitch) * view.distance,
      Math.cos(view.yaw) * Math.cos(view.pitch) * view.distance,
    ];
    const matrix = multiply(perspective(0.72, width / Math.max(1, height), 0.05, 20), lookAt(eye, [0, 0.08, 0], [0, 1, 0]));
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.024, 0.027, 0.035, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    gl.uniformMatrix4fv(uniform('u_matrix'), false, matrix);
    gl.uniform1f(uniform('u_lift'), lift);
    gl.uniform1i(uniform('u_photo'), 0);
    gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
  }

  const onDown = (event) => {
    drag = { x: event.clientX, y: event.clientY, yaw: view.targetYaw, pitch: view.targetPitch };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('is-dragging');
  };
  const onMove = (event) => {
    if (!drag) return;
    lastInteraction = performance.now();
    view.targetYaw = drag.yaw - (event.clientX - drag.x) * 0.006;
    view.targetPitch = Math.min(1.35, Math.max(0.18, drag.pitch + (event.clientY - drag.y) * 0.005));
  };
  const onUp = () => { drag = null; lastInteraction = performance.now(); canvas.classList.remove('is-dragging'); };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; });
  observer.observe(canvas);

  return {
    supported: true,
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Relief texture failed to load.'));
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

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const range = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (near + far) * range, -1, 0, 0, near * far * range * 2, 0];
}

function lookAt(eye, target, up) {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let k = 0; k < 4; k += 1) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return out;
}

function normalize(v) { const length = Math.hypot(...v) || 1; return v.map((value) => value / length); }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
