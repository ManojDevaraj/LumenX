export function createRelight(canvas, baseUrl, illuminationUrl, gammaInput, exposureInput, messageNode) {
  let baseImage = null;
  let lightImage = null;
  let disposed = false;
  let gl = null;
  let program = null;
  let baseTexture = null;
  let lightTexture = null;
  let buffer = null;
  let draw = () => {};
  const gammaOutput = document.querySelector('[data-role="gamma-value"]');
  const exposureOutput = document.querySelector('[data-role="exposure-value"]');

  const load = (url) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Relight texture failed to load.'));
    image.src = url;
  });

  Promise.all([load(baseUrl), load(illuminationUrl)]).then(([base, light]) => {
    if (disposed) return;
    baseImage = base;
    lightImage = light;
    gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!gl) {
      const context = canvas.getContext('2d');
      canvas.width = base.naturalWidth;
      canvas.height = base.naturalHeight;
      context?.drawImage(base, 0, 0);
      messageNode.textContent = 'WebGL is unavailable; showing the final image without a live light adjustment.';
      draw = () => context?.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
      return;
    }
    program = makeProgram(gl);
    const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    baseTexture = makeTexture(gl, base);
    lightTexture = makeTexture(gl, light);
    canvas.width = base.naturalWidth;
    canvas.height = base.naturalHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.uniform1i(gl.getUniformLocation(program, 'u_base'), 0);
    gl.uniform1i(gl.getUniformLocation(program, 'u_light'), 1);
    messageNode.textContent = 'Gamma and exposure update locally in the browser.';

    draw = () => {
      const gamma = clamp(Number(gammaInput.value), 0.4, 2.5);
      const exposure = clamp(Number(exposureInput.value), 0.25, 4.0);
      gammaOutput.value = gamma.toFixed(2);
      exposureOutput.value = `${exposure.toFixed(2)}×`;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, baseTexture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, lightTexture);
      gl.uniform1f(gl.getUniformLocation(program, 'u_gamma'), gamma);
      gl.uniform1f(gl.getUniformLocation(program, 'u_exposure'), exposure);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };
    draw();
  }).catch((error) => {
    messageNode.textContent = error.message;
  });

  const update = () => draw();
  gammaInput.addEventListener('input', update);
  exposureInput.addEventListener('input', update);
  return () => {
    disposed = true;
    gammaInput.removeEventListener('input', update);
    exposureInput.removeEventListener('input', update);
    if (gl) {
      if (baseTexture) gl.deleteTexture(baseTexture);
      if (lightTexture) gl.deleteTexture(lightTexture);
      if (buffer) gl.deleteBuffer(buffer);
      if (program) gl.deleteProgram(program);
    }
    baseImage = null;
    lightImage = null;
    draw = () => {};
  };
}

function makeTexture(gl, image) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  return texture;
}

function makeProgram(gl) {
  const vertex = compile(gl, gl.VERTEX_SHADER, `
    attribute vec2 a_position;
    varying vec2 v_uv;
    void main() {
      v_uv = a_position * 0.5 + 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_base;
    uniform sampler2D u_light;
    uniform float u_gamma;
    uniform float u_exposure;
    void main() {
      vec3 base = texture2D(u_base, v_uv).rgb;
      float light = max(texture2D(u_light, v_uv).r, 0.01);
      float multiplier = pow(light, u_gamma - 1.0) * u_exposure;
      gl_FragColor = vec4(clamp(base * multiplier, 0.0, 1.0), 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

function compile(gl, kind, source) {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
