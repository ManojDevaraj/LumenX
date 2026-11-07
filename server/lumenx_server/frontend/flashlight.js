// Cursor-driven light: eases --x/--y/--r on an element so CSS masks can reveal a restored layer.
export function createFlashlight(element, { radius = 0.2, wander = true, clickToFlood = true, reducedMotion = false } = {}) {
  const pointer = { x: 0.5, y: 0.5 };
  const light = { x: 0.5, y: 0.5, r: 0 };
  let flooded = false;
  let enabled = true;
  let lastMove = -Infinity;
  let frame = 0;
  const start = performance.now();
  const baseRadius = () => Math.max(element.clientWidth, element.clientHeight) * radius;

  const tick = (now) => {
    frame = requestAnimationFrame(tick);
    if (!enabled) return;
    const idle = now - lastMove > 2600;
    if (wander && idle && !flooded) {
      // A slow figure-of-eight path keeps the scene alive on touch screens and before the first mouse move.
      const t = (now - start) / 1000;
      pointer.x = 0.5 + Math.sin(t * 0.37) * 0.3;
      pointer.y = 0.48 + Math.sin(t * 0.74) * 0.18;
    }
    const ease = reducedMotion ? 1 : (idle ? 0.03 : 0.14);
    light.x += (pointer.x - light.x) * ease;
    light.y += (pointer.y - light.y) * ease;
    const targetRadius = flooded ? Math.hypot(element.clientWidth, element.clientHeight) * 1.1 : baseRadius();
    light.r += (targetRadius - light.r) * (reducedMotion ? 1 : (flooded ? 0.06 : 0.1));
    element.style.setProperty('--x', `${(light.x * 100).toFixed(2)}%`);
    element.style.setProperty('--y', `${(light.y * 100).toFixed(2)}%`);
    element.style.setProperty('--r', `${light.r.toFixed(1)}px`);
  };

  const onMove = (event) => {
    if (event.pointerType === 'touch') return;
    const rect = element.getBoundingClientRect();
    pointer.x = (event.clientX - rect.left) / rect.width;
    pointer.y = (event.clientY - rect.top) / rect.height;
    lastMove = performance.now();
  };
  const onClick = (event) => {
    if (!clickToFlood || event.target.closest('a, button, label, input')) return;
    flooded = !flooded;
    element.classList.toggle('is-flooded', flooded);
  };
  element.addEventListener('pointermove', onMove);
  element.addEventListener('click', onClick);
  frame = requestAnimationFrame(tick);

  return {
    setEnabled(value) { enabled = value; },
    dispose() {
      cancelAnimationFrame(frame);
      element.removeEventListener('pointermove', onMove);
      element.removeEventListener('click', onClick);
    },
  };
}
