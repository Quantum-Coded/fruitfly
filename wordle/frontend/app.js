/**
 * app.js — Wordle vs. a Fruit Fly
 * Real Connectome RL · 3D Biomechanical Embodiment · Interactive Race
 */

// ═══════════════════════════ GLOBAL STATE ═══════════════════════════
const STATE = {
  ws: null,
  connected: false,
  circuit: null,
  activeNeurons: new Set(),
  dopamineLevel: 0.0,
  gameActive: false,
  winner: null,
  resultDismissed: false,   // true once user manually closes the result card
  
  // Fly State
  fly: {
    phase: 'IDLE',
    pos: { x: 0, y: 0, z: 1.5 },
    heading: 0,
    wingAngle: 0,
    carriedLetter: null,
    guesses: [],
    feedbacks: [],
    done: false,
    won: false
  },

  // Player State
  player: {
    guesses: [],
    feedbacks: [],
    currentInput: '',
    done: false,
    won: false,
    keyStatuses: {} // letter -> 'correct' | 'present' | 'absent'
  }
};

// ═══════════════════════════ INITIALIZATION ═══════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  initThreeScenes();
  initVirtualKeyboard();
  initWordleGrids();
  setupDOMListeners();

  await fetchCircuitData();
  await fetchBenchmarkStats();
  connectWebSocket();
});

// ═══════════════════════════ 3D GRAPHICS (THREE.JS) ═══════════════════════════
let worldScene, worldCamera, worldRenderer, worldControls;
let brainScene, brainCamera, brainRenderer, brainControls;
let flyGroup, flyWings = [], flyLegs = [], carriedTileMesh;
let boardTiles3D = []; // 6 rows x 5 cols meshes
let brainPointsGeometry, brainPointsMesh, brainColorsDefault, brainColorsCurrent, brainActivityHeat, brainColorsResting;
let letterBoxMesh;

function initThreeScenes() {
  initWorldScene();
  initBrainScene();

  window.addEventListener('resize', () => {
    onWindowResizeWorld();
    onWindowResizeBrain();
  });
}

/* ──── 1. Left Viewport: 3D Embodied Fly World ──── */
function initWorldScene() {
  const container = document.getElementById('three-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  worldScene = new THREE.Scene();
  worldScene.background = new THREE.Color(0x06090e);
  worldScene.fog = new THREE.FogExp2(0x06090e, 0.08);

  worldCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  worldCamera.position.set(0, 3.8, 5.2);

  worldRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  worldRenderer.setSize(width, height);
  worldRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  worldRenderer.shadowMap.enabled = true;
  worldRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(worldRenderer.domElement);

  worldControls = new THREE.OrbitControls(worldCamera, worldRenderer.domElement);
  worldControls.enableDamping = true;
  worldControls.dampingFactor = 0.05;
  worldControls.maxPolarAngle = Math.PI / 2 - 0.05;
  worldControls.minDistance = 2.0;
  worldControls.maxDistance = 10.0;
  worldControls.target.set(0, 0.8, 0);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xdbeafe, 0.6);
  worldScene.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xffedd5, 1.3);
  sunLight.position.set(5, 10, 6);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 1024;
  sunLight.shadow.mapSize.height = 1024;
  sunLight.shadow.camera.near = 0.5;
  sunLight.shadow.camera.far = 25;
  worldScene.add(sunLight);

  const rimLight = new THREE.PointLight(0x00d4ff, 1.2, 10);
  rimLight.position.set(-4, 3, -3);
  worldScene.add(rimLight);

  // Grassy Ground Terrain
  createGrassTerrain();

  // 3D Wordle Board (Billboard)
  create3DWordleBoard();

  // 3D Wooden Letter Box
  create3DLetterBox();

  // 3D Fruit Fly Model
  create3DFruitFly();

  animateWorld();
}

function createGrassTerrain() {
  const geo = new THREE.PlaneGeometry(30, 30, 32, 32);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x142812,
    roughness: 0.9,
    metalness: 0.1
  });
  const ground = new THREE.Mesh(geo, mat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = 0;
  ground.receiveShadow = true;
  worldScene.add(ground);

  // Subtle ground grid lines
  const grid = new THREE.GridHelper(20, 20, 0x1e3a1e, 0x0f2010);
  grid.position.y = 0.005;
  worldScene.add(grid);
}

function create3DWordleBoard() {
  const boardGroup = new THREE.Group();
  boardGroup.position.set(0, 0.1, -1.0);

  // Board frame / stand
  const frameGeo = new THREE.BoxGeometry(4.2, 4.8, 0.2);
  const frameMat = new THREE.MeshStandardMaterial({
    color: 0x111622,
    roughness: 0.7,
    metalness: 0.3
  });
  const frame = new THREE.Mesh(frameGeo, frameMat);
  frame.position.set(0, 2.2, -0.1);
  frame.castShadow = true;
  frame.receiveShadow = true;
  boardGroup.add(frame);

  // Left & right stand legs
  const legGeo = new THREE.CylinderGeometry(0.06, 0.06, 2.5);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x222938, metalness: 0.8 });
  const leftLeg = new THREE.Mesh(legGeo, legMat);
  leftLeg.position.set(-1.8, 1.2, -0.15);
  boardGroup.add(leftLeg);
  const rightLeg = new THREE.Mesh(legGeo, legMat);
  rightLeg.position.set(1.8, 1.2, -0.15);
  boardGroup.add(rightLeg);

  // 6 rows x 5 columns 3D Tiles
  boardTiles3D = [];
  const startY = 4.0;
  const startX = -1.4;
  const spacing = 0.7;

  for (let r = 0; r < 6; r++) {
    boardTiles3D[r] = [];
    for (let c = 0; c < 5; c++) {
      const tileGroup = new THREE.Group();
      tileGroup.position.set(startX + c * spacing, startY - r * spacing, 0.05);

      const tGeo = new THREE.BoxGeometry(0.58, 0.58, 0.08);
      const tMat = new THREE.MeshStandardMaterial({
        color: 0x1e2738,
        roughness: 0.5,
        metalness: 0.2
      });
      const tMesh = new THREE.Mesh(tGeo, tMat);
      tMesh.castShadow = true;
      tileGroup.add(tMesh);

      // Letter canvas texture on front face
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#1e2738';
      ctx.fillRect(0, 0, 128, 128);

      const texture = new THREE.CanvasTexture(canvas);
      const labelMat = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
      const labelGeo = new THREE.PlaneGeometry(0.54, 0.54);
      const labelMesh = new THREE.Mesh(labelGeo, labelMat);
      labelMesh.position.z = 0.045;
      tileGroup.add(labelMesh);

      boardGroup.add(tileGroup);
      boardTiles3D[r][c] = {
        group: tileGroup,
        boxMesh: tMesh,
        labelMesh: labelMesh,
        canvas: canvas,
        ctx: ctx,
        texture: texture
      };
    }
  }

  worldScene.add(boardGroup);
}

function update3DBoardTile(r, c, letter, colorCode) {
  if (!boardTiles3D[r] || !boardTiles3D[r][c]) return;
  const tile = boardTiles3D[r][c];

  let bgHex = '#1e2738';
  let boxColor = 0x1e2738;
  if (colorCode === 2) {
    bgHex = '#22c55e'; // Green
    boxColor = 0x22c55e;
  } else if (colorCode === 1) {
    bgHex = '#eab308'; // Yellow
    boxColor = 0xeab308;
  } else if (colorCode === 0) {
    bgHex = '#334155'; // Gray
    boxColor = 0x334155;
  }

  tile.boxMesh.material.color.setHex(boxColor);

  const ctx = tile.ctx;
  ctx.fillStyle = bgHex;
  ctx.fillRect(0, 0, 128, 128);

  if (letter) {
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 84px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(letter.toUpperCase(), 64, 68);
  }
  tile.texture.needsUpdate = true;
}

function create3DLetterBox() {
  const boxGroup = new THREE.Group();
  boxGroup.position.set(-2.8, 0.15, 1.2);

  // Wooden Crate
  const crateGeo = new THREE.BoxGeometry(1.2, 0.3, 1.0);
  const crateMat = new THREE.MeshStandardMaterial({
    color: 0x8b5a2b,
    roughness: 0.8,
    metalness: 0.1
  });
  const crate = new THREE.Mesh(crateGeo, crateMat);
  crate.castShadow = true;
  crate.receiveShadow = true;
  boxGroup.add(crate);

  // Scattered 3D letter tiles inside crate
  const letters = ['W', 'O', 'R', 'D', 'L', 'E', 'F', 'L', 'Y'];
  letters.forEach((l, i) => {
    const tGeo = new THREE.BoxGeometry(0.22, 0.05, 0.22);
    const tMat = new THREE.MeshStandardMaterial({ color: 0xdeb887, roughness: 0.6 });
    const tile = new THREE.Mesh(tGeo, tMat);
    tile.position.set(
      (Math.random() - 0.5) * 0.8,
      0.18 + i * 0.015,
      (Math.random() - 0.5) * 0.6
    );
    tile.rotation.y = Math.random() * Math.PI;
    tile.castShadow = true;
    boxGroup.add(tile);
  });

  worldScene.add(boxGroup);
  letterBoxMesh = boxGroup;
}

/* ──── Realistic 3D Fruit Fly Model (Drosophila melanogaster) ──── */
function create3DFruitFly() {
  flyGroup = new THREE.Group();
  flyGroup.position.set(0, 0.2, 1.5);
  flyGroup.scale.set(0.65, 0.65, 0.65);

  // 1. Thorax (Golden-amber cuticle)
  const thoraxGeo = new THREE.SphereGeometry(0.35, 16, 16);
  thoraxGeo.scale(1.0, 0.9, 1.3);
  const thoraxMat = new THREE.MeshStandardMaterial({
    color: 0x96612b,
    roughness: 0.4,
    metalness: 0.3
  });
  const thorax = new THREE.Mesh(thoraxGeo, thoraxMat);
  thorax.castShadow = true;
  flyGroup.add(thorax);

  // 2. Abdomen (Striped amber & dark brown)
  const abdomenGeo = new THREE.SphereGeometry(0.42, 16, 16);
  abdomenGeo.scale(0.85, 0.8, 1.7);
  const abdomenMat = new THREE.MeshStandardMaterial({
    color: 0x6e3d17,
    roughness: 0.5,
    metalness: 0.2
  });
  const abdomen = new THREE.Mesh(abdomenGeo, abdomenMat);
  abdomen.position.set(0, 0.05, -0.75);
  abdomen.rotation.x = -0.15;
  abdomen.castShadow = true;
  flyGroup.add(abdomen);

  // 3. Head & Compound Eyes
  const headGeo = new THREE.SphereGeometry(0.24, 16, 16);
  headGeo.scale(1.1, 0.9, 0.9);
  const headMat = new THREE.MeshStandardMaterial({ color: 0x5a3416, roughness: 0.5 });
  const head = new THREE.Mesh(headGeo, headMat);
  head.position.set(0, 0.08, 0.45);
  head.castShadow = true;
  flyGroup.add(head);

  // Big Red Drosophila Compound Eyes
  const eyeGeo = new THREE.SphereGeometry(0.16, 16, 16);
  eyeGeo.scale(0.8, 1.1, 1.1);
  const eyeMat = new THREE.MeshStandardMaterial({
    color: 0xd90429, // Vivid Crimson Ruby Red
    roughness: 0.2,
    metalness: 0.4
  });
  
  const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
  leftEye.position.set(-0.2, 0.12, 0.48);
  leftEye.rotation.y = -0.35;
  flyGroup.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
  rightEye.position.set(0.2, 0.12, 0.48);
  rightEye.rotation.y = 0.35;
  flyGroup.add(rightEye);

  // 4. Wings (Translucent with delicate veins)
  const wingGeo = new THREE.PlaneGeometry(0.55, 1.3);
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xe2e8f0,
    transparent: true,
    opacity: 0.55,
    roughness: 0.1,
    metalness: 0.6,
    side: THREE.DoubleSide
  });

  const leftWing = new THREE.Mesh(wingGeo, wingMat);
  leftWing.position.set(-0.28, 0.32, -0.4);
  leftWing.rotation.x = Math.PI / 2 - 0.2;
  leftWing.rotation.y = -0.25;
  flyGroup.add(leftWing);

  const rightWing = new THREE.Mesh(wingGeo, wingMat);
  rightWing.position.set(0.28, 0.32, -0.4);
  rightWing.rotation.x = Math.PI / 2 - 0.2;
  rightWing.rotation.y = 0.25;
  flyGroup.add(rightWing);

  flyWings = [leftWing, rightWing];

  // 5. Six Articulated Legs
  const legMat = new THREE.MeshStandardMaterial({ color: 0x3d2310, roughness: 0.7 });
  flyLegs = [];
  const legPositions = [
    [-0.28, -0.1, 0.25], [-0.34, -0.1, 0.0], [-0.3, -0.1, -0.3],
    [0.28, -0.1, 0.25],  [0.34, -0.1, 0.0],  [0.3, -0.1, -0.3]
  ];

  legPositions.forEach((pos, idx) => {
    const leg = new THREE.Group();
    leg.position.set(pos[0], pos[1], pos[2]);

    const upperLegGeo = new THREE.CylinderGeometry(0.025, 0.02, 0.35);
    const upperLeg = new THREE.Mesh(upperLegGeo, legMat);
    upperLeg.rotation.z = (idx < 3 ? -1 : 1) * 0.7;
    leg.add(upperLeg);

    const lowerLegGeo = new THREE.CylinderGeometry(0.018, 0.012, 0.4);
    const lowerLeg = new THREE.Mesh(lowerLegGeo, legMat);
    lowerLeg.position.set((idx < 3 ? -0.22 : 0.22), -0.25, 0);
    lowerLeg.rotation.z = (idx < 3 ? 1 : -1) * 0.4;
    leg.add(lowerLeg);

    flyGroup.add(leg);
    flyLegs.push(leg);
  });

  // 6. Carried Letter Tile (shown when fly carries letter from crate)
  const carriedGeo = new THREE.BoxGeometry(0.35, 0.08, 0.35);
  const carriedMat = new THREE.MeshStandardMaterial({
    color: 0xdeb887,
    roughness: 0.5
  });
  carriedTileMesh = new THREE.Mesh(carriedGeo, carriedMat);
  carriedTileMesh.position.set(0, -0.35, 0.45);
  carriedTileMesh.visible = false;
  flyGroup.add(carriedTileMesh);

  worldScene.add(flyGroup);
}

function animateWorld() {
  requestAnimationFrame(animateWorld);

  // Update Fly Position and Heading smoothly
  if (flyGroup && STATE.fly) {
    const targetPos = STATE.fly.pos;
    flyGroup.position.x += (targetPos.x - flyGroup.position.x) * 0.2;
    flyGroup.position.y += (targetPos.y + 0.25 - flyGroup.position.y) * 0.2;
    flyGroup.position.z += (targetPos.z - flyGroup.position.z) * 0.2;

    // Smooth heading rotation
    const targetRot = STATE.fly.heading;
    flyGroup.rotation.y += (targetRot - flyGroup.rotation.y) * 0.2;

    // Wing Flutter Animation
    if (flyWings.length === 2) {
      const flap = Math.sin(Date.now() * 0.045) * 0.35;
      flyWings[0].rotation.z = -flap;
      flyWings[1].rotation.z = flap;
    }

    // Leg Walking Phase
    if (flyLegs.length === 6) {
      const walkGait = Math.sin(Date.now() * 0.015);
      flyLegs.forEach((leg, i) => {
        leg.rotation.x = Math.sin(Date.now() * 0.015 + (i % 2) * Math.PI) * 0.25;
      });
    }

    // Carried Letter Piece Visibility
    if (carriedTileMesh) {
      carriedTileMesh.visible = !!STATE.fly.carriedLetter;
    }
  }

  worldControls.update();
  worldRenderer.render(worldScene, worldCamera);
}

function onWindowResizeWorld() {
  const container = document.getElementById('three-container');
  if (!container || !worldRenderer) return;
  worldCamera.aspect = container.clientWidth / container.clientHeight;
  worldCamera.updateProjectionMatrix();
  worldRenderer.setSize(container.clientWidth, container.clientHeight);
}

/* ──── 2. Right Viewport: 3D Connectome Point Cloud ──── */
function initBrainScene() {
  const container = document.getElementById('connectome-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  brainScene = new THREE.Scene();
  brainScene.background = new THREE.Color(0x050608);

  brainCamera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
  brainCamera.position.set(0, 0, 2.2);

  brainRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  brainRenderer.setSize(width, height);
  brainRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(brainRenderer.domElement);

  brainControls = new THREE.OrbitControls(brainCamera, brainRenderer.domElement);
  brainControls.enableDamping = true;
  brainControls.dampingFactor = 0.05;
  brainControls.autoRotate = true;
  brainControls.autoRotateSpeed = 0.8;
  brainControls.minDistance = 1.0;
  brainControls.maxDistance = 5.0;

  animateBrain();
}

function buildBrainPointCloud(data) {
  const coords = data.coords; // (n, 3) in [-0.5, 0.5]
  const n = coords.length;

  const positions = new Float32Array(n * 3);
  brainColorsDefault = new Float32Array(n * 3);
  brainColorsResting = new Float32Array(n * 3);
  brainColorsCurrent = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    positions[i * 3]     = coords[i][0] * 1.5;
    positions[i * 3 + 1] = coords[i][1] * 1.5;
    positions[i * 3 + 2] = coords[i][2] * 1.5;

    // Assign color based on real biological population
    let r = 1.0, g = 0.7, b = 0.0; // Central Brain Amber
    if (data.is_dopamine[i]) {
      // Dopaminergic (PAM/PPL1 reward) -> Radiant Crimson Red
      r = 1.0; g = 0.16; b = 0.33;
    } else if (data.is_descending[i]) {
      // Motor / Descending -> Vivid Violet
      r = 0.85; g = 0.15; b = 1.0;
    } else if (data.is_orn_letter[i] >= 0) {
      // Sensory ORNs (Letters A-Z / Nose) -> Bright Emerald Green
      r = 0.0; g = 1.0; b = 0.45;
    } else if (data.super_classes[i] === 'optic' || data.is_visual[i]) {
      // Optic Lobe (Eyes) -> Cyan Blue
      r = 0.0; g = 0.85; b = 1.0;
    }

    brainColorsDefault[i * 3]     = r;
    brainColorsDefault[i * 3 + 1] = g;
    brainColorsDefault[i * 3 + 2] = b;

    // Dim resting background color (22% brightness) for maximum spike contrast
    const dim = 0.22;
    brainColorsResting[i * 3]     = r * dim;
    brainColorsResting[i * 3 + 1] = g * dim;
    brainColorsResting[i * 3 + 2] = b * dim;

    brainColorsCurrent[i * 3]     = r * dim;
    brainColorsCurrent[i * 3 + 1] = g * dim;
    brainColorsCurrent[i * 3 + 2] = b * dim;
  }

  brainPointsGeometry = new THREE.BufferGeometry();
  brainPointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  brainPointsGeometry.setAttribute('color', new THREE.BufferAttribute(brainColorsCurrent, 3));

  // Custom glow particle texture
  const particleCanvas = document.createElement('canvas');
  particleCanvas.width = 32;
  particleCanvas.height = 32;
  const pCtx = particleCanvas.getContext('2d');
  const gradient = pCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(0.35, 'rgba(255, 255, 255, 0.85)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
  pCtx.fillStyle = gradient;
  pCtx.fillRect(0, 0, 32, 32);
  const particleTex = new THREE.CanvasTexture(particleCanvas);

  const mat = new THREE.PointsMaterial({
    size: 0.052,
    vertexColors: true,
    map: particleTex,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  brainPointsMesh = new THREE.Points(brainPointsGeometry, mat);
  brainScene.add(brainPointsMesh);
  brainActivityHeat = new Float32Array(n);
}

function animateBrain() {
  requestAnimationFrame(animateBrain);

  if (brainPointsGeometry && brainColorsCurrent && brainActivityHeat && brainColorsResting) {
    const colors = brainPointsGeometry.attributes.color.array;
    const n = colors.length / 3;

    for (let i = 0; i < n; i++) {
      const h = brainActivityHeat[i];
      if (h > 0.01) {
        // High contrast active firing spike:
        // Surge to neon white at peak, decaying through vibrant saturated population hue
        const rPeak = brainColorsDefault[i * 3];
        const gPeak = brainColorsDefault[i * 3 + 1];
        const bPeak = brainColorsDefault[i * 3 + 2];

        const whiteBoost = Math.max(0, (h - 0.3) / 0.7);
        const rVal = rPeak + (1.0 - rPeak) * whiteBoost;
        const gVal = gPeak + (1.0 - gPeak) * whiteBoost;
        const bVal = bPeak + (1.0 - bPeak) * whiteBoost;

        colors[i * 3]     = rVal * h + brainColorsResting[i * 3] * (1.0 - h);
        colors[i * 3 + 1] = gVal * h + brainColorsResting[i * 3 + 1] * (1.0 - h);
        colors[i * 3 + 2] = bVal * h + brainColorsResting[i * 3 + 2] * (1.0 - h);

        // Smooth exponential decay (~160ms tail)
        brainActivityHeat[i] *= 0.88;
      } else {
        brainActivityHeat[i] = 0;
        colors[i * 3]     = brainColorsResting[i * 3];
        colors[i * 3 + 1] = brainColorsResting[i * 3 + 1];
        colors[i * 3 + 2] = brainColorsResting[i * 3 + 2];
      }
    }
    brainPointsGeometry.attributes.color.needsUpdate = true;
  }

  brainControls.update();
  brainRenderer.render(brainScene, brainCamera);
}

function onWindowResizeBrain() {
  const container = document.getElementById('connectome-container');
  if (!container || !brainRenderer) return;
  brainCamera.aspect = container.clientWidth / container.clientHeight;
  brainCamera.updateProjectionMatrix();
  brainRenderer.setSize(container.clientWidth, container.clientHeight);
}

// ═══════════════════════════ WORDLE 2D GRIDS ═══════════════════════════
function initWordleGrids() {
  createGridHTML('flyGrid');
  createGridHTML('playerGrid');
}

function createGridHTML(elementId) {
  const container = document.getElementById(elementId);
  container.innerHTML = '';
  for (let r = 0; r < 6; r++) {
    const row = document.createElement('div');
    row.className = 'grid-row';
    row.dataset.row = r;
    for (let c = 0; c < 5; c++) {
      const tile = document.createElement('div');
      tile.className = 'tile';
      tile.dataset.col = c;
      row.appendChild(tile);
    }
    container.appendChild(row);
  }
}

function renderFlyGrid() {
  const grid = document.getElementById('flyGrid');
  const guesses = STATE.fly.guesses || [];
  const feedbacks = STATE.fly.feedbacks || [];
  const activeRowIdx = guesses.length;
  const currentLetters = STATE.fly.current_guess_letters || [];
  const placedCount = STATE.fly.placed_letter_idx || 0;

  for (let r = 0; r < 6; r++) {
    const row = grid.children[r];
    const guess = guesses[r];
    const fb = feedbacks[r];

    for (let c = 0; c < 5; c++) {
      const tile = row.children[c];
      if (r < activeRowIdx) {
        // Committed past guesses with feedback colors
        if (guess && guess[c]) {
          tile.textContent = guess[c];
          tile.className = 'tile tile-filled';
          if (fb && fb[c] !== undefined) {
            applyTileColor(tile, fb[c]);
            update3DBoardTile(r, c, guess[c], fb[c]);
          }
        }
      } else if (r === activeRowIdx && STATE.gameActive && !STATE.fly.done) {
        // Active row: show letters placed one by one as the fly delivers each tile
        if (c < placedCount && currentLetters[c]) {
          tile.textContent = currentLetters[c];
          tile.className = 'tile tile-filled tile-fly-placed';
          update3DBoardTile(r, c, currentLetters[c], -1);
        } else if (c === placedCount && STATE.fly.carriedLetter) {
          // Live preview of letter currently being fetched/carried by the fly!
          tile.textContent = STATE.fly.carriedLetter;
          tile.className = 'tile tile-filled tile-carrying';
          update3DBoardTile(r, c, STATE.fly.carriedLetter, -1);
        } else {
          tile.textContent = '';
          tile.className = 'tile';
          update3DBoardTile(r, c, '', -1);
        }
      } else {
        tile.textContent = '';
        tile.className = 'tile';
        update3DBoardTile(r, c, '', -1);
      }
    }
  }

  document.getElementById('flyGuessCount').textContent = `${guesses.length} / 6`;
}

function renderPlayerGrid() {
  const grid = document.getElementById('playerGrid');
  const guesses = STATE.player.guesses || [];
  const feedbacks = STATE.player.feedbacks || [];
  const currInput = STATE.player.currentInput || '';
  const activeRowIdx = guesses.length;

  for (let r = 0; r < 6; r++) {
    const row = grid.children[r];
    const guess = guesses[r];
    const fb = feedbacks[r];

    for (let c = 0; c < 5; c++) {
      const tile = row.children[c];
      if (r < activeRowIdx) {
        // Committed past guesses
        tile.textContent = guess ? guess[c] : '';
        tile.className = 'tile tile-filled';
        if (fb && fb[c] !== undefined) {
          applyTileColor(tile, fb[c]);
        }
      } else if (r === activeRowIdx && !STATE.player.done) {
        // Current typing row
        const char = currInput[c] || '';
        tile.textContent = char;
        tile.className = char ? 'tile tile-filled' : 'tile';
      } else {
        tile.textContent = '';
        tile.className = 'tile';
      }
    }
  }

  document.getElementById('playerGuessCount').textContent = `${guesses.length} / 6`;
}

function applyTileColor(tile, code) {
  tile.classList.remove('tile-correct', 'tile-present', 'tile-absent');
  if (code === 2) tile.classList.add('tile-correct');
  else if (code === 1) tile.classList.add('tile-present');
  else if (code === 0) tile.classList.add('tile-absent');
}

// ═══════════════════════════ VIRTUAL KEYBOARD ═══════════════════════════
const KBD_LAYOUT = [
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '⌫']
];

function initVirtualKeyboard() {
  KBD_LAYOUT.forEach((rowKeys, rowIdx) => {
    const rowEl = document.getElementById(`kbdRow${rowIdx + 1}`);
    rowEl.innerHTML = '';
    rowKeys.forEach(key => {
      const keyBtn = document.createElement('button');
      keyBtn.className = 'key';
      keyBtn.textContent = key;
      keyBtn.dataset.key = key;
      if (key === 'ENTER' || key === '⌫') {
        keyBtn.classList.add('key-wide');
      }
      keyBtn.addEventListener('click', () => handleKeyInput(key));
      rowEl.appendChild(keyBtn);
    });
  });

  // Physical keyboard listener
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      handleKeyInput('ENTER');
    } else if (e.key === 'Backspace') {
      handleKeyInput('⌫');
    } else {
      const key = e.key.toUpperCase();
      if (/^[A-Z]$/.test(key)) {
        handleKeyInput(key);
      }
    }
  });
}

function handleKeyInput(key) {
  if (!STATE.gameActive || STATE.player.done) return;

  if (key === '⌫') {
    if (STATE.player.currentInput.length > 0) {
      STATE.player.currentInput = STATE.player.currentInput.slice(0, -1);
      renderPlayerGrid();
    }
  } else if (key === 'ENTER') {
    if (STATE.player.currentInput.length === 5) {
      submitPlayerGuess(STATE.player.currentInput);
    }
  } else {
    if (STATE.player.currentInput.length < 5) {
      STATE.player.currentInput += key;
      renderPlayerGrid();
    }
  }
}

async function submitPlayerGuess(word) {
  try {
    const res = await fetch('/player_guess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ guess: word })
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(err.error || 'Invalid guess');
      return;
    }
    const data = await res.json();
    STATE.player.guesses.push(data.guess);
    STATE.player.feedbacks.push(data.feedback);
    STATE.player.done = data.done;
    STATE.player.won = data.won;
    STATE.player.currentInput = '';

    // Update keyboard color status
    data.feedback.forEach((code, i) => {
      const char = data.guess[i];
      const prev = STATE.player.keyStatuses[char];
      if (code === 2) STATE.player.keyStatuses[char] = 'correct';
      else if (code === 1 && prev !== 'correct') STATE.player.keyStatuses[char] = 'present';
      else if (code === 0 && !prev) STATE.player.keyStatuses[char] = 'absent';
    });
    updateKeyboardUI();
    renderPlayerGrid();

    // Check race outcome
    checkGameRace();
  } catch (err) {
    console.error('Error submitting guess:', err);
  }
}

function updateKeyboardUI() {
  document.querySelectorAll('.key').forEach(keyBtn => {
    const k = keyBtn.dataset.key;
    const status = STATE.player.keyStatuses[k];
    keyBtn.classList.remove('key-correct', 'key-present', 'key-absent');
    if (status) {
      keyBtn.classList.add(`key-${status}`);
    }
  });
}

// ═══════════════════════════ WEBSOCKET TELEMETRY ═══════════════════════════
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  console.log('Connecting to WebSocket:', wsUrl);

  STATE.ws = new WebSocket(wsUrl);

  STATE.ws.onopen = () => {
    console.log('WebSocket connected!');
    STATE.connected = true;
    document.getElementById('connStatus').textContent = 'LIVE 20HZ';
    const dot = document.getElementById('pulseDot');
    if (dot) dot.classList.remove('offline');
  };

  STATE.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onServerTelemetry(data);
  };

  STATE.ws.onclose = () => {
    console.log('WebSocket disconnected. Reconnecting in 2s...');
    STATE.connected = false;
    document.getElementById('connStatus').textContent = 'OFFLINE';
    const dot = document.getElementById('pulseDot');
    if (dot) dot.classList.add('offline');
    setTimeout(connectWebSocket, 2000);
  };
}

function onServerTelemetry(data) {
  STATE.gameActive = data.game_active;
  STATE.winner = data.winner;

  // Fly state update
  if (data.fly) {
    STATE.fly = data.fly;
    renderFlyGrid();
    updateFlyHUD();
  }

  // Player state update from server sync
  if (data.player && data.player.guesses.length > STATE.player.guesses.length) {
    STATE.player.guesses = data.player.guesses;
    STATE.player.feedbacks = data.player.feedbacks;
    STATE.player.done = data.player.done;
    STATE.player.won = data.player.won;
    renderPlayerGrid();
  }

  // Reveal player's answer word when:
  //   (a) fly won and game ended, or (b) player used all 6 guesses without winning
  if (data.player && data.player.secret && !STATE.player.won) {
    showPlayerAnswerReveal(data.player.secret);
  }

  // Connectome brain spikes
  if (data.brain) {
    const newlyActive = data.brain.active_neurons || [];
    STATE.activeNeurons = new Set(newlyActive);
    if (brainActivityHeat && newlyActive.length > 0) {
      for (let i = 0; i < newlyActive.length; i++) {
        brainActivityHeat[newlyActive[i]] = 1.0;
      }
    }
    STATE.dopamineLevel = data.brain.dopamine_pulse || 0.0;

    // Update Sensory HUD card and dynamic glow
    updateSensoryHUD(data.brain.sensory_mode, data.brain.sensory_detail);

    // Update telemetry bar
    const tel = data.brain.telemetry || {};
    document.getElementById('metricNeuronsFired').textContent = (tel.active_count || 0).toLocaleString();
    const totalSpikes = (tel.spikes_orn || 0) + (tel.spikes_visual || 0) + (tel.spikes_mb || 0) + (tel.spikes_dopamine || 0) + (tel.spikes_motor || 0);
    document.getElementById('metricSpikes').textContent = totalSpikes.toLocaleString();
    document.getElementById('metricDopamine').textContent = (STATE.dopamineLevel || 0).toFixed(2);
    document.getElementById('metricSimTime').textContent = `${(data.t % 1000).toFixed(1)} ms`;

    // Dopamine reward flare effect
    const flare = document.getElementById('dopamineFlare');
    if (STATE.dopamineLevel > 0.3) {
      flare.classList.add('active');
    } else {
      flare.classList.remove('active');
    }

    // Brain status pill
    const brainPill = document.getElementById('brainStatePill');
    if (STATE.dopamineLevel > 0.3) {
      brainPill.textContent = 'DOPAMINE SURGE!';
      brainPill.className = 'status-pill status-dopamine';
    } else if (STATE.activeNeurons.size > 0) {
      brainPill.textContent = 'SPIKING';
      brainPill.className = 'status-pill status-spiking';
    } else {
      brainPill.textContent = 'IDLE';
      brainPill.className = 'status-pill status-idle';
    }
  }

  checkGameRace();
}

function updateSensoryHUD(mode, detail) {
  const dot = document.getElementById('sensoryHudDot');
  const title = document.getElementById('sensoryHudTitle');
  const detailEl = document.getElementById('sensoryHudDetail');
  const vignette = document.getElementById('sensoryVignette');

  if (!dot || !title || !detailEl) return;

  dot.className = 'sensory-hud-dot';
  if (vignette) vignette.className = 'sensory-vignette';

  if (mode === 'NOSE') {
    dot.classList.add('nose');
    title.textContent = '👃 OLFACTORY NOSE · SMELLING LETTER (26 ORN CHANNELS)';
    if (vignette) vignette.classList.add('active-nose');
  } else if (mode === 'EYES') {
    dot.classList.add('eyes');
    title.textContent = '👁 VISION · OPTIC LOBE (TILE COLOR FEEDBACK)';
    if (vignette) vignette.classList.add('active-eyes');
  } else if (mode === 'PLACEMENT') {
    dot.classList.add('placement');
    title.textContent = '📍 CONFIRMATION · RETINOTOPIC SENSORY BURST';
    if (vignette) vignette.classList.add('active-placement');
  } else if (mode === 'DOPAMINE') {
    dot.classList.add('dopamine');
    title.textContent = '⚡ REWARD · PAM/PPL1 DOPAMINE BURST';
    if (vignette) vignette.classList.add('active-dopamine');
  } else if (mode === 'MOTOR') {
    dot.classList.add('motor');
    title.textContent = '⚡ MOTOR · DESCENDING LOCOMOTION';
  } else if (mode === 'WORKING_MEMORY') {
    dot.classList.add('memory');
    title.textContent = '🧠 WORKING MEMORY · MUSHROOM BODY (KENYON CELLS)';
  } else {
    title.textContent = 'CONNECTOME RESTING';
  }

  detailEl.textContent = detail || 'Simulation running at 20 Hz';
}

function updateFlyHUD() {
  const bubbleText = document.getElementById('flyBubbleText');
  const tileBadge = document.getElementById('tileHeldBadge');
  const tileLetter = document.getElementById('tileHeldLetter');
  const phase = STATE.fly.phase;

  if (phase === 'THINKING') {
    bubbleText.textContent = 'Scanning connectome... Mushroom body associative memory evaluating words!';
    tileBadge.style.display = 'none';
  } else if (phase === 'WALKING_TO_BOX') {
    const nextNum = (STATE.fly.placed_letter_idx || 0) + 1;
    bubbleText.textContent = `Walking to letter crate for letter #${nextNum}... Descending motor neurons firing!`;
    tileBadge.style.display = 'none';
  } else if (phase === 'PICKING_TILE' || phase === 'CARRYING_TILE' || phase === 'WALKING_TO_BOARD') {
    const char = STATE.fly.carriedLetter || 'TILE';
    bubbleText.textContent = `Carrying letter '${char}' — Olfactory ORN ('${char}' smell) firing in antennal lobe!`;
    tileBadge.style.display = 'inline-block';
    tileLetter.textContent = char;
  } else if (phase === 'PLACING_TILE') {
    const char = STATE.fly.carriedLetter || (STATE.fly.current_guess_letters && STATE.fly.current_guess_letters[STATE.fly.current_letter_idx]) || '';
    const slot = (STATE.fly.current_letter_idx || 0) + 1;
    bubbleText.textContent = `Placing '${char}' into slot #${slot}! Visual retinotopic + sensory confirmation pulse!`;
    tileBadge.style.display = 'none';
  } else if (phase === 'REVEALING') {
    bubbleText.textContent = 'Word complete! Optic visual neurons processing tile colors (🟩/🟨/⬜)!';
    tileBadge.style.display = 'none';
  } else if (phase === 'DOPAMINE_PULSE') {
    bubbleText.textContent = 'GOAL STATE REACHED! Dopamine PAM/PPL1 firing massive reward surge!';
    tileBadge.style.display = 'none';
  } else if (phase === 'GAME_OVER') {
    if (STATE.fly.won) {
      bubbleText.textContent = `SOLVED in ${STATE.fly.guesses.length} guesses! 🪰🏆`;
    } else {
      bubbleText.textContent = `Fly finished all guesses! Word was: ${STATE.fly.secret || ''}`;
    }
    tileBadge.style.display = 'none';
  }
}

function checkGameRace() {
  const statusPill = document.getElementById('gameStatusPill');
  if (!STATE.gameActive) {
    if (STATE.winner === 'player') {
      statusPill.textContent = 'YOU WIN!';
      statusPill.className = 'status-pill status-won';
      showResultOverlay('🏆', 'You Win!', `You beat the fly! The fly needed ${STATE.fly.guesses.length || '?'} guesses.`);
    } else if (STATE.winner === 'fly') {
      statusPill.textContent = 'FLY WINS!';
      statusPill.className = 'status-pill status-lost';
      showResultOverlay('🪰', 'The Fly Won!', `The connectome solved it in ${STATE.fly.guesses.length || '?'} guesses. PAM neurons fired!`);
    } else if (STATE.winner === 'draw') {
      statusPill.textContent = 'DRAW!';
      statusPill.className = 'status-pill';
      showResultOverlay('🤝', 'It\'s a Draw!', 'Both solved it on the same guess — remarkable!');
    } else {
      statusPill.textContent = 'READY';
      statusPill.className = 'status-pill status-idle';
    }
  } else {
    statusPill.textContent = 'RACE IN PROGRESS!';
    statusPill.className = 'status-pill status-spiking';
  }
}

// ═══════════════════════════ REST API & DATA ═══════════════════════════
async function fetchCircuitData() {
  try {
    const res = await fetch('/circuit');
    if (!res.ok) throw new Error('Could not fetch circuit data');
    const data = await res.json();
    STATE.circuit = data;
    buildBrainPointCloud(data);
    console.log(`Loaded ${data.n} neurons from FlyWire connectome circuit!`);
  } catch (err) {
    console.error('Failed to load circuit:', err);
  }
}

async function fetchBenchmarkStats() {
  try {
    const res = await fetch('/stats');
    if (!res.ok) return;
    const stats = await res.json();
    console.log('Loaded benchmark stats:', stats);
  } catch (e) {}
}

function setupDOMListeners() {
  function resetPlayerState() {
    STATE.player.guesses = [];
    STATE.player.feedbacks = [];
    STATE.player.currentInput = '';
    STATE.player.done = false;
    STATE.player.won = false;
    STATE.player.keyStatuses = {};
    STATE.winner = null;
    STATE.resultDismissed = false;  // allow result card to show for the new game
    updateKeyboardUI();
    renderPlayerGrid();
    renderFlyGrid();
    hideResultOverlay();
    hidePlayerAnswerReveal();       // clear answer banner on new game
  }

  document.getElementById('btnStartGame').addEventListener('click', async () => {
    if (STATE.ws && STATE.connected) {
      STATE.ws.send(JSON.stringify({ action: 'start_game' }));
    } else {
      await fetch('/new_game', { method: 'POST' });
    }
    resetPlayerState();
  });

  document.getElementById('btnReset').addEventListener('click', async () => {
    if (STATE.ws && STATE.connected) {
      STATE.ws.send(JSON.stringify({ action: 'reset' }));
    }
    resetPlayerState();
  });

  const btnPlayAgain = document.getElementById('btnPlayAgain');
  if (btnPlayAgain) {
    btnPlayAgain.addEventListener('click', async () => {
      hideResultOverlay();
      if (STATE.ws && STATE.connected) {
        STATE.ws.send(JSON.stringify({ action: 'start_game' }));
      } else {
        await fetch('/new_game', { method: 'POST' });
      }
      resetPlayerState();
    });
  }

  // ── Sub-nav tab switching ──
  const sections = {
    game:    document.getElementById('app-container'),
    brain:   null, // shown via scroll to right-panel
    science: null,
    about:   null
  };

  document.querySelectorAll('.sub-nav-link').forEach(link => {
    link.addEventListener('click', () => {
      document.querySelectorAll('.sub-nav-link').forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      const section = link.dataset.section;
      if (section === 'game') {
        document.getElementById('left-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else if (section === 'brain') {
        document.getElementById('right-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else if (section === 'science') {
        document.querySelector('.science-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else if (section === 'about') {
        // Show a quick info toast linking to GitHub
        showToast('📖 Visit https://github.com/Quantum-Coded/fruitfly for full details');
      }
    });
  });

  // Close result overlay on backdrop click
  document.getElementById('result-overlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('result-overlay')) hideResultOverlay(true);
  });

  // Close X button (top-right of card)
  const btnCloseResult = document.getElementById('btnCloseResult');
  if (btnCloseResult) {
    btnCloseResult.addEventListener('click', () => hideResultOverlay(true));
  }

  // "Explore Site" dismiss button (no new game)
  const btnDismissResult = document.getElementById('btnDismissResult');
  if (btnDismissResult) {
    btnDismissResult.addEventListener('click', () => hideResultOverlay(true));
  }
}

function showToast(msg) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 2000);
}

function showResultOverlay(emoji, headline, sub) {
  const overlay = document.getElementById('result-overlay');
  if (!overlay) return;
  // Don't re-show if already visible OR if user manually dismissed it
  if (overlay.classList.contains('visible') || STATE.resultDismissed) return;
  document.getElementById('resultEmoji').textContent = emoji;
  document.getElementById('resultHeadline').textContent = headline;
  document.getElementById('resultSub').textContent = sub;
  overlay.classList.add('visible');
}

function hideResultOverlay(byUser = false) {
  const overlay = document.getElementById('result-overlay');
  if (overlay) overlay.classList.remove('visible');
  if (byUser) STATE.resultDismissed = true;
}

// ═══════════════════════════ ANSWER REVEAL ═══════════════════════════
let _answerRevealed = false; // guard: only slide in once per game end

function showPlayerAnswerReveal(secretWord) {
  if (_answerRevealed) return;           // don't re-trigger every WS tick
  _answerRevealed = true;
  const banner = document.getElementById('playerAnswerReveal');
  const wordEl = document.getElementById('playerAnswerWord');
  if (!banner || !wordEl) return;
  wordEl.textContent = secretWord.toUpperCase();
  banner.style.display = 'flex';
}

function hidePlayerAnswerReveal() {
  _answerRevealed = false;
  const banner = document.getElementById('playerAnswerReveal');
  if (banner) banner.style.display = 'none';
}
