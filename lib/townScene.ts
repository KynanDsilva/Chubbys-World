import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { ditherMaterial, type DitherUniforms, type StructureSlot } from "./scrapyardScene";
import { HELIPAD } from "./helicopterModel";

// ---------------------------------------------------------------------------
// Town layout constants (x = east, z = south). Crossroads at the origin.
// ---------------------------------------------------------------------------
export const TOWN_HALF = 30.0;
export const ROAD_HALF = 3.0;
export const SIDEWALK_W = 1.5;
export const FLOOR_H = 3.3;

export interface Rect {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

/** Reserved empty lots where newly approved structures are built (face the nearest street). */
export const LOT_SLOTS: StructureSlot[] = [
  { x: -11, z: -10, rot: 0 },
  { x: 11, z: -10, rot: 0 },
  { x: -11, z: 10, rot: Math.PI },
  { x: 11, z: 10, rot: Math.PI },
  { x: -21, z: -10, rot: 0 },
  { x: -21, z: 10, rot: Math.PI },
  { x: 21, z: 10, rot: Math.PI },
];

/** Three solar charging stations around town. Index 0 is the original DB structure. */
export const SOLAR_SLOTS: StructureSlot[] = [];

export interface BuildingSpec {
  name: string;
  cx: number;
  cz: number;
  w: number;
  d: number;
  floors: number;
  color: number;
  rot: number;
  awning?: number;
}

export const BUILDINGS: BuildingSpec[] = [
  { name: "BAKERY", cx: -8.5, cz: -22.5, w: 7, d: 9, floors: 2, color: 0xe6c78a, rot: 0, awning: 0xc96442 },
  { name: "LIBRARY", cx: 19, cz: -22, w: 13, d: 10, floors: 3, color: 0x8a8fa8, rot: 0 },
  { name: "APARTMENTS", cx: 8.5, cz: -22.5, w: 7, d: 9, floors: 3, color: 0xc7b79a, rot: 0 },
  { name: "CAFE", cx: -19, cz: 22, w: 11, d: 9, floors: 2, color: 0x8fb39a, rot: Math.PI, awning: 0x4d7c57 },
  { name: "WORKSHOP", cx: -8.5, cz: 22.5, w: 7, d: 8, floors: 2, color: 0xa8a39a, rot: Math.PI },
  { name: "POST OFFICE", cx: 19, cz: 22, w: 11, d: 9, floors: 2, color: 0xb5736a, rot: Math.PI, awning: 0x3b7ea1 },
  { name: "APARTMENTS", cx: 8.5, cz: 22.5, w: 7, d: 8, floors: 3, color: 0xd1c3a5, rot: Math.PI },
];

/** Art Studio footprint (2 floors, walk-in ground floor, front door faces +z). */
export const STUDIO = {
  cx: -19,
  cz: -22,
  w: 13,
  d: 10,
  floors: 2,
  doorW: 3.0,
  doorH: 2.6,
  // Easel (Chubby draws facing -z) and where Chubby stands (world coords).
  easelX: -23.2,
  easelZ: -25.2,
  standX: -23.2,
  standZ: -23.4,
};

const TREE_SPOTS: Array<[number, number]> = [
  [6.5, -8],
  [-6.8, -7],
  [-6.8, 9],
  [6.8, 12],
  [-16.5, -12],
  [-16.5, 8],
  [16.5, 13],
  [-27, -10],
  [-27, 11],
  [27, 14],
  [-27, -16],
  [-3.5, 15],
  [3.5, -15],
];

const BENCH_SPOTS: Array<{ x: number; z: number; rot: number }> = [
  { x: -16.5, z: -6, rot: 0 },
  { x: 16.5, z: 6, rot: Math.PI },
  { x: 6.5, z: -12, rot: Math.PI / 2 },
  { x: -6.8, z: 13, rot: -Math.PI / 2 },
];

/** Good spots for Chubby to wander to when he has no specific destination. */
export const WANDER_POINTS: Array<{ x: number; z: number }> = [
  { x: 0, z: 0 },
  { x: 0, z: -12 },
  { x: 0, z: 12 },
  { x: -12, z: 0 },
  { x: 12, z: 0 },
  { x: -16, z: -6 },
  { x: 16, z: 6 },
  { x: -26, z: 0 },
  { x: 26, z: 0 },
  { x: 7, z: -8 },
  { x: -7, z: 9 },
  { x: 0, z: 22 },
  { x: -15, z: -14 },
  { x: 15, z: 14 },
  { x: HELIPAD.standX, z: HELIPAD.standZ },
];

// ---------------------------------------------------------------------------
// Collision + pathfinding
// ---------------------------------------------------------------------------
export const COLLIDERS: Rect[] = [];

function addCollider(minX: number, maxX: number, minZ: number, maxZ: number): void {
  COLLIDERS.push({ minX, maxX, minZ, maxZ });
}

(function buildColliders() {
  for (const b of BUILDINGS) {
    addCollider(b.cx - b.w / 2, b.cx + b.w / 2, b.cz - b.d / 2, b.cz + b.d / 2);
  }
  // Art studio: walls only, with a door gap in the front wall (z = cz + d/2).
  const s = STUDIO;
  const t = 0.3;
  const x0 = s.cx - s.w / 2;
  const x1 = s.cx + s.w / 2;
  const z0 = s.cz - s.d / 2;
  const z1 = s.cz + s.d / 2;
  addCollider(x0, s.cx - s.doorW / 2, z1 - t, z1);
  addCollider(s.cx + s.doorW / 2, x1, z1 - t, z1);
  addCollider(x0, x1, z0, z0 + t);
  addCollider(x0, x0 + t, z0, z1);
  addCollider(x1 - t, x1, z0, z1);
  for (const [tx, tz] of TREE_SPOTS) addCollider(tx - 0.35, tx + 0.35, tz - 0.35, tz + 0.35);
  for (const p of SOLAR_SLOTS) addCollider(p.x - 0.5, p.x + 0.5, p.z - 0.5, p.z + 0.5);
  // Parked helicopter (fuselage + skids, tail boom along x).
  addCollider(HELIPAD.x - 5.3, HELIPAD.x + 5.3, HELIPAD.z - 1.2, HELIPAD.z + 1.2);
})();

/** Pushes a circle (x,z,r) out of every collider; keeps it inside the town. */
export function resolveCollisions(x: number, z: number, radius: number): { x: number; z: number } {
  let px = x;
  let pz = z;
  for (let iter = 0; iter < 3; iter++) {
    for (const r of COLLIDERS) {
      const cx = Math.max(r.minX, Math.min(px, r.maxX));
      const cz = Math.max(r.minZ, Math.min(pz, r.maxZ));
      const dx = px - cx;
      const dz = pz - cz;
      const dist = Math.hypot(dx, dz);
      if (dist < radius) {
        if (dist > 1e-5) {
          px = cx + (dx / dist) * radius;
          pz = cz + (dz / dist) * radius;
        } else {
          // Center is inside the rect: push out along the shallowest axis.
          const left = px - r.minX;
          const right = r.maxX - px;
          const top = pz - r.minZ;
          const bottom = r.maxZ - pz;
          const m = Math.min(left, right, top, bottom);
          if (m === left) px = r.minX - radius;
          else if (m === right) px = r.maxX + radius;
          else if (m === top) pz = r.minZ - radius;
          else pz = r.maxZ + radius;
        }
      }
    }
  }
  const lim = TOWN_HALF - 2.0;
  return { x: Math.max(-lim, Math.min(lim, px)), z: Math.max(-lim, Math.min(lim, pz)) };
}

const CELL = 0.5;
const GRID_N = Math.round((TOWN_HALF * 2) / CELL);
const PATH_RADIUS = 0.6;

function isBlocked(x: number, z: number): boolean {
  const lim = TOWN_HALF - 1.8;
  if (Math.abs(x) > lim || Math.abs(z) > lim) return true;
  for (const r of COLLIDERS) {
    if (x > r.minX - PATH_RADIUS && x < r.maxX + PATH_RADIUS && z > r.minZ - PATH_RADIUS && z < r.maxZ + PATH_RADIUS) {
      return true;
    }
  }
  return false;
}

function toCell(v: number): number {
  return Math.max(0, Math.min(GRID_N - 1, Math.floor((v + TOWN_HALF) / CELL)));
}
function cellCenter(i: number): number {
  return -TOWN_HALF + (i + 0.5) * CELL;
}

let blockedGrid: Uint8Array | null = null;
function getGrid(): Uint8Array {
  if (blockedGrid) return blockedGrid;
  const g = new Uint8Array(GRID_N * GRID_N);
  for (let iz = 0; iz < GRID_N; iz++) {
    for (let ix = 0; ix < GRID_N; ix++) {
      g[iz * GRID_N + ix] = isBlocked(cellCenter(ix), cellCenter(iz)) ? 1 : 0;
    }
  }
  blockedGrid = g;
  return g;
}

function segmentFree(ax: number, az: number, bx: number, bz: number): boolean {
  const dist = Math.hypot(bx - ax, bz - az);
  const steps = Math.max(1, Math.ceil(dist / 0.25));
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    if (isBlocked(ax + (bx - ax) * t, az + (bz - az) * t)) return false;
  }
  return true;
}

/** Breadth-first search over a blocked-cell grid, then line-of-sight smoothing. */
export function findPath(
  fx: number,
  fz: number,
  tx: number,
  tz: number
): Array<{ x: number; z: number }> {
  const grid = getGrid();
  const nearestFree = (x: number, z: number): [number, number] => {
    const cx = toCell(x);
    const cz = toCell(z);
    if (!grid[cz * GRID_N + cx]) return [cx, cz];
    for (let r = 1; r < 40; r++) {
      for (let dz = -r; dz <= r; dz++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
          const nx = cx + dx;
          const nz = cz + dz;
          if (nx < 0 || nz < 0 || nx >= GRID_N || nz >= GRID_N) continue;
          if (!grid[nz * GRID_N + nx]) return [nx, nz];
        }
      }
    }
    return [cx, cz];
  };

  const [sx, sz] = nearestFree(fx, fz);
  const [ex, ez] = nearestFree(tx, tz);
  const startIdx = sz * GRID_N + sx;
  const endIdx = ez * GRID_N + ex;
  const prev = new Int32Array(GRID_N * GRID_N).fill(-2);
  prev[startIdx] = -1;
  const queue: number[] = [startIdx];
  const dirs = [
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
  ];
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    if (cur === endIdx) break;
    const cx = cur % GRID_N;
    const cz = Math.floor(cur / GRID_N);
    for (const [dx, dz] of dirs) {
      const nx = cx + dx;
      const nz = cz + dz;
      if (nx < 0 || nz < 0 || nx >= GRID_N || nz >= GRID_N) continue;
      const ni = nz * GRID_N + nx;
      if (prev[ni] !== -2 || grid[ni]) continue;
      if (dx !== 0 && dz !== 0 && (grid[cz * GRID_N + nx] || grid[nz * GRID_N + cx])) continue;
      prev[ni] = cur;
      queue.push(ni);
    }
  }
  if (prev[endIdx] === -2) {
    return [{ x: tx, z: tz }];
  }
  const cells: Array<{ x: number; z: number }> = [];
  for (let c = endIdx; c !== -1; c = prev[c]) {
    cells.push({ x: cellCenter(c % GRID_N), z: cellCenter(Math.floor(c / GRID_N)) });
  }
  cells.reverse();
  // Replace the final cell center with the exact target when it is reachable.
  const exactEnd = isBlocked(tx, tz) ? cells[cells.length - 1] : { x: tx, z: tz };
  cells[cells.length - 1] = exactEnd;

  // Greedy line-of-sight smoothing.
  const out: Array<{ x: number; z: number }> = [];
  let anchor = { x: fx, z: fz };
  let i = 0;
  while (i < cells.length) {
    let far = i;
    for (let j = cells.length - 1; j > i; j--) {
      if (segmentFree(anchor.x, anchor.z, cells[j].x, cells[j].z)) {
        far = j;
        break;
      }
    }
    out.push(cells[far]);
    anchor = cells[far];
    i = far + 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Mesh helpers
// ---------------------------------------------------------------------------
function std(params: THREE.MeshStandardMaterialParameters, u: DitherUniforms): THREE.MeshStandardMaterial {
  return ditherMaterial(new THREE.MeshStandardMaterial({ flatShading: true, ...params }), u);
}

function box(
  w: number,
  h: number,
  d: number,
  mat: THREE.Material,
  x: number,
  y: number,
  z: number,
  shadow = true
): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = shadow;
  m.receiveShadow = true;
  return m;
}

function makeSignTexture(text: string, bg: string, fg: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 64;
  const ctx = c.getContext("2d");
  if (ctx) {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 256, 64);
    ctx.strokeStyle = fg;
    ctx.lineWidth = 4;
    ctx.strokeRect(4, 4, 248, 56);
    ctx.fillStyle = fg;
    ctx.font = "bold 30px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 128, 34);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function signMesh(text: string, width: number, bg: string, fg: string): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.PlaneGeometry(width, width / 4),
    new THREE.MeshBasicMaterial({ map: makeSignTexture(text, bg, fg) })
  );
}

interface TownMaterials {
  window: THREE.MeshStandardMaterial;
  windows: THREE.MeshStandardMaterial[];
}

function mergeAndDispose(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const merged = mergeGeometries(geos, false);
  for (const g of geos) g.dispose();
  return merged;
}

function addWindows(
  g: THREE.Group,
  spec: { w: number; d: number; floors: number },
  mats: TownMaterials,
  skipGroundFront: boolean,
  artX?: number
): void {
  const baseGeo = new THREE.BoxGeometry(1.1, 1.4, 0.1);
  const geos: THREE.BufferGeometry[] = [];
  const place = (x: number, y: number, z: number, ry: number) => {
    const cg = baseGeo.clone();
    if (ry !== 0) cg.rotateY(ry);
    cg.translate(x, y, z);
    geos.push(cg);
  };
  for (let f = 0; f < spec.floors; f++) {
    const y = f * FLOOR_H + 1.9;
    const nFront = Math.max(1, Math.floor(spec.w / 2.4));
    for (let i = 0; i < nFront; i++) {
      const x = -spec.w / 2 + (spec.w / nFront) * (i + 0.5);
      if (skipGroundFront && f === 0) {
        // Skip front door zone
        if (Math.abs(x) < 1.4) continue;
        // Never put drawings on a window: omit any ground-floor window near the wall art
        if (artX !== undefined && Math.abs(x - artX) < 1.6) continue;
      }
      place(x, y, spec.d / 2 + 0.02, 0);
      place(x, y, -spec.d / 2 - 0.02, Math.PI);
    }
    const nSide = Math.max(1, Math.floor(spec.d / 2.8));
    for (let i = 0; i < nSide; i++) {
      const z = -spec.d / 2 + (spec.d / nSide) * (i + 0.5);
      place(spec.w / 2 + 0.02, y, z, Math.PI / 2);
      place(-spec.w / 2 - 0.02, y, z, -Math.PI / 2);
    }
  }
  baseGeo.dispose();
  if (geos.length > 0) {
    g.add(new THREE.Mesh(mergeAndDispose(geos), mats.window));
  }
}

export interface WallArtSlot {
  id: string;
  name: string;
  buildingName: string;
  canvasMesh: THREE.Mesh;
  group: THREE.Group;
}

export function createWallArtMount(
  name: string,
  buildingName: string,
  id: string,
  u: DitherUniforms,
  w = 1.35,
  h = 1.35
): { group: THREE.Group; canvasMesh: THREE.Mesh; slot: WallArtSlot } {
  const g = new THREE.Group();
  const frameMat = std({ color: 0x3e2c1e, roughness: 0.7 }, u);
  const backMat = std({ color: 0x1f1914, roughness: 0.9 }, u);

  // Outer wooden frame trim bars surrounding the canvas
  const border = 0.09;
  const frameDepth = 0.06;
  const frameZ = 0.03;
  g.add(box(w + border * 2, border, frameDepth, frameMat, 0, h / 2 + border / 2, frameZ, false));
  g.add(box(w + border * 2, border, frameDepth, frameMat, 0, -h / 2 - border / 2, frameZ, false));
  g.add(box(border, h, frameDepth, frameMat, -w / 2 - border / 2, 0, frameZ, false));
  g.add(box(border, h, frameDepth, frameMat, w / 2 + border / 2, 0, frameZ, false));

  // Dark matte backboard
  g.add(box(w + border, h + border, 0.02, backMat, 0, 0, 0.01, false));

  // Overhead brass picture lamp
  const brass = std({ color: 0xa88f58, roughness: 0.4, metalness: 0.7 }, u);
  g.add(box(0.06, 0.06, 0.28, brass, 0, h / 2 + 0.15, 0.14, false));
  g.add(box(w * 0.75, 0.05, 0.08, brass, 0, h / 2 + 0.15, 0.28, false));
  const glowBar = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.7, 0.02, 0.05),
    new THREE.MeshBasicMaterial({ color: 0xfff3d4 })
  );
  glowBar.position.set(0, h / 2 + 0.125, 0.28);
  g.add(glowBar);

  // Artwork canvas plane - cleanly elevated in front of backboard and inside frame trim
  const canvasMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
  );
  canvasMesh.position.set(0, 0, 0.025);
  g.add(canvasMesh);

  const slot: WallArtSlot = {
    id,
    name,
    buildingName,
    canvasMesh,
    group: g,
  };
  return { group: g, canvasMesh, slot };
}

function buildBuilding(
  spec: BuildingSpec,
  u: DitherUniforms,
  mats: TownMaterials,
  wallArtSlots: WallArtSlot[]
): THREE.Group {
  const g = new THREE.Group();
  g.position.set(spec.cx, 0, spec.cz);
  g.rotation.y = spec.rot;
  const H = spec.floors * FLOOR_H;
  const wall = std({ color: spec.color, roughness: 0.9 }, u);
  const trim = std({ color: 0x4a4038, roughness: 0.9 }, u);

  g.add(box(spec.w, H, spec.d, wall, 0, H / 2, 0));
  for (let f = 1; f < spec.floors; f++) {
    g.add(box(spec.w + 0.14, 0.14, spec.d + 0.14, trim, 0, f * FLOOR_H, 0, false));
  }
  // Roof slab + parapet + rooftop clutter
  g.add(box(spec.w + 0.5, 0.25, spec.d + 0.5, trim, 0, H + 0.12, 0));
  g.add(box(2.0, 1.0, 1.6, std({ color: 0x8c9296, roughness: 0.6, metalness: 0.4 }, u), spec.w * 0.2, H + 0.75, -spec.d * 0.15));
  if (spec.floors >= 3) {
    const tank = new THREE.Mesh(
      new THREE.CylinderGeometry(0.8, 0.8, 1.4, 10),
      std({ color: 0x7a5c44, roughness: 0.9 }, u)
    );
    tank.position.set(-spec.w * 0.25, H + 0.95, spec.d * 0.1);
    tank.castShadow = true;
    g.add(tank);
  }

  // Pre-calculate wall art position so addWindows omits the window there
  let artX: number | undefined = undefined;
  if (spec.name === "BAKERY") artX = 2.2;
  else if (spec.name === "LIBRARY") artX = -3.8;
  else if (spec.name === "CAFE") artX = 3.2;
  else if (spec.name === "WORKSHOP") artX = 2.2;
  else if (spec.name === "POST OFFICE") artX = -3.2;
  else if (spec.name === "APARTMENTS") artX = spec.cz < 0 ? -2.2 : 2.2;

  // Front door
  g.add(box(1.5, 2.4, 0.12, std({ color: 0x3b2b22, roughness: 0.8 }, u), 0, 1.2, spec.d / 2 + 0.03, false));
  addWindows(g, spec, mats, true, artX);

  if (spec.awning !== undefined) {
    const aw = box(spec.w - 1.2, 0.12, 1.3, std({ color: spec.awning, roughness: 0.8 }, u), 0, 2.9, spec.d / 2 + 0.65);
    aw.rotation.x = 0.18;
    g.add(aw);
  }

  // Exterior wall art display mount on solid building wall (never on a window!)
  if (spec.name === "BAKERY") {
    const mount = createWallArtMount("Bakery Wall", "Bakery", "bakery", u);
    mount.group.position.set(artX!, 1.75, spec.d / 2 + 0.05);
    g.add(mount.group);
    wallArtSlots.push(mount.slot);
  } else if (spec.name === "LIBRARY") {
    const mount = createWallArtMount("Library Wall", "Library", "library", u, 1.45, 1.45);
    mount.group.position.set(artX!, 1.85, spec.d / 2 + 0.05);
    g.add(mount.group);
    wallArtSlots.push(mount.slot);
  } else if (spec.name === "CAFE") {
    const mount = createWallArtMount("Cafe Wall", "Cafe", "cafe", u);
    mount.group.position.set(artX!, 1.85, spec.d / 2 + 0.05);
    g.add(mount.group);
    wallArtSlots.push(mount.slot);
  } else if (spec.name === "WORKSHOP") {
    const mount = createWallArtMount("Workshop Wall", "Workshop", "workshop", u);
    mount.group.position.set(artX!, 1.85, spec.d / 2 + 0.05);
    g.add(mount.group);
    wallArtSlots.push(mount.slot);
  } else if (spec.name === "POST OFFICE") {
    const mount = createWallArtMount("Post Office Wall", "Post Office", "post_office", u);
    mount.group.position.set(artX!, 1.85, spec.d / 2 + 0.05);
    g.add(mount.group);
    wallArtSlots.push(mount.slot);
  } else if (spec.name === "APARTMENTS") {
    const isNorth = spec.cz < 0;
    const mount = createWallArtMount(
      isNorth ? "Apartments North Wall" : "Apartments South Wall",
      "Apartments",
      isNorth ? "apartments_north" : "apartments_south",
      u
    );
    mount.group.position.set(artX!, 1.85, spec.d / 2 + 0.05);
    g.add(mount.group);
    wallArtSlots.push(mount.slot);
  }

  return g;
}

export interface GalleryHandles {
  easelCanvas: THREE.Mesh;
  frames: THREE.Mesh[];
}

function buildArtStudioBuilding(
  u: DitherUniforms,
  mats: TownMaterials,
  wallArtSlots: WallArtSlot[]
): { group: THREE.Group; gallery: GalleryHandles } {
  const s = STUDIO;
  const g = new THREE.Group();
  g.position.set(s.cx, 0, s.cz);
  const t = 0.3;
  const wall = std({ color: 0xc98a62, roughness: 0.9 }, u);
  const trim = std({ color: 0x4a4038, roughness: 0.9 }, u);
  const fx = s.w / 2;
  const fz = s.d / 2;

  // Ground-floor walls
  const segW = (s.w - s.doorW) / 2;
  g.add(box(segW, FLOOR_H, t, wall, -(s.doorW / 2 + segW / 2), FLOOR_H / 2, fz - t / 2));
  g.add(box(segW, FLOOR_H, t, wall, s.doorW / 2 + segW / 2, FLOOR_H / 2, fz - t / 2));
  g.add(box(s.doorW, FLOOR_H - s.doorH, t, wall, 0, s.doorH + (FLOOR_H - s.doorH) / 2, fz - t / 2));
  g.add(box(s.w, FLOOR_H, t, wall, 0, FLOOR_H / 2, -fz + t / 2));
  g.add(box(t, FLOOR_H, s.d, wall, -fx + t / 2, FLOOR_H / 2, 0));
  g.add(box(t, FLOOR_H, s.d, wall, fx - t / 2, FLOOR_H / 2, 0));

  // Exterior facade wall art mount beside entrance
  const extMount = createWallArtMount("Art Studio Wall", "Art Studio", "art_studio_wall", u);
  extMount.group.position.set(3.5, 1.85, fz + 0.05);
  g.add(extMount.group);
  wallArtSlots.push(extMount.slot);

  // Interior floor, ceiling and the upper loft floor (solid)
  g.add(box(s.w - 0.4, 0.06, s.d - 0.4, std({ color: 0xb08a66, roughness: 0.85 }, u), 0, 0.03, 0, false));
  g.add(box(s.w, FLOOR_H, s.d, wall, 0, FLOOR_H + FLOOR_H / 2, 0));
  g.add(box(s.w + 0.14, 0.14, s.d + 0.14, trim, 0, FLOOR_H, 0, false));
  g.add(box(s.w + 0.5, 0.25, s.d + 0.5, trim, 0, 2 * FLOOR_H + 0.12, 0));
  // Rooftop skylight
  g.add(box(3.0, 0.6, 2.4, std({ color: 0x9fc4d4, roughness: 0.2, metalness: 0.3 }, u), 2, 2 * FLOOR_H + 0.55, -1));

  // Windows (front pieces beside the door, plus upper floor all around)
  const geo = new THREE.BoxGeometry(1.4, 1.5, 0.1);
  const addWin = (x: number, y: number, z: number, ry: number) => {
    const m = new THREE.Mesh(geo, mats.window);
    m.position.set(x, y, z);
    m.rotation.y = ry;
    g.add(m);
  };
  // Only add ground-floor front windows on the left side ([-4.6, -2.8]);
  // omit right side (+2.8, +4.6) so the exterior wall art display at +3.5 is never on a window!
  for (const wx of [-4.6, -2.8]) addWin(wx, 1.9, fz + 0.02, 0);
  for (const wx of [-4.5, -1.5, 1.5, 4.5]) {
    addWin(wx, FLOOR_H + 1.9, fz + 0.02, 0);
    addWin(wx, FLOOR_H + 1.9, -fz - 0.02, Math.PI);
  }
  for (const wz of [-2.5, 0, 2.5]) {
    addWin(fx + 0.02, FLOOR_H + 1.9, wz, Math.PI / 2);
    addWin(-fx - 0.02, FLOOR_H + 1.9, wz, -Math.PI / 2);
    addWin(fx + 0.02, 1.9, wz, Math.PI / 2);
    addWin(-fx - 0.02, 1.9, wz, -Math.PI / 2);
  }

  const sign = signMesh("ART STUDIO", 5.2, "#3b2a5a", "#ffd9a8");
  sign.position.set(0, FLOOR_H + 0.55 + 0.3, fz + 0.08);
  g.add(sign);
  const awning = box(s.doorW + 1.6, 0.12, 1.3, std({ color: 0x7a4fb5, roughness: 0.8 }, u), 0, 2.9, fz + 0.65);
  awning.rotation.x = 0.18;
  g.add(awning);

  // Interior: warm light, easel, paint table, rug
  const light = new THREE.PointLight(0xffe2b0, 1.1, 16);
  light.position.set(0, 2.8, 0);
  g.add(light);

  g.add(box(4, 0.03, 3, std({ color: 0x7a4fb5, roughness: 1 }, u), 1.5, 0.08, 1.0, false));

  const ex = s.easelX - s.cx;
  const ez = s.easelZ - s.cz;
  const wood = std({ color: 0x8a6a4a, roughness: 0.85 }, u);
  for (const lx of [-0.45, 0.45]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 1.9, 0.07), wood);
    leg.position.set(ex + lx, 0.95, ez);
    leg.rotation.z = lx > 0 ? -0.08 : 0.08;
    leg.castShadow = true;
    g.add(leg);
  }
  const backLeg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 1.8, 0.07), wood);
  backLeg.position.set(ex, 0.9, ez - 0.4);
  backLeg.rotation.x = 0.25;
  g.add(backLeg);
  g.add(box(1.3, 1.3, 0.06, std({ color: 0xf4efe4, roughness: 0.9 }, u), ex, 1.45, ez + 0.1, false));
  const easelCanvas = new THREE.Mesh(
    new THREE.PlaneGeometry(1.1, 1.1),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
  );
  easelCanvas.position.set(ex, 1.45, ez + 0.14);
  g.add(easelCanvas);
  wallArtSlots.push({
    id: "art_studio_easel",
    name: "Art Studio Easel",
    buildingName: "Art Studio",
    canvasMesh: easelCanvas,
    group: g,
  });

  // Paint table with jars
  g.add(box(1.6, 0.9, 0.7, wood, 0.8, 0.45, -fz + 0.9));
  const jarColors = [0xc96442, 0x3b7ea1, 0xd9a036, 0x4d7c57];
  jarColors.forEach((c, i) => {
    const jar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.1, 0.22, 8),
      std({ color: c, roughness: 0.5 }, u)
    );
    jar.position.set(0.3 + i * 0.35, 1.01, -fz + 0.9);
    g.add(jar);
  });

  // Gallery wall frames (back wall + right wall)
  const frames: THREE.Mesh[] = [];
  const frameMat = std({ color: 0x3a2a1a, roughness: 0.7 }, u);
  let frameIdx = 0;
  const addFrame = (x: number, y: number, z: number, ry: number) => {
    const fr = box(1.5, 1.5, 0.08, frameMat, 0, 0, 0, false);
    const holder = new THREE.Group();
    holder.position.set(x, y, z);
    holder.rotation.y = ry;
    holder.add(fr);
    const pic = new THREE.Mesh(
      new THREE.PlaneGeometry(1.25, 1.25),
      new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
    );
    pic.position.z = 0.05;
    holder.add(pic);
    g.add(holder);
    frames.push(pic);
    wallArtSlots.push({
      id: `art_studio_frame_${frameIdx}`,
      name: `Art Studio Gallery #${frameIdx + 1}`,
      buildingName: "Art Studio",
      canvasMesh: pic,
      group: holder,
    });
    frameIdx++;
  };
  const backZ = -fz + t + 0.05;
  for (const wx of [-1.4, 1.4, 4.2]) addFrame(wx, 2.0, backZ, 0);
  addFrame(fx - t - 0.05, 2.0, 0.5, -Math.PI / 2);

  return { group: g, gallery: { easelCanvas, frames } };
}

// ---------------------------------------------------------------------------
// Whole-town builder
// ---------------------------------------------------------------------------
export interface TownBuild {
  townGroup: THREE.Group;
  gallery: GalleryHandles;
  wallArtSlots: WallArtSlot[];
  windowMaterials: THREE.MeshStandardMaterial[];
}

export function buildTown(u: DitherUniforms): TownBuild {
  const town = new THREE.Group();
  const wallArtSlots: WallArtSlot[] = [];
  const windowMat = new THREE.MeshStandardMaterial({
    color: 0x9fc4d4,
    emissive: 0xffd98a,
    emissiveIntensity: 0,
    roughness: 0.25,
    metalness: 0.3,
  });
  const mats: TownMaterials = { window: windowMat, windows: [windowMat] };

  // Ground slab + grass
  const grass = new THREE.Mesh(
    new THREE.BoxGeometry(TOWN_HALF * 2, 0.8, TOWN_HALF * 2),
    std({ color: 0x86a266, roughness: 1 }, u)
  );
  grass.position.y = -0.4;
  grass.receiveShadow = true;
  town.add(grass);
  const baseSlab = new THREE.Mesh(
    new THREE.BoxGeometry(TOWN_HALF * 2 + 1.2, 0.6, TOWN_HALF * 2 + 1.2),
    std({ color: 0x4a4a46, roughness: 1 }, u)
  );
  baseSlab.position.y = -0.9;
  town.add(baseSlab);

  const asphalt = std({ color: 0x3c3f42, roughness: 0.95 }, u);
  const paving = std({ color: 0xb9b2a3, roughness: 0.95 }, u);
  const stripe = std({ color: 0xe8e2cf, roughness: 0.9 }, u);

  // Roads (E-W and N-S; N-S ends before the south gate plaza)
  town.add(box(TOWN_HALF * 2, 0.04, ROAD_HALF * 2, asphalt, 0, 0.02, 0, false));
  town.add(box(ROAD_HALF * 2, 0.04, TOWN_HALF * 2, asphalt, 0, 0.021, 0, false));
  // Sidewalks flush with roads
  const sw = SIDEWALK_W;
  const swOff = ROAD_HALF + sw / 2;
  for (const sgn of [-1, 1]) {
    town.add(box(TOWN_HALF * 2, 0.045, sw, paving, 0, 0.0225, sgn * swOff, false));
    town.add(box(sw, 0.046, TOWN_HALF * 2, paving, sgn * swOff, 0.023, 0, false));
  }
  // Center plaza paving over the intersection edges + south gate plaza
  town.add(box(14, 0.05, 6, paving, 0, 0.055, 25.5, false));

  // Dashed center lines + crosswalks (merged into 1 draw call)
  const stripeGeos: THREE.BufferGeometry[] = [];
  const pushBox = (
    list: THREE.BufferGeometry[],
    w: number,
    h: number,
    d: number,
    x: number,
    y: number,
    z: number
  ) => {
    const bg = new THREE.BoxGeometry(w, h, d);
    bg.translate(x, y, z);
    list.push(bg);
  };
  for (let x = -TOWN_HALF + 1.5; x < TOWN_HALF; x += 3) {
    if (Math.abs(x) < ROAD_HALF + 1) continue;
    pushBox(stripeGeos, 1.4, 0.05, 0.14, x, 0.045, 0);
  }
  for (let z = -TOWN_HALF + 1.5; z < TOWN_HALF - 6; z += 3) {
    if (Math.abs(z) < ROAD_HALF + 1) continue;
    pushBox(stripeGeos, 0.14, 0.05, 1.4, 0, 0.046, z);
  }
  for (const sgn of [-1, 1]) {
    for (let i = -2; i <= 2; i++) {
      pushBox(stripeGeos, 0.5, 0.05, 2.2, i * 1.0, 0.047, sgn * (ROAD_HALF + 2.3));
      pushBox(stripeGeos, 2.2, 0.05, 0.5, sgn * (ROAD_HALF + 2.3), 0.047, i * 1.0);
    }
  }
  if (stripeGeos.length > 0) {
    const stripeMesh = new THREE.Mesh(mergeAndDispose(stripeGeos), stripe);
    stripeMesh.receiveShadow = true;
    town.add(stripeMesh);
  }

  // Lot pads for future structures (merged)
  const lotPad = std({ color: 0xcfc7b4, roughness: 0.95 }, u);
  const lotEdge = std({ color: 0xe9e0c8, roughness: 0.9 }, u);
  const padGeos: THREE.BufferGeometry[] = [];
  const edgeGeos: THREE.BufferGeometry[] = [];
  for (const slot of LOT_SLOTS) {
    pushBox(padGeos, 6, 0.05, 6, slot.x, 0.025, slot.z);
    for (const [ox, oz, sx, sz] of [
      [0, -3, 6, 0.12],
      [0, 3, 6, 0.12],
      [-3, 0, 0.12, 6],
      [3, 0, 0.12, 6],
    ] as Array<[number, number, number, number]>) {
      pushBox(edgeGeos, sx, 0.06, sz, slot.x + ox, 0.03, slot.z + oz);
    }
  }
  for (const p of SOLAR_SLOTS) pushBox(padGeos, 4.2, 0.05, 4.2, p.x, 0.025, p.z);
  const padMesh = new THREE.Mesh(mergeAndDispose(padGeos), lotPad);
  padMesh.receiveShadow = true;
  town.add(padMesh);
  town.add(new THREE.Mesh(mergeAndDispose(edgeGeos), lotEdge));

  // Buildings
  for (const spec of BUILDINGS) town.add(buildBuilding(spec, u, mats, wallArtSlots));
  const studio = buildArtStudioBuilding(u, mats, wallArtSlots);
  town.add(studio.group);

  // Hedges around the town
  const hedgeMat = std({ color: 0x4d7c57, roughness: 1 }, u);
  const edge = TOWN_HALF - 0.6;
  town.add(box(TOWN_HALF * 2, 0.9, 0.8, hedgeMat, 0, 0.45, -edge));
  town.add(box(TOWN_HALF * 2, 0.9, 0.8, hedgeMat, 0, 0.45, edge));
  town.add(box(0.8, 0.9, TOWN_HALF * 2, hedgeMat, -edge, 0.45, 0));
  town.add(box(0.8, 0.9, TOWN_HALF * 2, hedgeMat, edge, 0.45, 0));

  // Trees
  const trunkMat = std({ color: 0x6f523b, roughness: 1 }, u);
  const leafMats = [
    std({ color: 0x4d7c57, roughness: 1 }, u),
    std({ color: 0x5e8c4a, roughness: 1 }, u),
    std({ color: 0x6b9a52, roughness: 1 }, u),
  ];
  TREE_SPOTS.forEach(([tx, tz], i) => {
    const tg = new THREE.Group();
    tg.position.set(tx, 0, tz);
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.22, 1.6, 6), trunkMat);
    trunk.position.y = 0.8;
    trunk.castShadow = true;
    tg.add(trunk);
    const crown = new THREE.Mesh(new THREE.IcosahedronGeometry(1.15 + (i % 3) * 0.12, 0), leafMats[i % 3]);
    crown.position.y = 2.3;
    crown.castShadow = true;
    tg.add(crown);
    town.add(tg);
  });

  // Benches
  const benchWood = std({ color: 0x8a6a4a, roughness: 0.9 }, u);
  for (const b of BENCH_SPOTS) {
    const bg = new THREE.Group();
    bg.position.set(b.x, 0, b.z);
    bg.rotation.y = b.rot;
    bg.add(box(1.6, 0.08, 0.5, benchWood, 0, 0.5, 0));
    bg.add(box(1.6, 0.5, 0.06, benchWood, 0, 0.8, -0.22));
    for (const lx of [-0.7, 0.7]) bg.add(box(0.08, 0.5, 0.4, trunkMat, lx, 0.25, 0));
    town.add(bg);
  }

  // Street lamps along both roads (merged into 2 draw calls)
  const lampMat = std({ color: 0x35393c, roughness: 0.6, metalness: 0.4 }, u);
  const lampGlow = new THREE.MeshStandardMaterial({ color: 0xfff1c4, emissive: 0xffd98a, emissiveIntensity: 0.4 });
  const lampPositions: Array<[number, number]> = [];
  for (const x of [-24, -14, 14, 24]) lampPositions.push([x, ROAD_HALF + 0.3], [x, -(ROAD_HALF + 0.3)]);
  for (const z of [-24, -14, 14, 20]) lampPositions.push([ROAD_HALF + 0.3, z], [-(ROAD_HALF + 0.3), z]);
  const poleGeos: THREE.BufferGeometry[] = [];
  const headGeos: THREE.BufferGeometry[] = [];
  for (const [lx, lz] of lampPositions) {
    pushBox(poleGeos, 0.12, 3.6, 0.12, lx, 1.8, lz);
    const hg = new THREE.SphereGeometry(0.22, 8, 6);
    hg.translate(lx, 3.75, lz);
    headGeos.push(hg);
  }
  town.add(new THREE.Mesh(mergeAndDispose(poleGeos), lampMat));
  town.add(new THREE.Mesh(mergeAndDispose(headGeos), lampGlow));
  (town.userData as { lampGlow?: THREE.MeshStandardMaterial }).lampGlow = lampGlow;

  return {
    townGroup: town,
    gallery: studio.gallery,
    wallArtSlots,
    windowMaterials: mats.windows,
  };
}

// ---------------------------------------------------------------------------
// SVG drawing -> texture
// ---------------------------------------------------------------------------

/** Ensures the root <svg> element explicitly specifies width and height for browser rendering. */
export function normalizeSvgMarkup(svg: string, size = 512): string {
  let s = svg.trim();
  if (!s.includes("xmlns=")) {
    s = s.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  s = s.replace(/<svg\b([^>]*)>/i, (_match, attrs) => {
    const stripped = attrs
      .replace(/\bwidth=[\"\'][^\"\']*[\"\']/gi, "")
      .replace(/\bheight=[\"\'][^\"\']*[\"\']/gi, "")
      .trim();
    return `<svg width="${size}" height="${size}" ${stripped}>`;
  });
  return s;
}

/** Fallback 2D canvas rasterizer for basic vector shapes in case browser image decode is unavailable. */
function renderSvgToCanvas2D(svgMarkup: string, ctx: CanvasRenderingContext2D, size: number): void {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  if (typeof DOMParser === "undefined") return;
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(svgMarkup, "image/svg+xml");
    const svgEl = doc.querySelector("svg");
    if (!svgEl) return;

    let vbW = 100;
    let vbH = 100;
    const vb = svgEl.getAttribute("viewBox");
    if (vb) {
      const parts = vb.trim().split(/[\s,]+/).map(Number);
      if (parts.length === 4 && parts[2] > 0 && parts[3] > 0) {
        vbW = parts[2];
        vbH = parts[3];
      }
    }
    const sx = size / vbW;
    const sy = size / vbH;

    const elements = doc.querySelectorAll("rect, circle, ellipse, line, polygon, polyline, path");
    elements.forEach((el) => {
      const tag = el.tagName.toLowerCase();
      const fill = el.getAttribute("fill") || "none";
      const stroke = el.getAttribute("stroke") || "none";
      const strokeWidth = parseFloat(el.getAttribute("stroke-width") || "1") * sx;

      ctx.save();
      if (fill !== "none") ctx.fillStyle = fill;
      if (stroke !== "none") {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = strokeWidth;
      }

      ctx.beginPath();
      if (tag === "rect") {
        const x = parseFloat(el.getAttribute("x") || "0") * sx;
        const y = parseFloat(el.getAttribute("y") || "0") * sy;
        const w = parseFloat(el.getAttribute("width") || `${vbW}`) * sx;
        const h = parseFloat(el.getAttribute("height") || `${vbH}`) * sy;
        if (fill !== "none") ctx.fillRect(x, y, w, h);
        if (stroke !== "none") ctx.strokeRect(x, y, w, h);
      } else if (tag === "circle") {
        const cx = parseFloat(el.getAttribute("cx") || "0") * sx;
        const cy = parseFloat(el.getAttribute("cy") || "0") * sy;
        const r = parseFloat(el.getAttribute("r") || "0") * sx;
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        if (fill !== "none") ctx.fill();
        if (stroke !== "none") ctx.stroke();
      } else if (tag === "ellipse") {
        const cx = parseFloat(el.getAttribute("cx") || "0") * sx;
        const cy = parseFloat(el.getAttribute("cy") || "0") * sy;
        const rx = parseFloat(el.getAttribute("rx") || "0") * sx;
        const ry = parseFloat(el.getAttribute("ry") || "0") * sy;
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        if (fill !== "none") ctx.fill();
        if (stroke !== "none") ctx.stroke();
      } else if (tag === "line") {
        const x1 = parseFloat(el.getAttribute("x1") || "0") * sx;
        const y1 = parseFloat(el.getAttribute("y1") || "0") * sy;
        const x2 = parseFloat(el.getAttribute("x2") || "0") * sx;
        const y2 = parseFloat(el.getAttribute("y2") || "0") * sy;
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        if (stroke !== "none") ctx.stroke();
      } else if (tag === "path") {
        const d = el.getAttribute("d") || "";
        if (d && typeof Path2D !== "undefined") {
          ctx.scale(sx, sy);
          const p = new Path2D(d);
          if (fill !== "none") ctx.fill(p);
          if (stroke !== "none") ctx.stroke(p);
        }
      }
      ctx.restore();
    });
  } catch {
    // Keep white background canvas
  }
}

export function loadSvgTexture(svg: string, size = 512): Promise<THREE.CanvasTexture> {
  return new Promise((resolve) => {
    const makeDirect2DTexture = () => {
      const c = document.createElement("canvas");
      c.width = size;
      c.height = size;
      const ctx = c.getContext("2d");
      if (ctx) renderSvgToCanvas2D(svg, ctx, size);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.generateMipmaps = true;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.needsUpdate = true;
      resolve(tex);
    };

    if (typeof window === "undefined" || typeof Image === "undefined") {
      makeDirect2DTexture();
      return;
    }

    try {
      const markup = normalizeSvgMarkup(svg, size);
      const blob = new Blob([markup], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      let finished = false;

      const finishWithImage = () => {
        if (finished) return;
        finished = true;
        try {
          const c = document.createElement("canvas");
          c.width = size;
          c.height = size;
          const ctx = c.getContext("2d");
          if (!ctx) {
            URL.revokeObjectURL(url);
            makeDirect2DTexture();
            return;
          }
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(0, 0, size, size);
          ctx.drawImage(img, 0, 0, size, size);
          URL.revokeObjectURL(url);
          const tex = new THREE.CanvasTexture(c);
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.generateMipmaps = true;
          tex.minFilter = THREE.LinearMipmapLinearFilter;
          tex.magFilter = THREE.LinearFilter;
          tex.needsUpdate = true;
          resolve(tex);
        } catch {
          URL.revokeObjectURL(url);
          makeDirect2DTexture();
        }
      };

      img.onload = () => {
        if ("decode" in img) {
          img.decode().then(finishWithImage).catch(finishWithImage);
        } else {
          finishWithImage();
        }
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        makeDirect2DTexture();
      };
      img.src = url;
    } catch {
      makeDirect2DTexture();
    }
  });
}
