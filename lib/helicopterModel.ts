import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { ditherMaterial, type DitherUniforms } from "./scrapyardScene";

export const HELICOPTER_MODEL_URL = "/assets/models/Helecopter.fbx";

/**
 * Real-world scale. The town uses ~1 unit = 1 m (floors are 3.3 tall, Chubby is 1.5 tall).
 * The model is an Airbus EC135: ~12.2 m long overall (rotors included), ~3.5 m tall.
 */
export const HELICOPTER_LENGTH_M = 12.0;

/** Where the helipad sits: the open lot east of the north road, in front of the Library. */
export const HELIPAD = {
  x: 21.5,
  z: -10.5,
  radius: 5.6,
  // Where Chubby stands to check it out (beside the cabin door, facing the helicopter).
  standX: 21.0,
  standZ: -6.9,
};

export interface HelicopterModel {
  root: THREE.Group;
  /** Spin the rotors; `speed` is 0..1 (0 = parked, 1 = full idle spin). */
  update: (dt: number, speed: number) => void;
}

// 1x1 transparent PNG so the FBX's missing .bmp texture references never hit the network.
const BLANK_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

/** Flat livery (the source textures are not shipped). */
const LIVERY = {
  body: 0xd94b2b, // rescue red
  trim: 0xf2eee6, // off-white
  glass: 0x24343f, // tinted canopy
  rotor: 0x2b2d30, // dark blades
};

function pickBodyColor(matName: string, index: number): number {
  const n = matName.toLowerCase();
  if (n.includes("glass") || n.includes("window") || n.includes("canopy")) return LIVERY.glass;
  // Untextured material on this model is the canopy glass.
  if (!n.includes(".bmp")) return LIVERY.glass;
  return index % 2 === 0 ? LIVERY.body : LIVERY.trim;
}

/** Re-centers a mesh's geometry on its bounding-box center so it spins around its own hub. */
function recenterPivot(mesh: THREE.Mesh): void {
  const geo = mesh.geometry;
  geo.computeBoundingBox();
  const bb = geo.boundingBox;
  if (!bb) return;
  const c = new THREE.Vector3();
  bb.getCenter(c);
  geo.translate(-c.x, -c.y, -c.z);
  // Move the mesh so the world-space geometry stays where it was.
  c.multiply(mesh.scale).applyQuaternion(mesh.quaternion);
  mesh.position.add(c);
}

/** The thinnest local axis of a rotor's geometry is the axis it spins around. */
function spinAxis(mesh: THREE.Mesh): THREE.Vector3 {
  const geo = mesh.geometry;
  geo.computeBoundingBox();
  const s = new THREE.Vector3();
  geo.boundingBox?.getSize(s);
  if (s.x <= s.y && s.x <= s.z) return new THREE.Vector3(1, 0, 0);
  if (s.y <= s.x && s.y <= s.z) return new THREE.Vector3(0, 1, 0);
  return new THREE.Vector3(0, 0, 1);
}

export function loadHelicopterModel(url: string, u: DitherUniforms): Promise<HelicopterModel> {
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((reqUrl) => (/\.(bmp|png|jpe?g|tga|dds)$/i.test(reqUrl) ? BLANK_PNG : reqUrl));

  return new Promise((resolve, reject) => {
    new FBXLoader(manager).load(
      url,
      (fbx) => {
        try {
          const rotors: Array<{ mesh: THREE.Mesh; axis: THREE.Vector3; rate: number }> = [];

          fbx.traverse((child) => {
            const mesh = child as THREE.Mesh;
            if (!mesh.isMesh) return;
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            const isRotor = /rotor/i.test(mesh.name);

            const srcMats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            const newMats = srcMats.map((m, i) => {
              const name = m?.name || "";
              const color = isRotor ? LIVERY.rotor : pickBodyColor(name, i);
              const isGlass = color === LIVERY.glass;
              const mat = new THREE.MeshStandardMaterial({
                color,
                roughness: isGlass ? 0.15 : isRotor ? 0.7 : 0.45,
                metalness: isGlass ? 0.5 : 0.25,
                flatShading: true,
                side: THREE.DoubleSide,
              });
              mat.name = name;
              m?.dispose();
              return ditherMaterial(mat, u);
            });
            mesh.material = Array.isArray(mesh.material) ? newMats : newMats[0];

            if (isRotor) {
              recenterPivot(mesh);
              // static_rotor2 is the small tail rotor: spins faster.
              const isTail = /2$/.test(mesh.name);
              rotors.push({ mesh, axis: spinAxis(mesh), rate: isTail ? 2.6 : 1.0 });
            }
          });

          const wrapper = new THREE.Group();
          wrapper.name = "helicopter";
          wrapper.add(fbx);
          fbx.updateMatrixWorld(true);

          // Lay the fuselage along world X so it fits the helipad.
          const box = new THREE.Box3().setFromObject(fbx);
          const size = new THREE.Vector3();
          box.getSize(size);
          if (size.z > size.x) fbx.rotation.y += Math.PI / 2;
          fbx.updateMatrixWorld(true);

          // Scale to real-world length, then stand it on the ground, centered.
          box.setFromObject(fbx);
          box.getSize(size);
          const length = Math.max(size.x, size.z);
          const k = length > 1e-6 ? HELICOPTER_LENGTH_M / length : 1;
          fbx.scale.multiplyScalar(k);
          fbx.updateMatrixWorld(true);
          box.setFromObject(fbx);
          const center = new THREE.Vector3();
          box.getCenter(center);
          fbx.position.x -= center.x;
          fbx.position.z -= center.z;
          fbx.position.y -= box.min.y;
          fbx.updateMatrixWorld(true);
          box.setFromObject(fbx);
          box.getSize(size);
          console.info(
            `Helicopter loaded: ${size.x.toFixed(1)}m long x ${size.z.toFixed(1)}m wide x ${size.y.toFixed(1)}m tall`
          );

          let angle = 0;
          const update = (dt: number, speed: number) => {
            if (speed <= 0.001) return;
            angle = dt * speed * 9.0;
            for (const r of rotors) r.mesh.rotateOnAxis(r.axis, angle * r.rate);
          };

          resolve({ root: wrapper, update });
        } catch (err) {
          reject(err);
        }
      },
      undefined,
      (err) => reject(err)
    );
  });
}

/** Painted helipad: dark asphalt disc, yellow ring and a white "H". */
export function buildHelipad(u: DitherUniforms): THREE.Group {
  const g = new THREE.Group();
  g.position.set(HELIPAD.x, 0, HELIPAD.z);
  const r = HELIPAD.radius;

  const pad = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, 0.08, 40),
    ditherMaterial(new THREE.MeshStandardMaterial({ color: 0x3a3d40, roughness: 0.95, flatShading: true }), u)
  );
  pad.position.y = 0.04;
  pad.receiveShadow = true;
  g.add(pad);

  const paint = (color: number) => new THREE.MeshBasicMaterial({ color });
  const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.55, r - 0.3, 48), paint(0xf2c230));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.085;
  g.add(ring);

  // "H" made of three white bars, readable from the street (+z side).
  const white = paint(0xf2eee6);
  const bar = (w: number, d: number, x: number, z: number) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), white);
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, 0.086, z);
    g.add(m);
  };
  bar(0.5, 3.2, -1.1, 0);
  bar(0.5, 3.2, 1.1, 0);
  bar(1.7, 0.5, 0, 0);

  // Small edge lights around the pad.
  const lightMat = new THREE.MeshStandardMaterial({ color: 0xfff1c4, emissive: 0xffd98a, emissiveIntensity: 0.5 });
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const l = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.14, 8), lightMat);
    l.position.set(Math.cos(a) * (r + 0.15), 0.07, Math.sin(a) * (r + 0.15));
    g.add(l);
  }
  return g;
}
