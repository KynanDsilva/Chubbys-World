import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { ditherMaterial, type DitherUniforms } from "./scrapyardScene";

export const CHUBBY_MODEL_URL = "/assets/models/chubby.fbx";
export const CHUBBY_TARGET_HEIGHT = 1.5;

export interface ChubbyModel {
  root: THREE.Group;
  mixer: THREE.AnimationMixer;
  /** Short lowercase clip names, e.g. "idle", "walking", "dance", "wave", "sitting". */
  clips: string[];
  /** Crossfades to the named clip (no-op if missing or already active). */
  play: (name: string, fade?: number) => void;
  current: () => string | null;
}

function shortClipName(name: string): string {
  const last = name.split("|").pop() || name;
  return last.replace(/^Robot_/i, "").toLowerCase();
}

/** Loads the FBX robot, normalizes its size/origin, and wires up an animation mixer. */
export function loadChubbyModel(url: string, u: DitherUniforms): Promise<ChubbyModel> {
  return new Promise((resolve, reject) => {
    new FBXLoader().load(
      url,
      (fbx) => {
        try {
          fbx.traverse((child) => {
            const mesh = child as THREE.Mesh;
            if (mesh.isMesh) {
              mesh.castShadow = true;
              mesh.receiveShadow = true;
              // Prevent skinned hand/arm meshes from being frustum-culled mid-animation.
              mesh.frustumCulled = false;
              const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
              for (const m of mats) {
                if (!m) continue;
                // Blender FBX exports often set material opacity=0 + transparent=true.
                m.opacity = 1.0;
                m.transparent = false;
                m.needsUpdate = true;
                ditherMaterial(m, u);
              }
            }
          });

          // Normalize: scale to target height, stand on y=0, centered on x/z.
          const wrapper = new THREE.Group();
          wrapper.add(fbx);
          fbx.updateMatrixWorld(true);
          const box = new THREE.Box3().setFromObject(fbx);
          const size = new THREE.Vector3();
          box.getSize(size);
          const k = size.y > 1e-6 ? CHUBBY_TARGET_HEIGHT / size.y : 1;
          fbx.scale.multiplyScalar(k);
          fbx.updateMatrixWorld(true);
          const box2 = new THREE.Box3().setFromObject(fbx);
          const center = new THREE.Vector3();
          box2.getCenter(center);
          fbx.position.x -= center.x;
          fbx.position.z -= center.z;
          fbx.position.y -= box2.min.y;

          const mixer = new THREE.AnimationMixer(fbx);
          const actions = new Map<string, THREE.AnimationAction>();
          for (const clip of fbx.animations) {
            actions.set(shortClipName(clip.name), mixer.clipAction(clip));
          }

          let active: THREE.AnimationAction | null = null;
          let activeName: string | null = null;
          const play = (name: string, fade = 0.25) => {
            const next = actions.get(name);
            if (!next || name === activeName) return;
            next.reset().fadeIn(fade).play();
            if (active) active.fadeOut(fade);
            active = next;
            activeName = name;
          };

          resolve({
            root: wrapper,
            mixer,
            clips: Array.from(actions.keys()),
            play,
            current: () => activeName,
          });
        } catch (err) {
          reject(err);
        }
      },
      undefined,
      (err) => reject(err)
    );
  });
}

/** Maps Chubby's current activity to the best available animation clip. */
export function pickClip(moving: boolean, category: string, available: string[]): string {
  const has = (n: string) => available.includes(n);
  if (moving) return has("walking") ? "walking" : has("running") ? "running" : "idle";
  const wanted: Record<string, string[]> = {
    rest: ["sitting", "idle"],
    recharge: ["sitting", "idle"],
    play: ["dance", "jump", "idle"],
    social: ["wave", "thumbsup", "idle"],
    explore: ["no", "idle"],
    reflect: ["standing", "idle"],
    draw: ["thumbsup", "yes", "idle"],
    make: ["yes", "punch", "idle"],
    maintain: ["yes", "idle"],
  };
  for (const n of wanted[category] || ["idle"]) if (has(n)) return n;
  return available[0] || "idle";
}
