"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  buildGateProps,
  classifyStructureType,
  createProceduralChubby,
  createStructureModel,
  ditherMaterial,
  disposeSceneSubtree,
  SAFE_BOUNDS,
  type ChubbyParts,
  type DitherUniforms,
  type InteractType,
  type StructureKind,
  type StructureSlot,
} from "@/lib/scrapyardScene";
import {
  buildTown,
  findPath,
  loadSvgTexture,
  LOT_SLOTS,
  resolveCollisions,
  SOLAR_SLOTS,
  STUDIO,
  WANDER_POINTS,
  type WallArtSlot,
} from "@/lib/townScene";
import {
  CHUBBY_MODEL_URL,
  loadChubbyModel,
  pickClip,
  type ChubbyModel,
} from "@/lib/chubbyModel";
import {
  buildHelipad,
  HELICOPTER_MODEL_URL,
  HELIPAD,
  loadHelicopterModel,
  type HelicopterModel,
} from "@/lib/helicopterModel";

type ModalId = "modal-diary" | "modal-offering" | "modal-blueprints" | "modal-status" | "modal-dev";
type CameraMode = "first_person" | "orbit";
type DiaryTab = "diary" | "memories" | "actions" | "gallery";
type TimeOfDayMode = "auto" | "dawn" | "noon" | "dusk" | "night" | "custom";

function formatHourShort(h: number): string {
  const norm = ((h % 24) + 24) % 24;
  const hours = Math.floor(norm);
  const minutes = Math.floor((norm - hours) * 60);
  const period = hours >= 12 ? "PM" : "AM";
  const displayHours = hours % 12 === 0 ? 12 : hours % 12;
  return `${displayHours}:${minutes.toString().padStart(2, "0")} ${period}`;
}

interface WorldStateNumbers {
  tick: number;
  sim_time: string;
  battery: number;
  curiosity: number;
  joy: number;
  calm: number;
  belonging: number;
  energy: number;
  mood: number;
  mood_modifier: number;
}

interface ActionEntry {
  id: number;
  tick: number;
  action: string;
  detail: string;
  thought: string;
  outcome?: { outcome?: string };
}

interface PendingRequest {
  id: number;
  name: string;
  purpose: string;
  why: string;
  priority: number;
}

interface StructureItem {
  id: number;
  name: string;
  unlocks?: string;
}

interface VisitorMessage {
  received_tick: number;
  text: string;
  drawing?: string;
  read?: boolean;
}

interface DiaryEntry {
  tick: number;
  sim_time: string;
  text: string;
}

interface MemoryEntry {
  tick_range: string;
  summary: string;
}

interface DrawingItem {
  id: number;
  tick: number;
  title: string;
  svg: string;
  location?: string;
}

interface LoopInfo {
  running?: boolean;
  policy?: string;
  effective_policy?: string;
  tick_in_progress?: boolean;
}

interface WorldPayload {
  state: WorldStateNumbers;
  loop?: LoopInfo;
  is_night?: boolean;
  is_dusk?: boolean;
  sim_hour?: number;
  latest_action?: ActionEntry | null;
  active_events?: string[];
  pending_requests?: PendingRequest[];
  structures?: StructureItem[];
  unread_messages?: VisitorMessage[];
  recent_messages?: VisitorMessage[];
  recent_diary?: DiaryEntry[];
  memories?: MemoryEntry[];
  recent_actions?: ActionEntry[];
  drawings?: DrawingItem[];
}

const INTERACT_LABELS: Record<InteractType, string> = {
  status: "[E / Click] Inspect Chubby",
  diary: "[E / Click] Read Chubby's Field Diary",
  offering: "[E / Click] Leave an Offering (Note & Sketch)",
  blueprints: "[E / Click] Review Structure Blueprints",
};

const INTERACT_MODAL: Record<InteractType, ModalId> = {
  status: "modal-status",
  diary: "modal-diary",
  offering: "modal-offering",
  blueprints: "modal-blueprints",
};

const SWATCH_COLORS: Array<{ color: string; title: string; isPaper?: boolean }> = [
  { color: "#24201c", title: "Charcoal Ink" },
  { color: "#c96442", title: "Rust Clay" },
  { color: "#4d7c57", title: "Moss Green" },
  { color: "#3b7ea1", title: "Sky Blue" },
  { color: "#d9a036", title: "Ochre Sun" },
  { color: "#f4efe4", title: "Paper Eraser", isPaper: true },
];

async function apiJson<T = Record<string, unknown>>(
  url: string,
  method = "GET",
  body: unknown = null
): Promise<T> {
  const opts: RequestInit = { method, headers: {} };
  if (body !== null) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const data = (await res.json()) as T & { ok?: boolean; error?: string };
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function describeFeeling(st?: WorldStateNumbers): string {
  if (!st) return "";
  const drives: Array<[string, number]> = [
    ["curious", Number(st.curiosity)],
    ["cheerful", Number(st.joy)],
    ["calm", Number(st.calm)],
    ["connected", Number(st.belonging)],
  ];
  drives.sort((a, b) => b[1] - a[1]);
  const batPct = Math.round(Number(st.battery) * 100);
  if (batPct < 30) {
    return `feeling low-power (${batPct}%) • slightly ${drives[0][0]}`;
  }
  return `feeling ${drives[0][0]} & ${drives[1][0]} • battery ${batPct}%`;
}

function clampTarget(x: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(
    Math.max(-SAFE_BOUNDS, Math.min(SAFE_BOUNDS, x)),
    0,
    Math.max(-SAFE_BOUNDS, Math.min(SAFE_BOUNDS, z))
  );
}

/**
 * Full-viewport Three.js scrapyard world + diegetic HUD and tactile paper modals.
 */
export default function ChubbyWorld(): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pixelCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bubbleRef = useRef<HTMLDivElement | null>(null);
  const controlsHintRef = useRef<HTMLDivElement | null>(null);

  const [world, setWorld] = useState<WorldPayload | null>(null);
  const [connected, setConnected] = useState(true);
  const [openModal, setOpenModal] = useState<ModalId | null>(null);
  const [activeTab, setActiveTab] = useState<DiaryTab>("diary");
  const [cameraMode, setCameraModeState] = useState<CameraMode>("first_person");
  const [hoveredInteract, setHoveredInteract] = useState<InteractType | null>(null);
  const [toastMsg, setToastMsg] = useState("");
  const [ditherEnabled, setDitherEnabled] = useState(true);
  const [policyValue, setPolicyValue] = useState("llm");
  const [stepBusy, setStepBusy] = useState(false);
  const [offeringText, setOfferingText] = useState("");
  const [selectedColor, setSelectedColor] = useState("#24201c");
  const [timeOfDayMode, setTimeOfDayMode] = useState<TimeOfDayMode>("auto");
  const [customHour, setCustomHour] = useState<number>(12);
  const [atmospherePanelOpen, setAtmospherePanelOpen] = useState<boolean>(false);

  const openModalRef = useRef<ModalId | null>(null);
  const selectedColorRef = useRef("#24201c");
  const hasPixelEditsRef = useRef(false);
  const toastTimerRef = useRef<number | null>(null);
  const ditherUniformsRef = useRef<DitherUniforms>({ uDitherEnabled: { value: 1.0 } });
  const cameraToggleRef = useRef<(() => void) | null>(null);
  const clearKeysRef = useRef<(() => void) | null>(null);
  const applyWorldRef = useRef<((w: WorldPayload) => void) | null>(null);
  const lightingOverrideRef = useRef<{ mode: TimeOfDayMode; customHour: number }>({
    mode: "auto",
    customHour: 12,
  });

  useEffect(() => {
    openModalRef.current = openModal;
  }, [openModal]);

  useEffect(() => {
    selectedColorRef.current = selectedColor;
  }, [selectedColor]);

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToastMsg("");
    }, 2600);
  }, []);

  const closeAllModals = useCallback(() => {
    setOpenModal(null);
    openModalRef.current = null;
  }, []);

  const triggerOpenModal = useCallback((id: ModalId) => {
    document.exitPointerLock?.();
    clearKeysRef.current?.();
    openModalRef.current = id;
    setOpenModal(id);
  }, []);

  const toggleFullUI = useCallback(() => {
    document.body.classList.toggle("ui-hidden");
    const isHidden = document.body.classList.contains("ui-hidden");
    showToast(isHidden ? "Zen mode active (Press H to show UI)" : "World HUD restored");
  }, [showToast]);

  const clearPixelCanvas = useCallback(() => {
    const pCanvas = pixelCanvasRef.current;
    if (!pCanvas) return;
    const ctx = pCanvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#f4efe4";
    ctx.fillRect(0, 0, 24, 24);
    hasPixelEditsRef.current = false;
  }, []);

  const syncWorldState = useCallback((nextWorld: WorldPayload) => {
    setWorld(nextWorld);
    setConnected(true);
    if (nextWorld.loop?.policy) {
      setPolicyValue(nextWorld.loop.policy);
    }
    applyWorldRef.current?.(nextWorld);
  }, []);

  // Initialize 24x24 pixel canvas and mouse/touch painting listeners.
  useEffect(() => {
    const pCanvas = pixelCanvasRef.current;
    if (!pCanvas) return;
    clearPixelCanvas();

    let isDrawing = false;
    const paintAt = (e: MouseEvent) => {
      const ctx = pCanvas.getContext("2d");
      if (!ctx) return;
      const rect = pCanvas.getBoundingClientRect();
      const px = Math.floor(((e.clientX - rect.left) / rect.width) * 24);
      const py = Math.floor(((e.clientY - rect.top) / rect.height) * 24);
      if (px < 0 || px >= 24 || py < 0 || py >= 24) return;
      ctx.fillStyle = selectedColorRef.current;
      ctx.fillRect(px, py, 1, 1);
      hasPixelEditsRef.current = true;
    };

    const onDown = (e: MouseEvent) => {
      isDrawing = true;
      paintAt(e);
    };
    const onMove = (e: MouseEvent) => {
      if (isDrawing) paintAt(e);
    };
    const onUp = () => {
      isDrawing = false;
    };

    pCanvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    return () => {
      pCanvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [clearPixelCanvas]);

  // Mount Three.js scene, player/Chubby controllers, raycaster, and API poller.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    const ditherUniforms = ditherUniformsRef.current;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xc4d6cf);
    scene.fog = new THREE.Fog(0xc4d6cf, 42, 125);

    const camera = new THREE.PerspectiveCamera(
      68,
      window.innerWidth / window.innerHeight,
      0.1,
      220
    );

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4.5;
    controls.maxDistance = 68.0;
    controls.maxPolarAngle = Math.PI * 0.46;
    controls.target.set(0, 1.5, 0);
    controls.enabled = false;

    const player = {
      x: 0.0,
      y: 1.65,
      z: 22.0,
      yaw: 0.0,
      pitch: -0.05,
      speed: 7.5,
      sprintSpeed: 13.5,
      walkTimer: 0,
    };

    let currentCamMode: CameraMode = "first_person";
    let isPointerLocked = false;
    let isDraggingMouse = false;
    let mouseDragMoved = false;
    let previousMousePosition = { x: 0, y: 0 };

    const keys = {
      forward: false,
      backward: false,
      left: false,
      right: false,
      sprint: false,
    };

    clearKeysRef.current = () => {
      isPointerLocked = false;
      isDraggingMouse = false;
      keys.forward = false;
      keys.backward = false;
      keys.left = false;
      keys.right = false;
      keys.sprint = false;
    };

    const hemiLight = new THREE.HemisphereLight(0xf2efe4, 0x5b6149, 0.9);
    scene.add(hemiLight);

    const sunLight = new THREE.DirectionalLight(0xfff0d4, 1.45);
    sunLight.position.set(18, 30, 16);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(1024, 1024);
    sunLight.shadow.camera.near = 1;
    sunLight.shadow.camera.far = 90;
    sunLight.shadow.camera.left = -34;
    sunLight.shadow.camera.right = 34;
    sunLight.shadow.camera.top = 34;
    sunLight.shadow.camera.bottom = -34;
    sunLight.shadow.bias = -0.0005;
    scene.add(sunLight);

    const interactables: THREE.Object3D[] = [];
    const tagInteractable = (group: THREE.Object3D, type: InteractType) => {
      group.userData.interactType = type;
      interactables.push(group);
    };

    const drawingPreviewCanvas = document.createElement("canvas");
    drawingPreviewCanvas.width = 24;
    drawingPreviewCanvas.height = 24;
    const drawingPreviewCtx = drawingPreviewCanvas.getContext("2d");
    if (drawingPreviewCtx) {
      drawingPreviewCtx.fillStyle = "#f4efe4";
      drawingPreviewCtx.fillRect(0, 0, 24, 24);
    }
    const mailboxDrawingTex = new THREE.CanvasTexture(drawingPreviewCanvas);
    mailboxDrawingTex.magFilter = THREE.NearestFilter;
    mailboxDrawingTex.minFilter = THREE.NearestFilter;
    mailboxDrawingTex.colorSpace = THREE.SRGBColorSpace;

    let mailboxFlagTargetRot = -Math.PI * 0.45;
    let lastRenderedDrawingData = "";

    const { townGroup, gallery, wallArtSlots, windowMaterials } = buildTown(ditherUniforms);
    scene.add(townGroup);

    // Tag all town wall art displays interactable to view the artwork / field diary
    for (const slot of wallArtSlots) {
      tagInteractable(slot.group, "diary");
    }

    // Helipad + parked helicopter (real-world scale next to 1.5 m tall Chubby).
    scene.add(buildHelipad(ditherUniforms));
    let helicopter: HelicopterModel | null = null;
    let rotorSpeed = 0;
    loadHelicopterModel(HELICOPTER_MODEL_URL, ditherUniforms)
      .then((model) => {
        if (disposed) {
          disposeSceneSubtree(model.root);
          return;
        }
        model.root.position.set(HELIPAD.x, 0.08, HELIPAD.z);
        scene.add(model.root);
        helicopter = model;
      })
      .catch((err) => console.warn("Helicopter FBX failed to load:", err));

    const { lotGroup, mailboxFlagPivot, mailboxDrawingMesh, blueprintPaperGroup } =
      buildGateProps(ditherUniforms, mailboxDrawingTex, tagInteractable, mailboxFlagTargetRot);
    scene.add(lotGroup);

    // Chubby robot instance
    const chubbyState: {
      group: THREE.Group;
      parts: ChubbyParts | null;
      fbx: ChubbyModel | null;
      target: THREE.Vector3;
      waypoints: Array<{ x: number; z: number }>;
      yaw: number;
      speed: number;
      activeCategory: string;
      lastActionId: number | null;
      activeStructureSlot: { x: number; z: number; rot: number; kind: StructureKind } | null;
    } = {
      group: new THREE.Group(),
      parts: null,
      fbx: null,
      target: new THREE.Vector3(0, 0, 0),
      waypoints: [],
      yaw: 0,
      speed: 2.35,
      activeCategory: "rest",
      lastActionId: null,
      activeStructureSlot: null,
    };
    tagInteractable(chubbyState.group, "status");
    scene.add(chubbyState.group);

    const setupProceduralChubby = () => {
      disposeSceneSubtree(chubbyState.group);
      chubbyState.group.clear();
      const built = createProceduralChubby(ditherUniforms);
      chubbyState.group.add(built.root);
      chubbyState.parts = built.parts;
      chubbyState.fbx = null;
    };

    setupProceduralChubby();
    loadChubbyModel(CHUBBY_MODEL_URL, ditherUniforms)
      .then((model) => {
        if (disposed) {
          disposeSceneSubtree(model.root);
          return;
        }
        disposeSceneSubtree(chubbyState.group);
        chubbyState.group.clear();
        chubbyState.group.add(model.root);
        chubbyState.parts = null;
        chubbyState.fbx = model;
        model.play("idle", 0);
      })
      .catch((err) => {
        console.warn("FBX load failed, keeping procedural Chubby:", err);
      });

    // Wheel dust puff pool
    interface DustSlot {
      mesh: THREE.Mesh;
      life: number;
      maxLife: number;
      vx: number;
      vz: number;
    }
    const dustPool: DustSlot[] = [];
    const dustGeo = new THREE.IcosahedronGeometry(0.1, 0);
    for (let i = 0; i < 14; i++) {
      const mat = ditherMaterial(
        new THREE.MeshStandardMaterial({
          color: 0xc2b6a2,
          roughness: 1.0,
          flatShading: true,
          transparent: true,
          opacity: 0.0,
        }),
        ditherUniforms
      );
      const mesh = new THREE.Mesh(dustGeo, mat);
      mesh.visible = false;
      scene.add(mesh);
      dustPool.push({ mesh, life: 0, maxLife: 0.55, vx: 0, vz: 0 });
    }
    let dustSpawnCooldown = 0;

    const spawnDustPuff = (x: number, z: number) => {
      const slot = dustPool.find((d) => d.life <= 0);
      if (!slot) return;
      slot.life = slot.maxLife;
      slot.mesh.visible = true;
      slot.mesh.position.set(
        x + (Math.random() - 0.5) * 0.35,
        0.1,
        z + (Math.random() - 0.5) * 0.35
      );
      slot.mesh.scale.setScalar(0.5);
      slot.vx = (Math.random() - 0.5) * 0.3;
      slot.vz = (Math.random() - 0.5) * 0.3;
    };

    const updateDustPuffs = (dt: number) => {
      for (const d of dustPool) {
        if (d.life <= 0) continue;
        d.life -= dt;
        if (d.life <= 0) {
          d.mesh.visible = false;
          continue;
        }
        const t = 1 - d.life / d.maxLife;
        d.mesh.position.y += dt * 0.55;
        d.mesh.position.x += d.vx * dt;
        d.mesh.position.z += d.vz * dt;
        d.mesh.scale.setScalar(0.5 + t * 0.9);
        (d.mesh.material as THREE.MeshStandardMaterial).opacity = (1 - t) * 0.65;
      }
    };

    const structureMeshes = new Map<
      number,
      { group: THREE.Group; slot: StructureSlot; pop: number; data: StructureItem }
    >();

    const isSlotNearUsed = (slot: StructureSlot, usedSlots: StructureSlot[]): boolean =>
      usedSlots.some((u) => Math.hypot(u.x - slot.x, u.z - slot.z) < 2.0);

    const syncStructuresWithWorld = (structuresList: StructureItem[]) => {
      const incomingIds = new Set(structuresList.map((s) => Number(s.id)));
      for (const [id, entry] of structureMeshes.entries()) {
        if (!incomingIds.has(id)) {
          scene.remove(entry.group);
          disposeSceneSubtree(entry.group);
          structureMeshes.delete(id);
        }
      }

      const usedSlots: StructureSlot[] = Array.from(structureMeshes.values()).map((e) => e.slot);

      structuresList.forEach((struct, index) => {
        const id = Number(struct.id);
        const kind = classifyStructureType(struct.name, struct.unlocks || "");
        // Town already has the Art Studio building and 3 solar charging stations built in.
        if (kind === "solar" || kind === "studio") return;
        if (!structureMeshes.has(id)) {
          let slot = LOT_SLOTS[index % LOT_SLOTS.length];
          if (isSlotNearUsed(slot, usedSlots)) {
            slot = LOT_SLOTS.find((s) => !isSlotNearUsed(s, usedSlots)) || slot;
          }
          usedSlots.push(slot);
          const modelGroup = createStructureModel(kind, ditherUniforms);
          modelGroup.position.set(slot.x, 0, slot.z);
          modelGroup.rotation.y = slot.rot;
          modelGroup.scale.setScalar(0.01);
          scene.add(modelGroup);

          structureMeshes.set(id, { group: modelGroup, slot, pop: 0.01, data: struct });
        }
      });
    };

    const solarStandTarget = (slot: StructureSlot): THREE.Vector3 => {
      chubbyState.activeStructureSlot = {
        x: slot.x,
        z: slot.z,
        rot: slot.rot,
        kind: "solar",
      };
      return new THREE.Vector3(
        slot.x + Math.sin(slot.rot) * 1.18,
        0,
        slot.z + Math.cos(slot.rot) * 1.18
      );
    };

    const getStructureStandTarget = (entry: {
      slot: StructureSlot;
      data: StructureItem;
    }): THREE.Vector3 => {
      const kind = classifyStructureType(entry.data.name, entry.data.unlocks || "");
      chubbyState.activeStructureSlot = {
        x: entry.slot.x,
        z: entry.slot.z,
        rot: entry.slot.rot,
        kind,
      };
      const offset = 1.4;
      return clampTarget(
        entry.slot.x + Math.sin(entry.slot.rot) * offset,
        entry.slot.z + Math.cos(entry.slot.rot) * offset
      );
    };

    const chooseTargetFromAction = (w: WorldPayload): THREE.Vector3 => {
      const action = w.latest_action;
      if (!action) {
        chubbyState.activeStructureSlot = null;
        return new THREE.Vector3(0, 0, 0);
      }

      const cat = action.action || "rest";
      const detail = (action.detail || "").toLowerCase();

      if (cat === "draw" || detail.includes("art studio") || detail.includes("easel")) {
        chubbyState.activeStructureSlot = {
          x: STUDIO.easelX,
          z: STUDIO.easelZ,
          rot: Math.PI,
          kind: "studio",
        };
        return new THREE.Vector3(STUDIO.standX, 0, STUDIO.standZ);
      }

      if (detail.includes("helicopter") || detail.includes("helipad") || detail.includes("rotor")) {
        chubbyState.activeStructureSlot = {
          x: HELIPAD.x,
          z: HELIPAD.z,
          rot: Math.PI,
          kind: "periscope",
        };
        return new THREE.Vector3(HELIPAD.standX, 0, HELIPAD.standZ);
      }

      if ((cat === "recharge" || detail.includes("solar") || detail.includes("charg")) && SOLAR_SLOTS.length > 0) {
        const pos = chubbyState.group.position;
        let chosen = SOLAR_SLOTS[Number(action.tick || 0) % SOLAR_SLOTS.length];
        if (Number(w.state?.battery ?? 1) < 0.4) {
          chosen = SOLAR_SLOTS.reduce((best, s) =>
            Math.hypot(s.x - pos.x, s.z - pos.z) < Math.hypot(best.x - pos.x, best.z - pos.z)
              ? s
              : best
          );
        }
        return solarStandTarget(chosen);
      }

      for (const [, entry] of structureMeshes.entries()) {
        const sName = entry.data.name.toLowerCase();
        if (detail.includes(sName)) {
          return getStructureStandTarget(entry);
        }
      }

      const matchingEntries: Array<{ slot: StructureSlot; data: StructureItem }> = [];
      for (const [, entry] of structureMeshes.entries()) {
        const kind = classifyStructureType(entry.data.name, entry.data.unlocks || "");
        const matchesCat =
          ((kind === "periscope" || kind === "scanner") && (cat === "explore" || cat === "reflect")) ||
          ((kind === "workbench" || kind === "scanner") && cat === "make") ||
          ((kind === "canopy" || kind === "zen" || kind === "lights") && cat === "rest") ||
          ((kind === "radio" || kind === "chime" || kind === "lights") && cat === "social");
        if (matchesCat) {
          matchingEntries.push(entry);
        }
      }
      if (matchingEntries.length > 0) {
        const chosen = matchingEntries[Number(action.tick || 1) % matchingEntries.length];
        return getStructureStandTarget(chosen);
      }

      chubbyState.activeStructureSlot = null;
      if (cat === "social") {
        return new THREE.Vector3(0, 0, 23.0);
      }

      const pt = WANDER_POINTS[Number(action.tick || 1) % WANDER_POINTS.length];
      return new THREE.Vector3(pt.x, 0, pt.z);
    };

    let activeWorld: WorldPayload | null = null;
    let lastGalleryKey = "";
    const galleryTextures: THREE.CanvasTexture[] = [];

    const syncGalleryDrawings = (drawingsList: DrawingItem[]) => {
      const key = drawingsList.map((d) => `${d.id}:${d.tick}:${d.location || ""}`).join(",");
      if (key === lastGalleryKey) return;
      lastGalleryKey = key;

      const setMeshBlank = (mesh: THREE.Mesh) => {
        const mat = mesh.material as THREE.MeshBasicMaterial;
        mat.map = null;
        mat.color.setHex(0xf4efe4);
        mat.needsUpdate = true;
      };

      // Match drawings to wall art slots across the town map
      const assigned = new Map<WallArtSlot, DrawingItem>();
      const usedDrawingIds = new Set<number>();

      // 1. Explicit building/location match
      for (const slot of wallArtSlots) {
        const match = drawingsList.find((d) => {
          if (usedDrawingIds.has(d.id)) return false;
          const loc = (d.location || "").toLowerCase().trim();
          if (!loc) return false;
          return (
            loc === slot.buildingName.toLowerCase() ||
            loc.includes(slot.buildingName.toLowerCase()) ||
            loc === slot.id.toLowerCase() ||
            slot.name.toLowerCase().includes(loc)
          );
        });
        if (match) {
          assigned.set(slot, match);
          usedDrawingIds.add(match.id);
        }
      }

      // 2. Unassigned drawings fill open wall slots (newest first)
      const remainingDrawings = drawingsList.filter((d) => !usedDrawingIds.has(d.id));
      const remainingSlots = wallArtSlots.filter((s) => !assigned.has(s) && s.id !== "art_studio_easel");
      for (let i = 0; i < remainingSlots.length && i < remainingDrawings.length; i++) {
        assigned.set(remainingSlots[i], remainingDrawings[i]);
      }

      // 3. Easel in Art Studio always showcases Chubby's active newest sketch
      if (gallery.easelCanvas) {
        const latest = drawingsList[0];
        if (latest?.svg) {
          loadSvgTexture(latest.svg).then((tex) => {
            if (disposed || key !== lastGalleryKey) {
              tex.dispose();
              return;
            }
            galleryTextures.push(tex);
            const mat = gallery.easelCanvas.material as THREE.MeshBasicMaterial;
            mat.map = tex;
            mat.color.setHex(0xffffff);
            mat.side = THREE.DoubleSide;
            mat.needsUpdate = true;
          });
        } else {
          setMeshBlank(gallery.easelCanvas);
        }
      }

      // 4. Render SVG textures onto their assigned canvas meshes across town
      for (const slot of wallArtSlots) {
        if (slot.id === "art_studio_easel") continue;
        const drawing = assigned.get(slot);
        if (!drawing?.svg) {
          setMeshBlank(slot.canvasMesh);
          continue;
        }
        loadSvgTexture(drawing.svg)
          .then((tex) => {
            if (disposed || key !== lastGalleryKey) {
              tex.dispose();
              return;
            }
            galleryTextures.push(tex);
            const mat = slot.canvasMesh.material as THREE.MeshBasicMaterial;
            mat.map = tex;
            mat.color.setHex(0xffffff);
            mat.side = THREE.DoubleSide;
            mat.needsUpdate = true;
          })
          .catch((err) => {
            console.warn("Wall art texture load error:", err);
          });
      }
    };

    const syncMailboxDrawing = (recentMessages: VisitorMessage[]) => {
      const withDrawing = recentMessages.find(
        (m) => typeof m.drawing === "string" && m.drawing.startsWith("data:image/png;base64,")
      );
      if (!withDrawing || !withDrawing.drawing) {
        mailboxDrawingMesh.visible = false;
        lastRenderedDrawingData = "";
        return;
      }
      if (withDrawing.drawing === lastRenderedDrawingData) return;
      lastRenderedDrawingData = withDrawing.drawing;
      const img = new Image();
      img.onload = () => {
        if (disposed || !drawingPreviewCtx) return;
        drawingPreviewCtx.clearRect(0, 0, 24, 24);
        drawingPreviewCtx.drawImage(img, 0, 0, 24, 24);
        mailboxDrawingTex.needsUpdate = true;
        mailboxDrawingMesh.visible = true;
      };
      img.src = withDrawing.drawing;
    };

    applyWorldRef.current = (w: WorldPayload) => {
      activeWorld = w;
      syncStructuresWithWorld(w.structures || []);
      blueprintPaperGroup.visible = (w.pending_requests || []).length > 0;
      mailboxFlagTargetRot = (w.unread_messages || []).length > 0 ? 0.0 : -Math.PI * 0.45;
      syncMailboxDrawing(w.recent_messages || []);
      syncGalleryDrawings(w.drawings || []);

      if (w.latest_action) {
        const act = w.latest_action;
        if (chubbyState.lastActionId !== act.id) {
          const isInitialLoad = chubbyState.lastActionId === null;
          chubbyState.lastActionId = act.id;
          chubbyState.activeCategory = act.action;
          const nextTarget = chooseTargetFromAction(w);
          chubbyState.target = nextTarget;
          if (isInitialLoad) {
            chubbyState.group.position.set(nextTarget.x, 0, nextTarget.z);
            chubbyState.waypoints = [];
          } else {
            chubbyState.waypoints = findPath(
              chubbyState.group.position.x,
              chubbyState.group.position.z,
              nextTarget.x,
              nextTarget.z
            );
          }
        }
      } else {
        chubbyState.lastActionId = null;
        chubbyState.activeCategory = "rest";
        chubbyState.activeStructureSlot = null;
        chubbyState.target.set(0, 0, 0);
        chubbyState.waypoints = [];
      }
    };

    const setCameraMode = (mode: CameraMode, silent = false) => {
      currentCamMode = mode;
      setCameraModeState(mode);
      if (mode === "first_person") {
        controls.enabled = false;
        camera.position.set(player.x, player.y, player.z);
        camera.rotation.order = "YXZ";
        camera.rotation.y = player.yaw;
        camera.rotation.x = player.pitch;
        camera.rotation.z = 0;
        if (!silent) {
          showToast("1st-Person POV: WASD to walk, Click to look, E to interact");
        }
      } else {
        document.exitPointerLock?.();
        isPointerLocked = false;
        controls.enabled = true;
        camera.position.set(28, 24, 34);
        controls.target.set(0, 1.5, 0);
        controls.update();
        if (!silent) {
          showToast("Orbit Camera: Drag to rotate town, click items to inspect");
        }
      }
    };

    cameraToggleRef.current = () => {
      setCameraMode(currentCamMode === "first_person" ? "orbit" : "first_person");
    };

    const raycaster = new THREE.Raycaster();
    const centerNdc = new THREE.Vector2(0, 0);
    const pointerNdc = new THREE.Vector2(0, 0);
    let lastHoveredType: InteractType | null = null;

    const resolveInteractType = (object: THREE.Object3D | null): InteractType | null => {
      let curr = object;
      while (curr) {
        if (curr.userData && curr.userData.interactType) {
          return curr.userData.interactType as InteractType;
        }
        curr = curr.parent;
      }
      return null;
    };

    const updateRaycastPrompt = () => {
      if (openModalRef.current !== null) {
        if (lastHoveredType !== null) {
          lastHoveredType = null;
          setHoveredInteract(null);
        }
        return;
      }
      const ndc = currentCamMode === "first_person" ? centerNdc : pointerNdc;
      raycaster.setFromCamera(ndc, camera);
      raycaster.far = currentCamMode === "first_person" ? 6.8 : 24.0;
      const hits = raycaster.intersectObjects(interactables, true);
      const nextType = hits.length > 0 ? resolveInteractType(hits[0].object) : null;
      if (nextType !== lastHoveredType) {
        lastHoveredType = nextType;
        setHoveredInteract(nextType);
      }
    };

    const updatePlayer = (dt: number) => {
      if (currentCamMode !== "first_person" || openModalRef.current !== null) return;

      const forwardX = -Math.sin(player.yaw);
      const forwardZ = -Math.cos(player.yaw);
      const rightX = Math.cos(player.yaw);
      const rightZ = -Math.sin(player.yaw);

      let mx = 0;
      let mz = 0;
      if (keys.forward) {
        mx += forwardX;
        mz += forwardZ;
      }
      if (keys.backward) {
        mx -= forwardX;
        mz -= forwardZ;
      }
      if (keys.right) {
        mx += rightX;
        mz += rightZ;
      }
      if (keys.left) {
        mx -= rightX;
        mz -= rightZ;
      }

      const len = Math.hypot(mx, mz);
      if (len > 0.001) {
        mx /= len;
        mz /= len;
        const speed = keys.sprint ? player.sprintSpeed : player.speed;
        player.x += mx * speed * dt;
        player.z += mz * speed * dt;
        player.walkTimer += dt * (keys.sprint ? 1.45 : 1.0);
      }

      player.y = 1.65 + (len > 0.001 ? Math.sin(player.walkTimer * 9.0) * 0.018 : 0);
      const resolvedPlayer = resolveCollisions(player.x, player.z, 0.45);
      player.x = resolvedPlayer.x;
      player.z = resolvedPlayer.z;

      const cPos = chubbyState.group.position;
      const distToChubby = Math.hypot(player.x - cPos.x, player.z - cPos.z);
      if (distToChubby < 0.95 && distToChubby > 0.001) {
        const push = 0.95 - distToChubby;
        const pushed = resolveCollisions(
          player.x + ((player.x - cPos.x) / distToChubby) * push,
          player.z + ((player.z - cPos.z) / distToChubby) * push,
          0.45
        );
        player.x = pushed.x;
        player.z = pushed.z;
      }

      camera.position.set(player.x, player.y, player.z);
      camera.rotation.order = "YXZ";
      camera.rotation.y = player.yaw;
      camera.rotation.x = player.pitch;
      camera.rotation.z = 0;
    };

    const updateChubbyMotion = (dt: number, elapsed: number) => {
      const pos = chubbyState.group.position;
      const waypoint =
        chubbyState.waypoints.length > 0 ? chubbyState.waypoints[0] : chubbyState.target;
      const dx = waypoint.x - pos.x;
      const dz = waypoint.z - pos.z;
      const dist = Math.hypot(dx, dz);

      let moving = false;
      if (dist > 0.12) {
        moving = true;
        const step = Math.min(dist, chubbyState.speed * dt);
        pos.x += (dx / dist) * step;
        pos.z += (dz / dist) * step;
        const desiredYaw = Math.atan2(dx, dz);
        let diff = desiredYaw - chubbyState.yaw;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        chubbyState.yaw += diff * Math.min(1, dt * 8);
        chubbyState.group.rotation.y = chubbyState.yaw;

        if (
          chubbyState.waypoints.length > 0 &&
          Math.hypot(waypoint.x - pos.x, waypoint.z - pos.z) <= 0.18
        ) {
          chubbyState.waypoints.shift();
        }

        dustSpawnCooldown -= dt;
        if (dustSpawnCooldown <= 0) {
          dustSpawnCooldown = 0.18;
          spawnDustPuff(pos.x, pos.z);
        }
      } else if (chubbyState.waypoints.length > 0) {
        chubbyState.waypoints.shift();
      } else if (chubbyState.activeStructureSlot) {
        const s = chubbyState.activeStructureSlot;
        const sx = s.x - pos.x;
        const sz = s.z - pos.z;
        const desiredYaw = Math.hypot(sx, sz) > 0.15 ? Math.atan2(sx, sz) : s.rot;
        let diff = desiredYaw - chubbyState.yaw;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        chubbyState.yaw += diff * Math.min(1, dt * 6);
        chubbyState.group.rotation.y = chubbyState.yaw;
      }

      const resolvedPos = resolveCollisions(pos.x, pos.z, 0.55);
      pos.x = resolvedPos.x;
      pos.z = resolvedPos.z;

      const slot = chubbyState.activeStructureSlot;
      const onSolarPad =
        slot && slot.kind === "solar" && Math.hypot(slot.x - pos.x, slot.z - pos.z) < 1.35;
      pos.y = THREE.MathUtils.lerp(pos.y, onSolarPad ? 0.14 : 0.0, Math.min(1, dt * 10));

      if (chubbyState.fbx) {
        const clip = pickClip(moving, chubbyState.activeCategory, chubbyState.fbx.clips);
        chubbyState.fbx.play(clip);
        chubbyState.fbx.mixer.update(dt);
      } else if (chubbyState.parts) {
        const p = chubbyState.parts;
        p.bodyPivot.position.y = 0;
        p.bodyPivot.rotation.set(0, 0, 0);
        p.head.rotation.set(0, 0, 0);
        p.leftArm.rotation.set(0, 0, 0);
        p.rightArm.rotation.set(0, 0, 0);
        p.leftEye.scale.y = 1.0;
        p.rightEye.scale.y = 1.0;

        if (moving) {
          p.leftWheel.rotation.x += dt * 7.5;
          p.rightWheel.rotation.x += dt * 7.5;
          p.bodyPivot.position.y = Math.abs(Math.sin(elapsed * 10)) * 0.04;
          p.leftArm.rotation.x = Math.sin(elapsed * 10) * 0.25;
          p.rightArm.rotation.x = -Math.sin(elapsed * 10) * 0.25;
        } else {
          const cat = chubbyState.activeCategory;
          if (cat === "rest") {
            p.bodyPivot.position.y = -0.06 + Math.sin(elapsed * 1.8) * 0.015;
            p.leftEye.scale.y = 0.28;
            p.rightEye.scale.y = 0.28;
          } else if (cat === "play") {
            p.bodyPivot.position.y = Math.abs(Math.sin(elapsed * 5.2)) * 0.16;
            p.bodyPivot.rotation.z = Math.sin(elapsed * 4.0) * 0.12;
          } else if (cat === "make" || cat === "maintain") {
            p.leftArm.rotation.x = -0.6 + Math.sin(elapsed * 8) * 0.35;
            p.rightArm.rotation.x = -0.6 - Math.sin(elapsed * 8) * 0.35;
            p.bodyPivot.rotation.x = 0.12;
          } else if (cat === "social") {
            p.rightArm.rotation.z = -0.9 + Math.sin(elapsed * 6) * 0.45;
          } else if (cat === "explore") {
            p.head.rotation.y = Math.sin(elapsed * 2.2) * 0.45;
          } else if (cat === "reflect") {
            p.head.rotation.x = -0.22;
            p.bodyPivot.rotation.z = Math.sin(elapsed * 1.4) * 0.05;
          } else {
            p.bodyPivot.position.y = Math.sin(elapsed * 2.5) * 0.025;
          }
        }
      }
    };

    const updateLighting = (w: WorldPayload | null) => {
      const override = lightingOverrideRef.current;
      let hour: number;
      let isNight: boolean;
      let isDusk: boolean;

      if (override.mode === "auto") {
        if (!w) return;
        hour = Number(w.sim_hour || 12);
        isNight = Boolean(w.is_night);
        isDusk = Boolean(w.is_dusk);
      } else if (override.mode === "dawn") {
        hour = 6.5;
        isNight = false;
        isDusk = false;
      } else if (override.mode === "noon") {
        hour = 12.0;
        isNight = false;
        isDusk = false;
      } else if (override.mode === "dusk") {
        hour = 19.0;
        isNight = false;
        isDusk = true;
      } else if (override.mode === "night") {
        hour = 23.5;
        isNight = true;
        isDusk = false;
      } else {
        hour = override.customHour;
        isNight = hour >= 21 || hour < 5.5;
        isDusk = hour >= 17.5 && hour < 21;
      }

      let skyColor = 0xc4d6cf;
      let fogColor = 0xc4d6cf;
      let sunColor = 0xfff0d4;
      let sunIntensity = 1.45;
      let hemiIntensity = 0.9;

      if (isNight) {
        skyColor = 0x111622;
        fogColor = 0x111622;
        sunColor = 0x90a8d8;
        sunIntensity = 0.4;
        hemiIntensity = 0.3;
      } else if (isDusk) {
        skyColor = 0xb87556;
        fogColor = 0xb87556;
        sunColor = 0xffa66b;
        sunIntensity = 1.0;
        hemiIntensity = 0.65;
      } else if (override.mode === "dawn" || (override.mode === "custom" && hour >= 5.5 && hour < 8.0)) {
        skyColor = 0xdba185;
        fogColor = 0xdba185;
        sunColor = 0xffc499;
        sunIntensity = 1.15;
        hemiIntensity = 0.7;
      }

      (scene.background as THREE.Color).lerp(new THREE.Color(skyColor), 0.05);
      if (scene.fog) {
        scene.fog.color.lerp(new THREE.Color(fogColor), 0.05);
      }
      sunLight.color.lerp(new THREE.Color(sunColor), 0.05);
      sunLight.intensity = THREE.MathUtils.lerp(sunLight.intensity, sunIntensity, 0.05);
      hemiLight.intensity = THREE.MathUtils.lerp(hemiLight.intensity, hemiIntensity, 0.05);

      const angle = (hour / 24) * Math.PI * 2 - Math.PI / 2;
      sunLight.position.set(Math.cos(angle) * 26, Math.max(5, Math.sin(angle) * 32), 14);

      const windowGlow = isNight ? 0.85 : isDusk ? 0.35 : 0.0;
      for (const wm of windowMaterials) {
        wm.emissiveIntensity = THREE.MathUtils.lerp(wm.emissiveIntensity, windowGlow, 0.06);
      }
      const lampGlow = (townGroup.userData as { lampGlow?: THREE.MeshStandardMaterial }).lampGlow;
      if (lampGlow) {
        const targetGlow = isNight ? 1.6 : isDusk ? 0.9 : 0.2;
        lampGlow.emissiveIntensity = THREE.MathUtils.lerp(
          lampGlow.emissiveIntensity,
          targetGlow,
          0.06
        );
      }

      if (chubbyState.parts && w?.state) {
        const st = w.state;
        let colorHex = 0x6dd5a0;
        if (st.battery < 0.32) colorHex = 0xc96442;
        else if (st.joy > 0.8) colorHex = 0xf0c954;
        chubbyState.parts.eyeMat.color.setHex(colorHex);
        chubbyState.parts.eyeMat.emissive.setHex(colorHex);
        chubbyState.parts.antennaTipMat.color.setHex(colorHex);
        chubbyState.parts.antennaTipMat.emissive.setHex(colorHex);
        chubbyState.parts.antennaLight.color.setHex(colorHex);
        chubbyState.parts.antennaLight.intensity = isNight ? 1.15 : isDusk ? 0.75 : 0.4;
      }
    };

    const projVec = new THREE.Vector3();
    const updateBubblePosition = () => {
      const bubbleEl = bubbleRef.current;
      if (!bubbleEl) return;
      if (!activeWorld?.latest_action || openModalRef.current !== null) {
        bubbleEl.style.opacity = "0";
        return;
      }
      projVec.set(
        chubbyState.group.position.x,
        chubbyState.group.position.y + 1.88,
        chubbyState.group.position.z
      );
      projVec.project(camera);
      if (projVec.z > 1) {
        bubbleEl.style.opacity = "0";
        return;
      }
      bubbleEl.style.opacity = "1";
      bubbleEl.style.left = `${(projVec.x * 0.5 + 0.5) * window.innerWidth}px`;
      bubbleEl.style.top = `${(-projVec.y * 0.5 + 0.5) * window.innerHeight}px`;
    };

    // Event handlers
    const onCanvasClick = () => {
      if (openModalRef.current !== null) return;
      if (mouseDragMoved) {
        mouseDragMoved = false;
        return;
      }
      if (lastHoveredType && INTERACT_MODAL[lastHoveredType]) {
        triggerOpenModal(INTERACT_MODAL[lastHoveredType]);
        return;
      }
      if (currentCamMode === "first_person" && !isPointerLocked) {
        canvas.requestPointerLock?.();
      }
    };

    const onCanvasMouseDown = (e: MouseEvent) => {
      if (currentCamMode === "first_person" && !isPointerLocked && openModalRef.current === null) {
        isDraggingMouse = true;
        mouseDragMoved = false;
        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    };

    const onPointerLockChange = () => {
      isPointerLocked = document.pointerLockElement === canvas;
    };

    const onDocumentMouseMove = (e: MouseEvent) => {
      pointerNdc.x = (e.clientX / window.innerWidth) * 2 - 1;
      pointerNdc.y = -(e.clientY / window.innerHeight) * 2 + 1;

      if (currentCamMode !== "first_person" || openModalRef.current !== null) return;

      if (isPointerLocked) {
        const mx = Math.max(-120, Math.min(120, e.movementX));
        const my = Math.max(-120, Math.min(120, e.movementY));
        player.yaw -= mx * 0.0028;
        player.pitch = Math.max(-1.45, Math.min(1.45, player.pitch - my * 0.0028));
      } else if (isDraggingMouse) {
        const dx = e.clientX - previousMousePosition.x;
        const dy = e.clientY - previousMousePosition.y;
        if (Math.hypot(dx, dy) > 3) mouseDragMoved = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
        player.yaw -= dx * 0.004;
        player.pitch = Math.max(-1.45, Math.min(1.45, player.pitch - dy * 0.004));
      }
    };

    const onWindowMouseUp = () => {
      isDraggingMouse = false;
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Escape" && openModalRef.current !== null) {
        closeAllModals();
        return;
      }
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (openModalRef.current !== null) return;

      if (e.code === "KeyW" || e.code === "ArrowUp") keys.forward = true;
      if (e.code === "KeyS" || e.code === "ArrowDown") keys.backward = true;
      if (e.code === "KeyA" || e.code === "ArrowLeft") keys.left = true;
      if (e.code === "KeyD" || e.code === "ArrowRight") keys.right = true;
      if (e.shiftKey) keys.sprint = true;

      if (e.code === "KeyE" && lastHoveredType && INTERACT_MODAL[lastHoveredType]) {
        e.preventDefault();
        triggerOpenModal(INTERACT_MODAL[lastHoveredType]);
      } else if (e.code === "KeyC") {
        cameraToggleRef.current?.();
      } else if (e.code === "KeyH") {
        toggleFullUI();
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "KeyW" || e.code === "ArrowUp") keys.forward = false;
      if (e.code === "KeyS" || e.code === "ArrowDown") keys.backward = false;
      if (e.code === "KeyA" || e.code === "ArrowLeft") keys.left = false;
      if (e.code === "KeyD" || e.code === "ArrowRight") keys.right = false;
      if (!e.shiftKey) keys.sprint = false;
    };

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    canvas.addEventListener("click", onCanvasClick);
    canvas.addEventListener("mousedown", onCanvasMouseDown);
    document.addEventListener("pointerlockchange", onPointerLockChange);
    document.addEventListener("mousemove", onDocumentMouseMove);
    window.addEventListener("mouseup", onWindowMouseUp);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("resize", onResize);

    const refreshWorld = async () => {
      try {
        const w = await apiJson<WorldPayload>("/api/world");
        if (!disposed) syncWorldState(w);
      } catch (err) {
        if (!disposed) {
          setConnected(false);
          console.warn("Poll error:", err);
        }
      }
    };

    setCameraMode("first_person", true);
    void refreshWorld();
    const pollTimer = window.setInterval(refreshWorld, 2000);
    const hintTimer = window.setTimeout(() => {
      if (controlsHintRef.current) {
        controlsHintRef.current.style.opacity = "0.28";
      }
    }, 8000);

    const clock = new THREE.Clock();
    let rafId = 0;
    const animate = () => {
      if (disposed) return;
      rafId = window.requestAnimationFrame(animate);
      const dt = Math.min(0.05, clock.getDelta());
      const elapsed = clock.elapsedTime;

      updatePlayer(dt);
      updateChubbyMotion(dt, elapsed);
      updateDustPuffs(dt);
      updateLighting(activeWorld);

      if (helicopter) {
        // Rotors spin up gently while Chubby is checking the helicopter out.
        const cp = chubbyState.group.position;
        const near = Math.hypot(cp.x - HELIPAD.x, cp.z - HELIPAD.z) < 7.5;
        rotorSpeed = THREE.MathUtils.lerp(rotorSpeed, near ? 1 : 0, Math.min(1, dt * 0.6));
        helicopter.update(dt, rotorSpeed);
      }

      mailboxFlagPivot.rotation.z = THREE.MathUtils.lerp(
        mailboxFlagPivot.rotation.z,
        mailboxFlagTargetRot,
        0.12
      );

      for (const [, entry] of structureMeshes.entries()) {
        if (entry.pop < 1.0) {
          entry.pop = Math.min(1.0, entry.pop + dt * 3.2);
          const s = 1 - Math.pow(1 - entry.pop, 3);
          entry.group.scale.setScalar(s);
        }
      }

      if (currentCamMode === "orbit") {
        controls.update();
      }

      updateRaycastPrompt();
      renderer.render(scene, camera);
      updateBubblePosition();
    };
    animate();

    return () => {
      disposed = true;
      window.cancelAnimationFrame(rafId);
      window.clearInterval(pollTimer);
      window.clearTimeout(hintTimer);
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
      }

      canvas.removeEventListener("click", onCanvasClick);
      canvas.removeEventListener("mousedown", onCanvasMouseDown);
      document.removeEventListener("pointerlockchange", onPointerLockChange);
      document.removeEventListener("mousemove", onDocumentMouseMove);
      window.removeEventListener("mouseup", onWindowMouseUp);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("resize", onResize);
      document.body.classList.remove("ui-hidden");

      controls.dispose();
      mailboxDrawingTex.dispose();
      for (const tex of galleryTextures) tex.dispose();
      dustGeo.dispose();
      disposeSceneSubtree(scene);
      renderer.dispose();
    };
  }, [closeAllModals, showToast, syncWorldState, toggleFullUI, triggerOpenModal]);

  // Derived view values
  const st = world?.state;
  const loop = world?.loop;
  const pending = world?.pending_requests || [];
  const built = world?.structures || [];
  const unread = world?.unread_messages || [];
  const recentOfferings = world?.recent_messages || [];
  const diaryEntries = [...(world?.recent_diary || [])].reverse();
  const memoryEntries = [...(world?.memories || [])].reverse();
  const actionEntries = [...(world?.recent_actions || [])].reverse();
  const galleryDrawings = (world?.drawings || []).slice(0, 15);
  const activeEvents = world?.active_events || [];
  const latestAction = world?.latest_action || null;

  const activePolicyLabel = loop?.effective_policy || loop?.policy || "mock";
  const policyStatus = loop?.running
    ? `autonomous • ${activePolicyLabel}`
    : `paused • ${activePolicyLabel}`;
  const isAtmosphereOverridden = timeOfDayMode !== "auto";
  const effectiveIsNight = isAtmosphereOverridden
    ? timeOfDayMode === "night" || (timeOfDayMode === "custom" && (customHour >= 21 || customHour < 5.5))
    : Boolean(world?.is_night);
  const effectiveIsDusk = isAtmosphereOverridden
    ? timeOfDayMode === "dusk" || (timeOfDayMode === "custom" && customHour >= 17.5 && customHour < 21)
    : Boolean(world?.is_dusk);

  const phaseClass = effectiveIsNight
    ? "badge badge-night"
    : effectiveIsDusk
      ? "badge badge-dusk"
      : "badge badge-day";
  const phaseLabel = isAtmosphereOverridden
    ? timeOfDayMode === "custom"
      ? `Ovr: ${formatHourShort(customHour)}`
      : `${timeOfDayMode.charAt(0).toUpperCase() + timeOfDayMode.slice(1)} (Ovr)`
    : world?.is_night
      ? "Night"
      : world?.is_dusk
        ? "Dusk"
        : "Daytime";
  const dotClass = !connected ? "dot" : loop?.tick_in_progress ? "dot busy" : "dot online";
  const latestOutcomeText =
    latestAction?.outcome?.outcome || "Town block initialized. Chubby is waking up autonomously.";

  const handleToggleDither = () => {
    const nextEnabled = !ditherEnabled;
    ditherUniformsRef.current.uDitherEnabled.value = nextEnabled ? 1.0 : 0.0;
    setDitherEnabled(nextEnabled);
    showToast(nextEnabled ? "Ordered Bayer dithering enabled" : "Clean low-poly shading enabled");
  };

  const handleHangDrawing = async (id: number, location: string) => {
    try {
      const resp = await fetch("/api/drawings/hang", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, location }),
      });
      if (resp.ok) {
        showToast(`Moved artwork to ${location} wall`);
        setWorld((prev) => {
          if (!prev || !prev.drawings) return prev;
          return {
            ...prev,
            drawings: prev.drawings.map((d) => (d.id === id ? { ...d, location } : d)),
          };
        });
      }
    } catch (err) {
      console.warn("Failed to update wall art location:", err);
    }
  };

  const handleStepTick = async () => {
    try {
      setStepBusy(true);
      const resp = await apiJson<{ world: WorldPayload }>("/api/tick", "POST", {
        policy: policyValue,
      });
      syncWorldState(resp.world);
    } catch (err) {
      showToast(`Tick failed: ${(err as Error).message}`);
    } finally {
      setStepBusy(false);
    }
  };

  const handleToggleLoop = async () => {
    const currentlyRunning = Boolean(loop?.running);
    try {
      const resp = await apiJson<{ world: WorldPayload }>("/api/loop", "POST", {
        running: !currentlyRunning,
        policy: policyValue,
      });
      syncWorldState(resp.world);
      showToast(resp.world.loop?.running ? "Autonomous loop running" : "Autonomous loop paused");
    } catch (err) {
      showToast(`Loop update failed: ${(err as Error).message}`);
    }
  };

  const handlePolicyChange = async (nextPolicy: string) => {
    setPolicyValue(nextPolicy);
    try {
      const resp = await apiJson<{ world: WorldPayload }>("/api/loop", "POST", {
        policy: nextPolicy,
      });
      syncWorldState(resp.world);
    } catch (err) {
      showToast(`Policy error: ${(err as Error).message}`);
    }
  };

  const handleReset = async () => {
    try {
      const resp = await apiJson<{ world: WorldPayload }>("/api/reset", "POST", {});
      syncWorldState(resp.world);
      closeAllModals();
      showToast("Lot reset to Tick 0");
    } catch (err) {
      showToast(`Reset failed: ${(err as Error).message}`);
    }
  };

  const handleOfferingSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const clean = offeringText.trim();
    if (!clean) {
      showToast("Please write a short message for Chubby first");
      return;
    }
    const drawingData =
      hasPixelEditsRef.current && pixelCanvasRef.current
        ? pixelCanvasRef.current.toDataURL("image/png")
        : "";
    try {
      const resp = await apiJson<{ world: WorldPayload }>("/api/message", "POST", {
        text: clean,
        drawing: drawingData,
      });
      syncWorldState(resp.world);
      setOfferingText("");
      clearPixelCanvas();
      showToast("Offering placed in the mailbox!");
    } catch (err) {
      showToast(`Failed to post offering: ${(err as Error).message}`);
    }
  };

  const handleApproveRequest = async (requestId: number) => {
    try {
      const resp = await apiJson<{ world: WorldPayload; decision: { name: string } }>(
        `/api/requests/${requestId}/approve`,
        "POST",
        { note: "Approved from fence noticeboard" }
      );
      syncWorldState(resp.world);
      showToast(`Built '${resp.decision.name}' inside the yard!`);
    } catch (err) {
      showToast(`Approve error: ${(err as Error).message}`);
    }
  };

  const handleRejectRequest = async (requestId: number) => {
    const reason = window.prompt("Optional reason for declining:", "Not enough scrap right now");
    if (reason === null) return;
    try {
      const resp = await apiJson<{ world: WorldPayload; decision: { name: string } }>(
        `/api/requests/${requestId}/reject`,
        "POST",
        { reason: reason || "Not needed right now" }
      );
      syncWorldState(resp.world);
      showToast(`Declined '${resp.decision.name}'`);
    } catch (err) {
      showToast(`Decline error: ${(err as Error).message}`);
    }
  };

  return (
    <>
      <canvas id="world-canvas" ref={canvasRef} />

      <div
        id="reticle"
        className={`reticle${cameraMode !== "first_person" ? " hidden" : ""}${
          hoveredInteract && !openModal ? " active" : ""
        }`}
      />

      <div
        id="interact-prompt"
        className={`interact-prompt${hoveredInteract && !openModal ? "" : " hidden"}`}
      >
        {hoveredInteract ? INTERACT_LABELS[hoveredInteract] : ""}
      </div>

      <div ref={controlsHintRef} id="controls-hint" className="controls-hint">
        <span>
          <kbd>WASD</kbd> Walk
        </span>
        <span className="sep">•</span>
        <span>
          <b>Mouse</b> Look
        </span>
        <span className="sep">•</span>
        <span>
          <kbd>E</kbd> Interact
        </span>
        <span className="sep">•</span>
        <span>
          <kbd>C</kbd> Camera
        </span>
        <span className="sep">•</span>
        <span>
          <kbd>H</kbd> Zen
        </span>
      </div>

      <div
        ref={bubbleRef}
        id="chubby-bubble"
        className={`chubby-bubble${latestAction ? "" : " hidden"}`}
      >
        <div className="bubble-header">
          <span id="bubble-category" className="badge badge-category">
            {latestAction?.action || "idle"}
          </span>
          <span id="bubble-tick" className="bubble-tick">
            tick {latestAction?.tick ?? 0}
          </span>
        </div>
        <p id="bubble-thought" className="bubble-thought">
          {latestAction ? `"${latestAction.thought}"` : "Waking up in the lot..."}
        </p>
        <p id="bubble-intent" className="bubble-intent">
          {latestAction?.detail || ""}
        </p>
        <p id="bubble-feeling" className="bubble-feeling">
          {describeFeeling(st)}
        </p>
      </div>

      <header id="world-stamp" className="world-stamp">
        <div className="stamp-title-row">
          <span id="conn-dot" className={dotClass} title="World connection" />
          <h1 className="stamp-title">chubby.world</h1>
          <span id="policy-pill" className="stamp-pill">
            {policyStatus}
          </span>
        </div>
        <div className="stamp-meta-row">
          <span id="sim-time" className="clock-text">
            {st?.sim_time || "Day 1, 08:00"}
          </span>
          <span
            id="phase-badge"
            className={phaseClass}
            onClick={() => setAtmospherePanelOpen((prev) => !prev)}
            style={{ cursor: "pointer" }}
            title="Click to toggle observer atmosphere lighting & celestial angle control"
          >
            {phaseLabel}
          </span>
        </div>
      </header>

      <nav id="world-dock" className="world-dock" aria-label="World actions">
        <button
          id="btn-open-diary"
          className="dock-pill"
          title="Open Chubby's Field Diary"
          onClick={() => triggerOpenModal("modal-diary")}
        >
          📖 Diary
        </button>
        <button
          id="btn-open-offering"
          className="dock-pill"
          title="Leave a note & pixel drawing at the mailbox"
          onClick={() => triggerOpenModal("modal-offering")}
        >
          📮 Leave Offering
        </button>
        <button
          id="btn-open-blueprints"
          className="dock-pill"
          title="Review Chubby's structure requests"
          onClick={() => triggerOpenModal("modal-blueprints")}
        >
          📌 Blueprints{" "}
          <span
            id="badge-alert"
            className={`badge-count${pending.length === 0 ? " hidden" : ""}`}
          >
            {pending.length}
          </span>
        </button>
        <button
          id="btn-open-status"
          className="dock-pill"
          title="Inspect Chubby's mood & needs"
          onClick={() => triggerOpenModal("modal-status")}
        >
          🤖 Chubby
        </button>
        <button
          id="btn-cam-mode"
          className={`dock-pill${cameraMode === "first_person" ? " dock-accent" : ""}`}
          title="Switch First-Person / Orbit camera (C)"
          onClick={() => cameraToggleRef.current?.()}
        >
          {cameraMode === "first_person" ? "🚶 1st Person" : "🛸 Orbit View"}
        </button>
        <button
          id="btn-atmosphere"
          className={`dock-pill${atmospherePanelOpen || isAtmosphereOverridden ? " dock-accent" : ""}`}
          title="Observer Atmosphere & Celestial Lighting Override"
          onClick={() => setAtmospherePanelOpen((prev) => !prev)}
        >
          {timeOfDayMode === "auto"
            ? (world?.is_night ? "🌙 Sim Night" : world?.is_dusk ? "🌇 Sim Dusk" : "☀️ Sim Day")
            : (timeOfDayMode === "dawn" ? "🌅 Dawn (Ovr)" : timeOfDayMode === "noon" ? "☀️ Noon (Ovr)" : timeOfDayMode === "dusk" ? "🌇 Dusk (Ovr)" : timeOfDayMode === "night" ? "🌙 Night (Ovr)" : `🕐 ${formatHourShort(customHour)} (Ovr)`)}
        </button>
        <button
          id="btn-open-dev"
          className="dock-pill dock-icon"
          title="Simulation & visual settings"
          onClick={() => triggerOpenModal("modal-dev")}
        >
          ⚙️
        </button>
      </nav>

      {/* Floating Atmosphere & Lighting Observer Bar */}
      {atmospherePanelOpen && (
        <div id="atmosphere-bar" className="atmosphere-bar" role="region" aria-label="Observer Atmosphere Control">
          <div className="atmosphere-header">
            <div className="atmosphere-title-group">
              <span className="atmosphere-kicker">OBSERVER ENVIRONMENT</span>
              <span className="atmosphere-title">Lighting &amp; Celestial Cycle</span>
            </div>
            <button
              className="btn-close-sm"
              aria-label="Close atmosphere control"
              onClick={() => setAtmospherePanelOpen(false)}
            >
              ✕
            </button>
          </div>

          <div className="atmosphere-presets">
            <button
              className={`preset-btn${timeOfDayMode === "auto" ? " active" : ""}`}
              onClick={() => {
                setTimeOfDayMode("auto");
                lightingOverrideRef.current = { mode: "auto", customHour };
              }}
              title="Synchronize viewport with simulation tick clock"
            >
              🌐 Sim Auto
            </button>
            <button
              className={`preset-btn${timeOfDayMode === "dawn" ? " active" : ""}`}
              onClick={() => {
                setTimeOfDayMode("dawn");
                lightingOverrideRef.current = { mode: "dawn", customHour: 6.5 };
              }}
              title="Dawn / Early Morning (06:30)"
            >
              🌅 Dawn
            </button>
            <button
              className={`preset-btn${timeOfDayMode === "noon" ? " active" : ""}`}
              onClick={() => {
                setTimeOfDayMode("noon");
                lightingOverrideRef.current = { mode: "noon", customHour: 12.0 };
              }}
              title="Midday Sun (12:00)"
            >
              ☀️ Noon
            </button>
            <button
              className={`preset-btn${timeOfDayMode === "dusk" ? " active" : ""}`}
              onClick={() => {
                setTimeOfDayMode("dusk");
                lightingOverrideRef.current = { mode: "dusk", customHour: 19.0 };
              }}
              title="Dusk / Sunset (19:00)"
            >
              🌇 Dusk
            </button>
            <button
              className={`preset-btn${timeOfDayMode === "night" ? " active" : ""}`}
              onClick={() => {
                setTimeOfDayMode("night");
                lightingOverrideRef.current = { mode: "night", customHour: 23.5 };
              }}
              title="Midnight Moon &amp; Streetlamps (23:30)"
            >
              🌙 Night
            </button>
          </div>

          <div className="atmosphere-slider-group">
            <div className="slider-label-row">
              <span className="slider-label">Scrub Celestial Angle:</span>
              <span className="slider-value-badge">
                {timeOfDayMode === "auto"
                  ? `Sim Clock (${world?.state?.sim_time?.split(", ")[1] || "08:00"})`
                  : timeOfDayMode === "custom"
                    ? formatHourShort(customHour)
                    : `${timeOfDayMode.charAt(0).toUpperCase() + timeOfDayMode.slice(1)} (${formatHourShort(
                        timeOfDayMode === "dawn" ? 6.5 : timeOfDayMode === "noon" ? 12 : timeOfDayMode === "dusk" ? 19 : 23.5
                      )})`}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="23.75"
              step="0.25"
              value={timeOfDayMode === "custom" ? customHour : timeOfDayMode === "dawn" ? 6.5 : timeOfDayMode === "noon" ? 12 : timeOfDayMode === "dusk" ? 19 : timeOfDayMode === "night" ? 23.5 : (world?.sim_hour ?? 8)}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setCustomHour(val);
                setTimeOfDayMode("custom");
                lightingOverrideRef.current = { mode: "custom", customHour: val };
              }}
              className="atmosphere-slider"
            />
          </div>

          <div className="atmosphere-caption">
            Simulation decisions, referee decay, and field diary continue at autonomous tick time. This override adjusts observer viewport lighting and celestial angles.
          </div>
        </div>
      )}

      {/* MODAL: Field Diary & Memories Book */}
      <div
        id="modal-diary"
        className={`modal-backdrop${openModal === "modal-diary" ? "" : " hidden"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="diary-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) closeAllModals();
        }}
      >
        <div className="paper-modal notebook-modal">
          <div className="modal-header">
            <div>
              <span className="modal-kicker">FIELD JOURNAL</span>
              <h2 id="diary-title">Chubby’s Notebook &amp; Memories</h2>
            </div>
            <button
              className="btn-close-modal"
              aria-label="Close diary"
              onClick={closeAllModals}
            >
              ✕
            </button>
          </div>
          <div className="tabs" role="tablist" aria-label="Field Journal sections">
            {(
              [
                ["diary", "Field Diary"],
                ["memories", "Memories"],
                ["actions", "Action Log"],
                [
                  "gallery",
                  galleryDrawings.length > 0
                    ? `Art Gallery (${galleryDrawings.length})`
                    : "Art Gallery",
                ],
              ] as Array<[DiaryTab, string]>
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={activeTab === key}
                className={`tab-btn${activeTab === key ? " active" : ""}`}
                onClick={() => setActiveTab(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="notebook-body">
            {activeTab === "diary" && (
              <div className="tab-pane scroll-box active" role="tabpanel">
                {galleryDrawings.length > 0 && (
                  <div
                    className="diary-sketches-strip"
                    style={{
                      marginBottom: 14,
                      padding: "8px 10px",
                      background: "rgba(244, 239, 228, 0.45)",
                      border: "1px dashed #b8a892",
                      borderRadius: 8,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: 6,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-mono)",
                          fontWeight: 700,
                          textTransform: "uppercase",
                          letterSpacing: "0.08em",
                          color: "var(--accent-warm)",
                        }}
                      >
                        🎨 Chubby&apos;s Field Sketches ({galleryDrawings.length}/15)
                      </span>
                      <button
                        type="button"
                        onClick={() => setActiveTab("gallery")}
                        style={{
                          background: "none",
                          border: "none",
                          color: "var(--accent-warm)",
                          fontSize: 11,
                          cursor: "pointer",
                          textDecoration: "underline",
                          fontFamily: "inherit",
                        }}
                      >
                        Open Art Gallery →
                      </button>
                    </div>
                    <div
                      style={{
                        display: "flex",
                        gap: 8,
                        overflowX: "auto",
                        paddingBottom: 4,
                      }}
                    >
                      {galleryDrawings.map((dr) => (
                        <div
                          key={dr.id}
                          style={{
                            flex: "0 0 95px",
                            cursor: "pointer",
                            background: "#d4d0c8",
                            border: "1px solid #808080",
                            boxShadow: "inset 1px 1px 0 #fff",
                            borderRadius: 4,
                            padding: 4,
                            textAlign: "center",
                            transition: "transform 0.12s ease",
                          }}
                          onClick={() => setActiveTab("gallery")}
                          title={`"${dr.title}" • Tick ${dr.tick} • Hanging at ${dr.location || "Art Studio"}`}
                        >
                          <img
                            src={`data:image/svg+xml;utf8,${encodeURIComponent(dr.svg)}`}
                            alt={dr.title}
                            style={{
                              width: "100%",
                              height: 62,
                              objectFit: "contain",
                              background: "#ffffff",
                              border: "1px inset #808080",
                              display: "block",
                            }}
                          />
                          <div
                            style={{
                              fontSize: 9.5,
                              fontWeight: 700,
                              color: "#0a246a",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                              marginTop: 3,
                            }}
                          >
                            {dr.title}
                          </div>
                          <div style={{ fontSize: 8.5, color: "#555" }}>
                            tick {dr.tick}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {diaryEntries.length > 0 ? (
                  diaryEntries.map((d, idx) => {
                    const drawing = galleryDrawings.find((dr) => dr.tick === d.tick);
                    return (
                      <div key={`${d.tick}-${idx}`} className="log-entry diary-page">
                        <div className="log-meta">
                          {d.sim_time} • tick {d.tick}
                        </div>
                        <div>{d.text}</div>
                        {drawing && (
                          <div
                            style={{
                              marginTop: 8,
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 10,
                              padding: "6px 10px",
                              background: "#d4d0c8",
                              border: "1px solid #808080",
                              borderRadius: 4,
                              boxShadow: "inset 1px 1px 0 #fff",
                              maxWidth: "100%",
                            }}
                          >
                            <img
                              src={`data:image/svg+xml;utf8,${encodeURIComponent(drawing.svg)}`}
                              alt={drawing.title}
                              style={{
                                width: 54,
                                height: 44,
                                objectFit: "contain",
                                background: "#fff",
                                border: "1px inset #808080",
                                flexShrink: 0,
                              }}
                            />
                            <div style={{ fontSize: 11 }}>
                              <div style={{ fontWeight: 700, color: "#0a246a" }}>
                                🎨 Drawing: {drawing.title}
                              </div>
                              <div style={{ fontSize: 10, color: "#444" }}>
                                Displayed on {drawing.location || "Art Studio"} wall
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <div className="empty-notice">Chubby has not written in the field diary yet.</div>
                )}
              </div>
            )}
            {activeTab === "memories" && (
              <div className="tab-pane scroll-box active" role="tabpanel">
                {memoryEntries.length > 0 ? (
                  memoryEntries.map((m, idx) => (
                    <div key={`${m.tick_range}-${idx}`} className="log-entry">
                      <div className="log-meta">Ticks {m.tick_range}</div>
                      <div>{m.summary}</div>
                    </div>
                  ))
                ) : (
                  <div className="empty-notice">
                    Long-term memory cards summarize every 50 ticks.
                  </div>
                )}
              </div>
            )}
            {activeTab === "actions" && (
              <div className="tab-pane scroll-box active" role="tabpanel">
                {actionEntries.length > 0 ? (
                  actionEntries.map((a) => (
                    <div key={a.id} className="log-entry">
                      <div className="log-meta">
                        <span className="badge badge-category">{a.action}</span> • tick {a.tick}
                      </div>
                      <div>{a.detail}</div>
                      <div className="muted" style={{ marginTop: "3px" }}>
                        {a.outcome?.outcome || ""}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="empty-notice">No actions recorded yet.</div>
                )}
              </div>
            )}
            {activeTab === "gallery" && (
              <div className="tab-pane scroll-box active" role="tabpanel">
                {galleryDrawings.length > 0 ? (
                  <div className="gallery-grid">
                    {galleryDrawings.map((d) => (
                      <div
                        key={d.id}
                        className="gallery-card"
                        style={{
                          background: "#d4d0c8",
                          border: "2px solid #808080",
                          boxShadow: "inset 1px 1px 0 #ffffff",
                          padding: 6,
                          borderRadius: 4,
                        }}
                      >
                        <div
                          style={{
                            background: "#0a246a",
                            color: "#ffffff",
                            fontSize: 11,
                            fontWeight: 700,
                            padding: "2px 6px",
                            marginBottom: 4,
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <span>🎨 {d.title} - Paint</span>
                          <span style={{ opacity: 0.85, fontWeight: 400 }}>t{d.tick}</span>
                        </div>
                        <div style={{ display: "flex", gap: 4, alignItems: "stretch" }}>
                          <div
                            style={{
                              display: "flex",
                              flexDirection: "column",
                              gap: 2,
                              background: "#d4d0c8",
                              padding: 2,
                              border: "1px solid #808080",
                              fontSize: 10,
                              lineHeight: 1.2,
                            }}
                            title="Paint Tools"
                          >
                            <span>✏️</span>
                            <span>🖌️</span>
                            <span>🪣</span>
                            <span>⬜</span>
                            <span>⭕</span>
                          </div>
                          <img
                            className="gallery-art"
                            src={`data:image/svg+xml;utf8,${encodeURIComponent(d.svg)}`}
                            alt={d.title}
                            style={{
                              background: "#ffffff",
                              border: "2px inset #808080",
                              flex: 1,
                            }}
                          />
                        </div>
                        <div
                          style={{
                            display: "flex",
                            gap: 2,
                            marginTop: 4,
                            padding: "2px 3px",
                            background: "#c0c0c0",
                            border: "1px inset #808080",
                          }}
                        >
                          {["#000000", "#ffffff", "#d94b2b", "#f2c230", "#4d7c57", "#3b7ea1", "#7a4fb5", "#8a6a4a"].map((c) => (
                            <span
                              key={c}
                              style={{
                                width: 10,
                                height: 10,
                                background: c,
                                border: "1px solid #404040",
                                display: "inline-block",
                              }}
                            />
                          ))}
                        </div>
                        <div
                          className="gallery-location-row"
                          style={{
                            marginTop: 6,
                            display: "flex",
                            alignItems: "center",
                            gap: 4,
                            fontSize: 11,
                          }}
                        >
                          <span style={{ opacity: 0.85, color: "#1f1914" }}>📍 Wall:</span>
                          <select
                            value={d.location || "Art Studio"}
                            onChange={(e) => handleHangDrawing(d.id, e.target.value)}
                            style={{
                              background: "#f4efe4",
                              border: "1px solid #808080",
                              borderRadius: 2,
                              fontSize: 11,
                              padding: "2px 4px",
                              cursor: "pointer",
                              fontFamily: "inherit",
                              color: "#2a2420",
                            }}
                          >
                            <option value="Bakery">Bakery Wall</option>
                            <option value="Library">Library Wall</option>
                            <option value="Cafe">Cafe Wall</option>
                            <option value="Workshop">Workshop Wall</option>
                            <option value="Post Office">Post Office Wall</option>
                            <option value="Apartments">Apartments Wall</option>
                            <option value="Art Studio">Art Studio</option>
                          </select>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-notice">
                    No paintings in the Art Studio yet. Watch Chubby visit the studio to sketch!
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* MODAL: Visitor Offering Post */}
      <div
        id="modal-offering"
        className={`modal-backdrop${openModal === "modal-offering" ? "" : " hidden"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="offering-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) closeAllModals();
        }}
      >
        <div className="paper-modal offering-modal">
          <div className="modal-header">
            <div>
              <span className="modal-kicker">SOUTH GATE MAILBOX</span>
              <h2 id="offering-title">Leave an Offering for Chubby</h2>
            </div>
            <button
              className="btn-close-modal"
              aria-label="Close offering"
              onClick={closeAllModals}
            >
              ✕
            </button>
          </div>
          <p className="modal-intro">
            Drop a note and an optional tiny 24×24 pixel sketch into the gate mailbox. Your latest
            sketch hangs on the post in 3D, and Chubby reads offerings in his own time.
          </p>
          <form id="offering-form" className="offering-form" onSubmit={handleOfferingSubmit}>
            <div className="offering-compose">
              <div className="sketch-pad-wrap">
                <canvas
                  id="pixel-canvas"
                  ref={pixelCanvasRef}
                  width={24}
                  height={24}
                  title="Click or drag to draw a 24x24 pixel sketch"
                />
                <div className="palette-row" aria-label="Sketch colors">
                  {SWATCH_COLORS.map((s) => (
                    <button
                      key={s.color}
                      type="button"
                      className={`swatch-btn${s.isPaper ? " swatch-paper" : ""}${
                        selectedColor === s.color ? " active" : ""
                      }`}
                      style={{ background: s.color }}
                      title={s.title}
                      onClick={() => setSelectedColor(s.color)}
                    />
                  ))}
                </div>
                <button
                  type="button"
                  id="btn-clear-pixel"
                  className="btn btn-paper btn-sm"
                  onClick={clearPixelCanvas}
                >
                  Clear Sketch
                </button>
              </div>
              <div className="note-compose-wrap">
                <label htmlFor="offering-input" className="field-label">
                  Visitor Note{" "}
                  <span id="unread-count" className="badge badge-muted">
                    {unread.length} unread
                  </span>
                </label>
                <textarea
                  id="offering-input"
                  rows={4}
                  maxLength={280}
                  placeholder="Write something kind or curious for Chubby..."
                  autoComplete="off"
                  value={offeringText}
                  onChange={(e) => setOfferingText(e.target.value)}
                />
                <div className="offering-submit-row">
                  <span className="hint-small">Max 280 chars • Sketch is optional</span>
                  <button type="submit" className="btn btn-ink">
                    Post Offering
                  </button>
                </div>
              </div>
            </div>
          </form>
          <div className="section-divider">
            <h3>Recent Offerings on the Post</h3>
            <div id="offerings-list" className="scroll-box small">
              {recentOfferings.length > 0 ? (
                recentOfferings.map((m, idx) => {
                  const hasImg =
                    typeof m.drawing === "string" &&
                    m.drawing.startsWith("data:image/png;base64,");
                  return (
                    <div key={`${m.received_tick}-${idx}`} className="offering-item">
                      {hasImg && (
                        <img
                          className="offering-thumb"
                          src={m.drawing}
                          alt="Visitor pixel sketch"
                        />
                      )}
                      <div className="offering-item-body">
                        <div className="log-meta">
                          tick {m.received_tick} • {m.read ? "read by Chubby" : "waiting in mailbox"}
                        </div>
                        <div className="offering-item-text">{m.text}</div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="empty-notice">
                  No offerings left at the mailbox yet. Be the first visitor!
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* MODAL: Fence Blueprint Noticeboard */}
      <div
        id="modal-blueprints"
        className={`modal-backdrop${openModal === "modal-blueprints" ? "" : " hidden"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="blueprints-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) closeAllModals();
        }}
      >
        <div className="paper-modal">
          <div className="modal-header">
            <div>
              <span className="modal-kicker">FENCE NOTICEBOARD</span>
              <h2 id="blueprints-title">Structure Blueprints &amp; Built Yard</h2>
            </div>
            <button
              className="btn-close-modal"
              aria-label="Close blueprints"
              onClick={closeAllModals}
            >
              ✕
            </button>
          </div>
          <section className="modal-section">
            <div className="section-title">
              <h3>Pending Blueprint Requests</h3>
              <span id="pending-count" className="badge badge-muted">
                {pending.length} pending
              </span>
            </div>
            <div id="requests-list" className="scroll-box">
              {pending.length > 0 ? (
                pending.map((r) => (
                  <div key={r.id} className="request-card">
                    <div className="request-card-header">
                      <span className="request-title">{r.name}</span>
                      <span className="badge badge-category">Priority {r.priority}</span>
                    </div>
                    <div className="request-meta">
                      <em>Purpose:</em> {r.purpose}
                    </div>
                    <div className="request-meta">
                      <em>Why:</em> {r.why}
                    </div>
                    <div className="request-actions">
                      <button
                        className="btn btn-leaf btn-sm"
                        onClick={() => handleApproveRequest(r.id)}
                      >
                        Approve Build
                      </button>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => handleRejectRequest(r.id)}
                      >
                        Decline
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-notice">
                  No blueprint requests pinned on the fence board right now.
                </div>
              )}
            </div>
          </section>
          <section className="modal-section">
            <div className="section-title">
              <h3>Built Structures in the Lot</h3>
              <span id="structures-count" className="badge badge-muted">
                {built.length}
              </span>
            </div>
            <div id="structures-list" className="structures-list">
              {built.length > 0 ? (
                built.map((s) => (
                  <span key={s.id} className="structure-chip">
                    🏗️ {s.name}
                  </span>
                ))
              ) : (
                <div className="empty-notice">
                  The fenced lot is completely empty. Approved blueprints appear here and in the
                  yard.
                </div>
              )}
            </div>
          </section>
        </div>
      </div>

      {/* MODAL: Chubby Inspection Card */}
      <div
        id="modal-status"
        className={`modal-backdrop${openModal === "modal-status" ? "" : " hidden"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="status-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) closeAllModals();
        }}
      >
        <div className="paper-modal">
          <div className="modal-header">
            <div>
              <span className="modal-kicker">RESIDENT DIAGNOSTIC</span>
              <h2 id="status-title">Chubby’s Needs &amp; Mood</h2>
            </div>
            <button
              className="btn-close-modal"
              aria-label="Close inspection"
              onClick={closeAllModals}
            >
              ✕
            </button>
          </div>
          <section className="modal-section">
            <div className="section-title">
              <h3>Internal Drives</h3>
              <span id="tick-indicator" className="badge badge-muted">
                Tick {st?.tick ?? 0}
              </span>
            </div>
            <div className="bars">
              {(
                [
                  ["Battery", st?.battery ?? 0.8],
                  ["Curiosity", st?.curiosity ?? 0.7],
                  ["Joy", st?.joy ?? 0.7],
                  ["Calm", st?.calm ?? 0.75],
                  ["Belonging", st?.belonging ?? 0.65],
                ] as Array<[string, number]>
              ).map(([label, rawVal]) => {
                const val = Number(rawVal);
                const pct = Math.max(0, Math.min(100, val * 100));
                const fillColor =
                  val < 0.25
                    ? "var(--danger)"
                    : val < 0.45
                      ? "var(--accent-ochre)"
                      : "var(--accent-leaf)";
                return (
                  <div key={label} className="bar-row">
                    <div className="bar-label">
                      <span>{label}</span>
                      <span>{val.toFixed(2)}</span>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${pct.toFixed(1)}%`, backgroundColor: fillColor }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
          <section className="modal-section">
            <h3>Core Summary</h3>
            <div className="core-grid">
              <div className="core-card">
                <span className="core-label">Energy</span>
                <strong>{Number(st?.energy ?? 0.79).toFixed(2)}</strong>
              </div>
              <div className="core-card">
                <span className="core-label">Mood</span>
                <strong>{Number(st?.mood ?? 0.7).toFixed(2)}</strong>
              </div>
              <div className="core-card">
                <span className="core-label">Mood Mod</span>
                <strong>
                  {Number(st?.mood_modifier ?? 0) >= 0 ? "+" : ""}
                  {Number(st?.mood_modifier ?? 0).toFixed(2)}
                </strong>
              </div>
            </div>
          </section>
          <section className="modal-section">
            <h3>Latest Outcome &amp; Events</h3>
            <p id="latest-outcome" className="outcome-text">
              {latestOutcomeText}
            </p>
            <div id="active-events" className="events-list">
              {activeEvents.map((ev, idx) => (
                <div key={`${ev}-${idx}`} className="event-chip">
                  ⚡ {ev}
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>

      {/* MODAL: Simulation & Visuals Settings */}
      <div
        id="modal-dev"
        className={`modal-backdrop${openModal === "modal-dev" ? "" : " hidden"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dev-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) closeAllModals();
        }}
      >
        <div className="paper-modal dev-modal">
          <div className="modal-header">
            <div>
              <span className="modal-kicker">WORLD SETTINGS</span>
              <h2 id="dev-title">Simulation &amp; Visuals</h2>
            </div>
            <button
              className="btn-close-modal"
              aria-label="Close settings"
              onClick={closeAllModals}
            >
              ✕
            </button>
          </div>
          <div className="dev-grid">
            <div className="dev-row">
              <label htmlFor="policy-select" className="field-label">
                Brain Policy
              </label>
              <select
                id="policy-select"
                className="paper-select"
                value={policyValue}
                onChange={(e) => handlePolicyChange(e.target.value)}
              >
                <option value="llm">Ollama (qwen3:4b + Auto Fallback)</option>
                <option value="mock">Mock (Fast Deterministic)</option>
                <option value="random">Random Policy</option>
              </select>
            </div>
            <div className="dev-actions">
              <button id="btn-loop" className="btn btn-ink" onClick={handleToggleLoop}>
                {loop?.running ? "Auto Loop: ON" : "Auto Loop: OFF"}
              </button>
              <button
                id="btn-step"
                className="btn btn-paper"
                disabled={stepBusy || Boolean(loop?.tick_in_progress)}
                onClick={handleStepTick}
              >
                Step 1 Tick
              </button>
              <button id="btn-dither" className="btn btn-paper" onClick={handleToggleDither}>
                {ditherEnabled ? "Retro Dithering: ON" : "Retro Dithering: OFF"}
              </button>
              <button
                id="btn-toggle-ui"
                className="btn btn-paper"
                onClick={() => {
                  closeAllModals();
                  toggleFullUI();
                }}
              >
                Hide HUD (H)
              </button>
            </div>
            <div className="section-divider">
              <label className="field-label" style={{ display: "block", marginBottom: "6px" }}>
                Observer Atmosphere &amp; Celestial Cycle
              </label>
              <div className="atmosphere-presets">
                <button
                  className={`preset-btn${timeOfDayMode === "auto" ? " active" : ""}`}
                  onClick={() => {
                    setTimeOfDayMode("auto");
                    lightingOverrideRef.current = { mode: "auto", customHour };
                  }}
                  title="Follow simulation tick clock"
                >
                  🌐 Sim Auto
                </button>
                <button
                  className={`preset-btn${timeOfDayMode === "dawn" ? " active" : ""}`}
                  onClick={() => {
                    setTimeOfDayMode("dawn");
                    lightingOverrideRef.current = { mode: "dawn", customHour: 6.5 };
                  }}
                  title="Dawn (06:30)"
                >
                  🌅 Dawn
                </button>
                <button
                  className={`preset-btn${timeOfDayMode === "noon" ? " active" : ""}`}
                  onClick={() => {
                    setTimeOfDayMode("noon");
                    lightingOverrideRef.current = { mode: "noon", customHour: 12.0 };
                  }}
                  title="Noon (12:00)"
                >
                  ☀️ Noon
                </button>
                <button
                  className={`preset-btn${timeOfDayMode === "dusk" ? " active" : ""}`}
                  onClick={() => {
                    setTimeOfDayMode("dusk");
                    lightingOverrideRef.current = { mode: "dusk", customHour: 19.0 };
                  }}
                  title="Dusk (19:00)"
                >
                  🌇 Dusk
                </button>
                <button
                  className={`preset-btn${timeOfDayMode === "night" ? " active" : ""}`}
                  onClick={() => {
                    setTimeOfDayMode("night");
                    lightingOverrideRef.current = { mode: "night", customHour: 23.5 };
                  }}
                  title="Night (23:30)"
                >
                  🌙 Night
                </button>
              </div>
              <div className="atmosphere-slider-group" style={{ marginTop: "8px" }}>
                <div className="slider-label-row">
                  <span className="slider-label" style={{ color: "var(--ink-muted)" }}>Scrub Solar Angle:</span>
                  <span className="slider-value-badge">
                    {timeOfDayMode === "auto"
                      ? `Sim Clock (${world?.state?.sim_time?.split(", ")[1] || "08:00"})`
                      : formatHourShort(timeOfDayMode === "custom" ? customHour : timeOfDayMode === "dawn" ? 6.5 : timeOfDayMode === "noon" ? 12 : timeOfDayMode === "dusk" ? 19 : 23.5)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="23.75"
                  step="0.25"
                  value={timeOfDayMode === "custom" ? customHour : timeOfDayMode === "dawn" ? 6.5 : timeOfDayMode === "noon" ? 12 : timeOfDayMode === "dusk" ? 19 : timeOfDayMode === "night" ? 23.5 : (world?.sim_hour ?? 8)}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setCustomHour(val);
                    setTimeOfDayMode("custom");
                    lightingOverrideRef.current = { mode: "custom", customHour: val };
                  }}
                  className="atmosphere-slider"
                />
              </div>
              <span className="hint-small" style={{ display: "block", marginTop: "6px" }}>
                Simulation ticks, actions, and diary continue at simulated pace.
              </span>
            </div>
            <div className="section-divider dev-danger-row">
              <span className="hint-small">
                Reset clears Chubby’s memories, diary, and structures back to Tick 0.
              </span>
              <button id="btn-reset" className="btn btn-danger btn-sm" onClick={handleReset}>
                Reset Lot
              </button>
            </div>
          </div>
        </div>
      </div>

      <div id="toast" className={`toast${toastMsg ? "" : " hidden"}`}>
        {toastMsg}
      </div>
    </>
  );
}
