import * as THREE from "three";

export const TOWN_HALF_SIZE = 30.0;
export const FENCE_LIMIT = 26.5;
export const SAFE_BOUNDS = 28.0;

export interface StructureSlot {
  x: number;
  z: number;
  rot: number;
}

export interface DitherUniforms {
  uDitherEnabled: { value: number };
}

export type InteractType = "status" | "diary" | "offering" | "blueprints";

export type StructureKind =
  | "periscope"
  | "scanner"
  | "windmill"
  | "chime"
  | "lights"
  | "zen"
  | "solar"
  | "workbench"
  | "canopy"
  | "radio"
  | "studio"
  | "generic";

export interface ChubbyParts {
  bodyPivot: THREE.Group;
  head: THREE.Group;
  leftArm: THREE.Group;
  rightArm: THREE.Group;
  leftWheel: THREE.Mesh;
  rightWheel: THREE.Mesh;
  leftEye: THREE.Mesh;
  rightEye: THREE.Mesh;
  eyeMat: THREE.MeshStandardMaterial;
  antennaTipMat: THREE.MeshStandardMaterial;
  antennaLight: THREE.PointLight;
}

/**
 * Injects an ordered 4x4 Bayer screen-space dither pass onto a Three.js material
 * before tone mapping so low-poly surfaces retain crisp tactile dithering.
 */
export function ditherMaterial<T extends THREE.Material>(
  material: T,
  ditherUniforms: DitherUniforms
): T {
  material.customProgramCacheKey = () => "pino-bayer-dither-v1";
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uDitherEnabled = ditherUniforms.uDitherEnabled;
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <common>",
      `#include <common>
      uniform float uDitherEnabled;
      float pinoBayer4x4(vec2 coord) {
        vec2 p = mod(floor(coord * 0.5), 4.0);
        int x = int(p.x);
        int y = int(p.y);
        int idx = y * 4 + x;
        if (idx == 0) return 0.0 / 16.0;
        if (idx == 1) return 8.0 / 16.0;
        if (idx == 2) return 2.0 / 16.0;
        if (idx == 3) return 10.0 / 16.0;
        if (idx == 4) return 12.0 / 16.0;
        if (idx == 5) return 4.0 / 16.0;
        if (idx == 6) return 14.0 / 16.0;
        if (idx == 7) return 6.0 / 16.0;
        if (idx == 8) return 3.0 / 16.0;
        if (idx == 9) return 11.0 / 16.0;
        if (idx == 10) return 1.0 / 16.0;
        if (idx == 11) return 9.0 / 16.0;
        if (idx == 12) return 15.0 / 16.0;
        if (idx == 13) return 7.0 / 16.0;
        if (idx == 14) return 13.0 / 16.0;
        return 5.0 / 16.0;
      }`
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <dithering_fragment>",
      `#include <dithering_fragment>
      if (uDitherEnabled > 0.5) {
        float b = pinoBayer4x4(gl_FragCoord.xy) - 0.5;
        float levels = 12.0;
        vec3 dithered = clamp(gl_FragColor.rgb + (b / levels), 0.0, 1.0);
        gl_FragColor.rgb = floor(dithered * levels + 0.5) / levels;
      }`
    );
  };
  return material;
}

function ditherStd(
  params: THREE.MeshStandardMaterialParameters,
  u: DitherUniforms
): THREE.MeshStandardMaterial {
  return ditherMaterial(new THREE.MeshStandardMaterial({ flatShading: true, ...params }), u);
}

/**
 * Builds the 32x32 empty fenced lot plus the three diegetic south-gate props
 * (Offering Mailbox, Field Diary Stand, and Blueprint Noticeboard).
 */
export function buildGateProps(
  u: DitherUniforms,
  mailboxDrawingTex: THREE.CanvasTexture,
  tagInteractable: (group: THREE.Object3D, type: InteractType) => void,
  initialFlagRot: number
): {
  lotGroup: THREE.Group;
  mailboxFlagPivot: THREE.Group;
  mailboxDrawingMesh: THREE.Mesh;
  blueprintPaperGroup: THREE.Group;
} {
  const lotGroup = new THREE.Group();

  const woodMat = ditherStd({ color: 0x5c4433, roughness: 0.9 }, u);

  // 1) South-gate Mailbox (Offering)
  const mailboxGroup = new THREE.Group();
  mailboxGroup.position.set(0.0, 0, FENCE_LIMIT - 0.45);
  mailboxGroup.rotation.y = Math.PI;

  const mbPost = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.95, 0.14), woodMat);
  mbPost.position.y = 0.475;
  mbPost.castShadow = true;
  mailboxGroup.add(mbPost);

  const mbBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.46, 0.32, 0.38),
    ditherStd({ color: 0x4d7c57, roughness: 0.7 }, u)
  );
  mbBody.position.y = 1.02;
  mbBody.castShadow = true;
  mailboxGroup.add(mbBody);

  const mbRoof = new THREE.Mesh(
    new THREE.CylinderGeometry(0.23, 0.23, 0.4, 8, 1, false, 0, Math.PI),
    ditherStd({ color: 0x3b6344, roughness: 0.65 }, u)
  );
  mbRoof.rotation.z = Math.PI / 2;
  mbRoof.rotation.y = Math.PI / 2;
  mbRoof.position.y = 1.18;
  mailboxGroup.add(mbRoof);

  const mailboxFlagPivot = new THREE.Group();
  mailboxFlagPivot.position.set(0.25, 1.02, 0.04);
  mailboxFlagPivot.rotation.z = initialFlagRot;
  const flagMat = ditherStd({ color: 0xc96442 }, u);
  const flagArm = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.28, 0.04), flagMat);
  flagArm.position.y = 0.14;
  mailboxFlagPivot.add(flagArm);
  const flagTip = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.11, 0.14), flagMat);
  flagTip.position.set(0, 0.25, 0.05);
  mailboxFlagPivot.add(flagTip);
  mailboxGroup.add(mailboxFlagPivot);

  const cardBack = new THREE.Mesh(
    new THREE.BoxGeometry(0.36, 0.36, 0.03),
    ditherStd({ color: 0xf4efe4 }, u)
  );
  cardBack.position.set(0, 0.64, 0.1);
  mailboxGroup.add(cardBack);

  const mailboxDrawingMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(0.3, 0.3),
    new THREE.MeshBasicMaterial({ map: mailboxDrawingTex })
  );
  mailboxDrawingMesh.position.set(0, 0.64, 0.12);
  mailboxDrawingMesh.visible = false;
  mailboxGroup.add(mailboxDrawingMesh);

  tagInteractable(mailboxGroup, "offering");
  lotGroup.add(mailboxGroup);

  // 2) Field Diary Stand
  const diaryStandGroup = new THREE.Group();
  diaryStandGroup.position.set(-1.8, 0, FENCE_LIMIT - 0.45);
  diaryStandGroup.rotation.y = Math.PI;

  const standPost = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.05, 0.12), woodMat);
  standPost.position.y = 0.525;
  standPost.castShadow = true;
  diaryStandGroup.add(standPost);

  const lecternTop = new THREE.Group();
  lecternTop.position.set(0, 1.05, 0);
  lecternTop.rotation.x = 0.42;

  const board = new THREE.Mesh(
    new THREE.BoxGeometry(0.56, 0.05, 0.42),
    ditherStd({ color: 0x6f523b }, u)
  );
  board.castShadow = true;
  lecternTop.add(board);

  const bookCover = new THREE.Mesh(
    new THREE.BoxGeometry(0.46, 0.05, 0.34),
    ditherStd({ color: 0xc96442 }, u)
  );
  bookCover.position.y = 0.04;
  lecternTop.add(bookCover);

  const bookPages = new THREE.Mesh(
    new THREE.BoxGeometry(0.42, 0.04, 0.31),
    ditherStd({ color: 0xf4efe4 }, u)
  );
  bookPages.position.y = 0.06;
  lecternTop.add(bookPages);

  diaryStandGroup.add(lecternTop);
  tagInteractable(diaryStandGroup, "diary");
  lotGroup.add(diaryStandGroup);

  // 3) Blueprint Noticeboard
  const blueprintBoardGroup = new THREE.Group();
  blueprintBoardGroup.position.set(1.8, 0, FENCE_LIMIT - 0.45);
  blueprintBoardGroup.rotation.y = Math.PI;

  for (const lx of [-0.32, 0.32]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1.35, 0.1), woodMat);
    leg.position.set(lx, 0.675, 0);
    leg.castShadow = true;
    blueprintBoardGroup.add(leg);
  }

  const corkPanel = new THREE.Mesh(
    new THREE.BoxGeometry(0.82, 0.56, 0.07),
    ditherStd({ color: 0xa48058, roughness: 0.9 }, u)
  );
  corkPanel.position.set(0, 1.15, 0);
  corkPanel.castShadow = true;
  blueprintBoardGroup.add(corkPanel);

  const blueprintPaperGroup = new THREE.Group();
  blueprintPaperGroup.visible = false;

  const blueSheet = new THREE.Mesh(
    new THREE.BoxGeometry(0.56, 0.38, 0.02),
    ditherStd({ color: 0x3b7ea1 }, u)
  );
  blueSheet.position.set(-0.04, 1.15, 0.045);
  blueSheet.rotation.z = 0.05;
  blueprintPaperGroup.add(blueSheet);

  const pinHead = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 6, 5),
    ditherStd({ color: 0xc96442 }, u)
  );
  pinHead.position.set(-0.04, 1.31, 0.065);
  blueprintPaperGroup.add(pinHead);

  const alertOrb = new THREE.Mesh(
    new THREE.SphereGeometry(0.075, 8, 6),
    new THREE.MeshStandardMaterial({
      color: 0xffd54f,
      emissive: 0xffca28,
      emissiveIntensity: 1.5,
    })
  );
  alertOrb.position.set(0, 1.54, 0.04);
  blueprintPaperGroup.add(alertOrb);

  blueprintBoardGroup.add(blueprintPaperGroup);
  tagInteractable(blueprintBoardGroup, "blueprints");
  lotGroup.add(blueprintBoardGroup);

  return { lotGroup, mailboxFlagPivot, mailboxDrawingMesh, blueprintPaperGroup };
}

/**
 * Builds the procedural Chubby robot mesh and returns articulated animation handles.
 */
export function createProceduralChubby(u: DitherUniforms): {
  root: THREE.Group;
  parts: ChubbyParts;
} {
  const root = new THREE.Group();
  const bodyMat = ditherStd({ color: 0xeae4d6, roughness: 0.55, metalness: 0.15 }, u);
  const accentMat = ditherStd({ color: 0x3d3b38, roughness: 0.6, metalness: 0.3 }, u);
  const wheelMat = ditherStd({ color: 0x24201c, roughness: 0.9 }, u);
  const eyeMat = new THREE.MeshStandardMaterial({
    color: 0x6dd5a0,
    emissive: 0x4d7c57,
    emissiveIntensity: 1.35,
    roughness: 0.25,
  });
  const antennaTipMat = new THREE.MeshStandardMaterial({
    color: 0xd9a036,
    emissive: 0xc96442,
    emissiveIntensity: 1.2,
  });

  const bodyPivot = new THREE.Group();
  root.add(bodyPivot);

  const torso = new THREE.Mesh(new THREE.SphereGeometry(0.56, 10, 8), bodyMat);
  torso.scale.set(1.06, 0.95, 1.0);
  torso.position.y = 0.72;
  torso.castShadow = true;
  torso.receiveShadow = true;
  bodyPivot.add(torso);

  const bellyPlate = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.14, 8), accentMat);
  bellyPlate.rotation.x = Math.PI / 2;
  bellyPlate.position.set(0, 0.7, 0.5);
  bellyPlate.castShadow = true;
  bodyPivot.add(bellyPlate);

  const head = new THREE.Group();
  head.position.set(0, 1.25, 0);
  bodyPivot.add(head);

  const dome = new THREE.Mesh(new THREE.SphereGeometry(0.44, 10, 8), bodyMat);
  dome.scale.set(1.0, 0.85, 0.95);
  dome.castShadow = true;
  head.add(dome);

  const visor = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.18, 0.15), accentMat);
  visor.position.set(0, 0.05, 0.38);
  head.add(visor);

  const eyeGeo = new THREE.SphereGeometry(0.065, 8, 6);
  const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
  leftEye.position.set(-0.13, 0.05, 0.44);
  head.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
  rightEye.position.set(0.13, 0.05, 0.44);
  head.add(rightEye);

  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.025, 0.32, 6), accentMat);
  mast.position.set(0, 0.48, 0);
  head.add(mast);

  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.065, 8, 6), antennaTipMat);
  tip.position.set(0, 0.65, 0);
  head.add(tip);

  const antennaLight = new THREE.PointLight(0xd9a036, 0.6, 3.5);
  antennaLight.position.set(0, 0.68, 0);
  head.add(antennaLight);

  const armGeo = new THREE.CylinderGeometry(0.055, 0.065, 0.34, 6);
  const handGeo = new THREE.SphereGeometry(0.075, 6, 5);

  const makeArm = (x: number) => {
    const arm = new THREE.Group();
    arm.position.set(x, 0.76, 0);
    const limb = new THREE.Mesh(armGeo, accentMat);
    limb.position.y = -0.17;
    arm.add(limb);
    const hand = new THREE.Mesh(handGeo, bodyMat);
    hand.position.y = -0.34;
    arm.add(hand);
    bodyPivot.add(arm);
    return arm;
  };
  const leftArm = makeArm(-0.62);
  const rightArm = makeArm(0.62);

  const wheelGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.16, 10);
  const leftWheel = new THREE.Mesh(wheelGeo, wheelMat);
  leftWheel.rotation.z = Math.PI / 2;
  leftWheel.position.set(-0.54, 0.24, 0);
  leftWheel.castShadow = true;
  root.add(leftWheel);

  const rightWheel = new THREE.Mesh(wheelGeo, wheelMat);
  rightWheel.rotation.z = Math.PI / 2;
  rightWheel.position.set(0.54, 0.24, 0);
  rightWheel.castShadow = true;
  root.add(rightWheel);

  const casterWheel = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 6), wheelMat);
  casterWheel.position.set(0, 0.1, -0.38);
  casterWheel.castShadow = true;
  root.add(casterWheel);

  return {
    root,
    parts: {
      bodyPivot,
      head,
      leftArm,
      rightArm,
      leftWheel,
      rightWheel,
      leftEye,
      rightEye,
      eyeMat,
      antennaTipMat,
      antennaLight,
    },
  };
}

/**
 * Maps a structure's name and unlocks/purpose text to its 3D procedural builder key.
 */
export function classifyStructureType(name: string, purpose: string): StructureKind {
  const combined = `${name} ${purpose}`.toLowerCase();
  if (combined.includes("studio") || combined.includes("easel")) {
    return "studio";
  }
  if (combined.includes("periscope") || combined.includes("lookout") || combined.includes("telescope")) {
    return "periscope";
  }
  if (combined.includes("scanner") || combined.includes("drafting") || combined.includes("schematic")) {
    return "scanner";
  }
  if (combined.includes("windmill") || combined.includes("turbine")) {
    return "windmill";
  }
  if (combined.includes("chime")) {
    return "chime";
  }
  if (combined.includes("string light") || combined.includes("lantern") || combined.includes("light post")) {
    return "lights";
  }
  if (combined.includes("zen") || combined.includes("pebble")) {
    return "zen";
  }
  if (combined.includes("solar") || combined.includes("charg") || combined.includes("battery")) {
    return "solar";
  }
  if (combined.includes("workbench") || combined.includes("table") || combined.includes("tool") || combined.includes("make")) {
    return "workbench";
  }
  if (combined.includes("shelter") || combined.includes("canopy") || combined.includes("roof") || combined.includes("rain")) {
    return "canopy";
  }
  if (combined.includes("radio") || combined.includes("antenna") || combined.includes("music")) {
    return "radio";
  }
  return "generic";
}

function buildFencePeriscope(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const baseMat = ditherStd({ color: 0x5c5348, roughness: 0.85 }, u);
  const brassMat = ditherStd({ color: 0xc2983d, roughness: 0.45, metalness: 0.45 }, u);
  const pipeMat = ditherStd({ color: 0x3b6e66, roughness: 0.55, metalness: 0.25 }, u);
  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x7be8d0,
    emissive: 0x39b89c,
    emissiveIntensity: 1.25,
    roughness: 0.2,
  });

  const plinth = new THREE.Mesh(new THREE.CylinderGeometry(0.68, 0.82, 0.22, 8), baseMat);
  plinth.position.y = 0.11;
  plinth.castShadow = true;
  plinth.receiveShadow = true;
  group.add(plinth);

  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.14, 2.25, 8), pipeMat);
  mast.position.y = 1.24;
  mast.castShadow = true;
  group.add(mast);

  const collarLow = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.12, 8), brassMat);
  collarLow.position.y = 0.95;
  group.add(collarLow);

  const compassDial = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 0.34), brassMat);
  compassDial.position.set(0, 1.02, 0.22);
  compassDial.rotation.x = 0.42;
  group.add(compassDial);

  const elbow = new THREE.Mesh(new THREE.SphereGeometry(0.15, 8, 6), brassMat);
  elbow.position.y = 2.38;
  elbow.castShadow = true;
  group.add(elbow);

  const hood = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.12, 0.52, 8), pipeMat);
  hood.rotation.x = Math.PI / 2;
  hood.position.set(0, 2.38, 0.22);
  hood.castShadow = true;
  group.add(hood);

  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.105, 0.105, 0.05, 8), lensMat);
  lens.rotation.x = Math.PI / 2;
  lens.position.set(0, 2.38, 0.49);
  group.add(lens);

  const eyepiece = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.075, 0.24, 6), brassMat);
  eyepiece.rotation.x = Math.PI / 2;
  eyepiece.position.set(0, 1.08, 0.2);
  group.add(eyepiece);

  return group;
}

function buildSalvageScannerTable(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const woodMat = ditherStd({ color: 0x7c583a, roughness: 0.85 }, u);
  const steelMat = ditherStd({ color: 0x454d54, roughness: 0.6, metalness: 0.3 }, u);
  const sheetMat = new THREE.MeshStandardMaterial({
    color: 0x2f749c,
    emissive: 0x1b4c6b,
    emissiveIntensity: 0.75,
    roughness: 0.4,
    flatShading: true,
  });
  const scanGlowMat = new THREE.MeshStandardMaterial({
    color: 0x7ae2ff,
    emissive: 0x3ec5f0,
    emissiveIntensity: 1.5,
  });

  const legGeo = new THREE.BoxGeometry(0.11, 0.82, 0.11);
  for (const [lx, lz] of [
    [-0.78, -0.46],
    [0.78, -0.46],
    [-0.78, 0.46],
    [0.78, 0.46],
  ]) {
    const leg = new THREE.Mesh(legGeo, steelMat);
    leg.position.set(lx, 0.41, lz);
    leg.castShadow = true;
    group.add(leg);
  }

  const tableTop = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.1, 1.15), woodMat);
  tableTop.position.set(0, 0.86, 0);
  tableTop.rotation.x = 0.14;
  tableTop.castShadow = true;
  tableTop.receiveShadow = true;
  group.add(tableTop);

  const blueprint = new THREE.Mesh(new THREE.BoxGeometry(1.12, 0.03, 0.78), sheetMat);
  blueprint.position.set(-0.15, 0.92, 0.02);
  blueprint.rotation.x = 0.14;
  group.add(blueprint);

  const armBase = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.15, 0.18, 6), steelMat);
  armBase.position.set(0.68, 0.96, -0.34);
  group.add(armBase);

  const armPost = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.68, 0.08), steelMat);
  armPost.position.set(0.68, 1.32, -0.34);
  armPost.castShadow = true;
  group.add(armPost);

  const armReach = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.07, 0.08), steelMat);
  armReach.position.set(0.28, 1.64, -0.14);
  armReach.rotation.y = 0.42;
  armReach.castShadow = true;
  group.add(armReach);

  const scannerHead = new THREE.Mesh(new THREE.ConeGeometry(0.14, 0.22, 6), scanGlowMat);
  scannerHead.position.set(-0.08, 1.54, 0.04);
  group.add(scannerHead);

  const tubeMat = ditherStd({ color: 0xe6dcc8, roughness: 0.75 }, u);
  for (let i = 0; i < 3; i++) {
    const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.56, 6), tubeMat);
    roll.rotation.z = Math.PI / 2;
    roll.position.set(0.45, 0.96 + i * 0.08, 0.26 - i * 0.06);
    group.add(roll);
  }

  return group;
}

function buildWindChimeArch(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const woodMat = ditherStd({ color: 0x855a3c, roughness: 0.88 }, u);
  const chimeMat = ditherStd({ color: 0xd6b05c, roughness: 0.35, metalness: 0.6 }, u);

  for (const px of [-0.82, 0.82]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.14, 2.35, 0.14), woodMat);
    post.position.set(px, 1.175, 0);
    post.castShadow = true;
    group.add(post);
  }

  const beam = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.14, 0.18), woodMat);
  beam.position.set(0, 2.32, 0);
  beam.castShadow = true;
  group.add(beam);

  [0.62, 0.78, 0.94, 0.72, 0.56].forEach((len, idx) => {
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, len, 6), chimeMat);
    tube.position.set(-0.44 + idx * 0.22, 2.24 - len / 2 - 0.1, 0);
    tube.castShadow = true;
    group.add(tube);
  });

  return group;
}

function buildWarmStringLights(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const postMat = ditherStd({ color: 0x5c4330, roughness: 0.9 }, u);
  const wireMat = new THREE.MeshStandardMaterial({ color: 0x262422, roughness: 0.8 });
  const bulbMat = new THREE.MeshStandardMaterial({
    color: 0xffd77a,
    emissive: 0xffab3b,
    emissiveIntensity: 1.5,
  });

  const corners: Array<[number, number]> = [
    [-0.95, 0.75],
    [0.95, 0.75],
    [0.0, -0.88],
  ];
  for (const [px, pz] of corners) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 2.15, 6), postMat);
    post.position.set(px, 1.075, pz);
    post.castShadow = true;
    group.add(post);
  }

  for (let i = 0; i < corners.length; i++) {
    const [x1, z1] = corners[i];
    const [x2, z2] = corners[(i + 1) % corners.length];
    const dx = x2 - x1;
    const dz = z2 - z1;
    const dist = Math.hypot(dx, dz);
    const wire = new THREE.Mesh(new THREE.BoxGeometry(dist, 0.025, 0.025), wireMat);
    wire.position.set((x1 + x2) / 2, 2.08, (z1 + z2) / 2);
    wire.rotation.y = -Math.atan2(dz, dx);
    group.add(wire);

    for (let b = 1; b <= 3; b++) {
      const t = b / 4;
      const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.075, 6, 5), bulbMat);
      bulb.position.set(x1 + dx * t, 2.01 - Math.sin(t * Math.PI) * 0.1, z1 + dz * t);
      group.add(bulb);
    }
  }

  return group;
}

function buildPebbleZenGarden(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const frameMat = ditherStd({ color: 0x523f31, roughness: 0.9 }, u);
  const sandMat = ditherStd({ color: 0xddd1b6, roughness: 0.95 }, u);
  const stoneMat = ditherStd({ color: 0x6f7370, roughness: 0.85 }, u);
  const mossMat = ditherStd({ color: 0x6a8d59, roughness: 0.85 }, u);

  const border = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.14, 1.75), frameMat);
  border.position.y = 0.07;
  border.castShadow = true;
  border.receiveShadow = true;
  group.add(border);

  const sand = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.16, 1.55), sandMat);
  sand.position.y = 0.08;
  sand.receiveShadow = true;
  group.add(sand);

  [
    [-0.45, -0.22, 0.28],
    [-0.22, -0.1, 0.19],
    [0.48, 0.24, 0.22],
  ].forEach(([sx, sz, r], i) => {
    const stone = new THREE.Mesh(new THREE.DodecahedronGeometry(r, 0), i === 0 ? stoneMat : mossMat);
    stone.scale.set(1.15, 0.72, 0.95);
    stone.position.set(sx, 0.2 + r * 0.25, sz);
    stone.castShadow = true;
    group.add(stone);
  });

  return group;
}

function buildMiniWindmill(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const steelMat = ditherStd({ color: 0x5e686d, roughness: 0.65, metalness: 0.25 }, u);
  const bladeMat = ditherStd({ color: 0xe6decb, roughness: 0.55 }, u);
  const beaconMat = new THREE.MeshStandardMaterial({
    color: 0xffd54f,
    emissive: 0xffb300,
    emissiveIntensity: 1.25,
  });

  const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.52, 2.3, 4), steelMat);
  tower.position.y = 1.15;
  tower.rotation.y = Math.PI / 4;
  tower.castShadow = true;
  group.add(tower);

  const nacelle = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.28, 0.52), steelMat);
  nacelle.position.set(0, 2.38, 0);
  nacelle.castShadow = true;
  group.add(nacelle);

  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.075, 6, 5), beaconMat);
  beacon.position.set(0, 2.58, -0.1);
  group.add(beacon);

  const rotor = new THREE.Group();
  rotor.position.set(0, 2.38, 0.29);
  for (let i = 0; i < 4; i++) {
    const arm = new THREE.Group();
    arm.rotation.z = (i * Math.PI) / 2 + 0.3;
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.76, 0.03), bladeMat);
    blade.position.y = 0.42;
    blade.rotation.y = 0.25;
    blade.castShadow = true;
    arm.add(blade);
    rotor.add(arm);
  }
  group.add(rotor);

  return group;
}

function buildSolarChargingPad(u: DitherUniforms): THREE.Group {
  // Town solar station: charging dock in front, tilted solar array on a mast behind.
  const group = new THREE.Group();
  const concrete = ditherStd({ color: 0x8d8f8c, roughness: 0.9 }, u);
  const metal = ditherStd({ color: 0x4b5559, roughness: 0.5, metalness: 0.5 }, u);

  const plinth = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.14, 3.4), concrete);
  plinth.position.y = 0.07;
  plinth.receiveShadow = true;
  group.add(plinth);

  const dock = new THREE.Mesh(
    new THREE.CylinderGeometry(0.95, 0.95, 0.06, 20),
    ditherStd({ color: 0x23303a, roughness: 0.4, metalness: 0.4 }, u)
  );
  dock.position.set(0, 0.17, 0.3);
  group.add(dock);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.82, 0.045, 6, 28),
    new THREE.MeshStandardMaterial({ color: 0x59e0a5, emissive: 0x2fc98a, emissiveIntensity: 1.3 })
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.set(0, 0.22, 0.3);
  group.add(ring);

  const mast = new THREE.Mesh(new THREE.BoxGeometry(0.16, 2.6, 0.16), metal);
  mast.position.set(0, 1.4, -1.3);
  mast.castShadow = true;
  group.add(mast);

  const array = new THREE.Group();
  array.position.set(0, 2.75, -1.3);
  array.rotation.x = -0.55;
  const panelMat = ditherStd({ color: 0x1f4a8a, emissive: 0x0d2447, roughness: 0.25, metalness: 0.7 }, u);
  for (const px of [-0.78, 0.78]) {
    const panel = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.07, 1.1), panelMat);
    panel.position.x = px;
    panel.castShadow = true;
    array.add(panel);
    for (let g = -1; g <= 1; g++) {
      const gridLine = new THREE.Mesh(
        new THREE.BoxGeometry(0.03, 0.01, 1.1),
        ditherStd({ color: 0x9db4d6 }, u)
      );
      gridLine.position.set(px + g * 0.45, 0.045, 0);
      array.add(gridLine);
    }
  }
  group.add(array);

  const cabinet = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 1.0, 0.5),
    ditherStd({ color: 0xd9d4c4, roughness: 0.6 }, u)
  );
  cabinet.position.set(1.2, 0.64, -1.0);
  cabinet.castShadow = true;
  group.add(cabinet);

  const statusLight = new THREE.Mesh(
    new THREE.SphereGeometry(0.09, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0xffd54f, emissive: 0xffca28, emissiveIntensity: 1.5 })
  );
  statusLight.position.set(1.2, 1.25, -1.0);
  group.add(statusLight);

  const glow = new THREE.PointLight(0x59e0a5, 0.8, 5.0);
  glow.position.set(0, 0.8, 0.3);
  group.add(glow);

  return group;
}

function buildArtStudioMarker(u: DitherUniforms): THREE.Group {
  // The real Art Studio is a town building (lib/townScene.ts); this is a fallback easel.
  const group = new THREE.Group();
  const wood = ditherStd({ color: 0x8a6a4a, roughness: 0.85 }, u);
  for (const lx of [-0.4, 0.4]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 1.6, 0.07), wood);
    leg.position.set(lx, 0.8, 0);
    leg.rotation.z = lx > 0 ? -0.1 : 0.1;
    group.add(leg);
  }
  const canvas = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.7, 0.05), ditherStd({ color: 0xf4efe4 }, u));
  canvas.position.set(0, 1.1, 0.05);
  group.add(canvas);
  return group;
}

function buildScrapWorkbench(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.9, 0.14, 1.1),
    ditherStd({ color: 0x6f4f3a, roughness: 0.85 }, u)
  );
  top.position.y = 0.88;
  top.castShadow = true;
  top.receiveShadow = true;
  group.add(top);

  const legGeo = new THREE.BoxGeometry(0.12, 0.88, 0.12);
  const legMat = ditherStd({ color: 0x454947, metalness: 0.4 }, u);
  for (const [lx, lz] of [
    [-0.82, -0.42],
    [0.82, -0.42],
    [-0.82, 0.42],
    [0.82, 0.42],
  ]) {
    const leg = new THREE.Mesh(legGeo, legMat);
    leg.position.set(lx, 0.44, lz);
    leg.castShadow = true;
    group.add(leg);
  }

  const peg = new THREE.Mesh(
    new THREE.BoxGeometry(1.85, 0.85, 0.06),
    ditherStd({ color: 0x7a7f7c }, u)
  );
  peg.position.set(0, 1.35, -0.5);
  peg.castShadow = true;
  group.add(peg);

  const vise = new THREE.Mesh(
    new THREE.BoxGeometry(0.24, 0.22, 0.22),
    ditherStd({ color: 0xc96442, metalness: 0.4 }, u)
  );
  vise.position.set(0.68, 1.05, 0.25);
  vise.castShadow = true;
  group.add(vise);

  return group;
}

function buildRainShelterCanopy(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const poleGeo = new THREE.CylinderGeometry(0.06, 0.08, 2.3, 6);
  const poleMat = ditherStd({ color: 0x5c5349, metalness: 0.25 }, u);

  for (const [px, pz] of [
    [-1.2, -1.0],
    [1.2, -1.0],
    [-1.2, 1.0],
    [1.2, 1.0],
  ]) {
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.set(px, 1.15, pz);
    pole.castShadow = true;
    group.add(pole);
  }

  const roof = new THREE.Mesh(
    new THREE.ConeGeometry(1.9, 0.65, 4),
    ditherStd({ color: 0x9b5a42, roughness: 0.8 }, u)
  );
  roof.rotation.y = Math.PI / 4;
  roof.position.y = 2.45;
  roof.castShadow = true;
  group.add(roof);

  const matFloor = new THREE.Mesh(
    new THREE.BoxGeometry(2.1, 0.06, 1.8),
    ditherStd({ color: 0x585f57, roughness: 0.9 }, u)
  );
  matFloor.position.y = 0.03;
  matFloor.receiveShadow = true;
  group.add(matFloor);

  return group;
}

function buildTinyRadioCorner(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const table = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 0.55, 0.8),
    ditherStd({ color: 0x5c4433, roughness: 0.9 }, u)
  );
  table.position.y = 0.28;
  table.castShadow = true;
  group.add(table);

  const radio = new THREE.Mesh(
    new THREE.BoxGeometry(0.55, 0.35, 0.38),
    ditherStd({ color: 0x4d7c57, metalness: 0.2 }, u)
  );
  radio.position.set(0, 0.72, 0);
  radio.castShadow = true;
  group.add(radio);

  const mast = new THREE.Mesh(
    new THREE.CylinderGeometry(0.02, 0.03, 1.7, 6),
    ditherStd({ color: 0xb5b8b1, metalness: 0.7 }, u)
  );
  mast.position.set(0.18, 1.6, -0.08);
  mast.castShadow = true;
  group.add(mast);

  const dish = new THREE.Mesh(
    new THREE.CylinderGeometry(0.32, 0.05, 0.12, 8),
    ditherStd({ color: 0x8b908c, metalness: 0.5 }, u)
  );
  dish.rotation.x = Math.PI / 3;
  dish.position.set(0.18, 2.35, -0.08);
  group.add(dish);

  const redLight = new THREE.Mesh(
    new THREE.SphereGeometry(0.06, 6, 6),
    new THREE.MeshStandardMaterial({ color: 0xef5350, emissive: 0xe53935, emissiveIntensity: 1.8 })
  );
  redLight.position.set(0.18, 2.5, -0.08);
  group.add(redLight);

  return group;
}

function buildGenericStructure(u: DitherUniforms): THREE.Group {
  const group = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.22, 1.5),
    ditherStd({ color: 0x67706d, metalness: 0.35 }, u)
  );
  base.position.y = 0.11;
  base.castShadow = true;
  group.add(base);

  const column = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.24, 1.4, 8),
    ditherStd({ color: 0x8f948e, metalness: 0.5 }, u)
  );
  column.position.y = 0.85;
  column.castShadow = true;
  group.add(column);

  const orb = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.35),
    new THREE.MeshStandardMaterial({
      color: 0xd9a036,
      emissive: 0xc96442,
      emissiveIntensity: 1.2,
      flatShading: true,
    })
  );
  orb.position.y = 1.8;
  group.add(orb);

  return group;
}

const STRUCTURE_BUILDERS: Record<StructureKind, (u: DitherUniforms) => THREE.Group> = {
  periscope: buildFencePeriscope,
  scanner: buildSalvageScannerTable,
  windmill: buildMiniWindmill,
  chime: buildWindChimeArch,
  lights: buildWarmStringLights,
  zen: buildPebbleZenGarden,
  solar: buildSolarChargingPad,
  workbench: buildScrapWorkbench,
  canopy: buildRainShelterCanopy,
  radio: buildTinyRadioCorner,
  studio: buildArtStudioMarker,
  generic: buildGenericStructure,
};

/**
 * Instantiates a procedural 3D structure model matching the classified structure kind.
 */
export function createStructureModel(kind: StructureKind, u: DitherUniforms): THREE.Group {
  return STRUCTURE_BUILDERS[kind](u);
}

/**
 * Disposes all geometries and materials within a Three.js scene subtree.
 */
export function disposeSceneSubtree(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (mesh.geometry) {
      mesh.geometry.dispose();
    }
    if (mesh.material) {
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((m) => m.dispose());
      } else {
        mesh.material.dispose();
      }
    }
  });
}
