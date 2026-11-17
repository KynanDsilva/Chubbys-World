# Custom 3D Models & Audio Drop-In Guide

While temporary optimized low-poly procedural models are active, you can replace any character, structure, or sound with your own files at any time.

## Folders

* `web/assets/models/` — Place `.glb` (glTF binary) 3D models here:
  * `chubby.glb` — Replaces the procedural low-poly Chubby robot.
  * `solar_charging_pad.glb` — Used for approved solar/charging pad structures.
  * `scrap_workbench.glb` — Used for approved workbench structures.
  * `rain_shelter_canopy.glb` — Used for approved rain canopy / shelter structures.
  * `tiny_radio_corner.glb` — Used for approved radio / music corner structures.
* `web/assets/sounds/` — Place optional `.mp3` or `.wav` audio loops/effects here (e.g., `rain_loop.mp3`, `message_chime.wav`).

## Wiring Custom Models

Open `web/app.js` and set the corresponding entry in `ASSET_MANIFEST` to the asset URL (for example, `chubby: "/assets/models/chubby.glb"`). When set, `app.js` loads the `.glb` using `GLTFLoader` and falls back to the temporary low-poly procedural mesh if loading fails or is left `null`.
