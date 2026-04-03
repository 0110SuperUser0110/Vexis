# Unreal Presence Setup

## What is wired now
- Vexis Python runs as the backend/controller.
- Unreal polls the live bridge at `http://127.0.0.1:8765/v1/presence`.
- The Unreal project spawns `AVexisPresenceActor` automatically through `AVexisPresenceGameMode`.
- The actor responds to live states such as `idle`, `thinking`, `speaking`, and `dancing`.

## Start the stack
- Backend only: `powershell -ExecutionPolicy Bypass -File E:\Vexis\tools\start_vexis_backend_unreal.ps1`
- Backend + Unreal Editor: `powershell -ExecutionPolicy Bypass -File E:\Vexis\tools\start_vexis_unreal.ps1`

## Import the head asset
1. Open `/E:/Vexis/unreal/VexisPresence/VexisPresence.uproject` in Unreal.
2. Import your FBX head asset into the Content Browser.
3. Enter the default map and drag a `VexisPresenceActor` into the level.
4. In the actor's `AvatarMesh` component, assign the imported mesh.
5. Adjust transform and scale until the head sits correctly in frame.
6. Save the map.

Notes:
- If a `VexisPresenceActor` is already in the level, the game mode will not spawn a duplicate.
- If there is no actor placed in the level, the game mode spawns one at the world origin.
- The current actor uses a static mesh component. If you later move to a rigged skeletal avatar, swap the component type or create a dedicated skeletal actor class.

## What to test
- Start the backend and Unreal.
- Say `hello` and confirm the actor idles instead of staying frozen.
- Ask a question and confirm Unreal shifts into `thinking` then `speaking`.
- Say `dance` and confirm the actor uses the stronger motion state.

## Thesis snapshot note
- The repo snapshot is intended to show current architecture, Python core progress, tests, and Unreal integration source.
- Generated Unreal output (`Binaries`, `DerivedDataCache`, `Intermediate`, `Saved`) is not part of the review snapshot.
- The current Unreal presence code expects local character/content assets during development; third-party content packs are not included in the GitHub push.
