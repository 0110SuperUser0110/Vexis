import unreal

SOURCE = r"C:\Users\Richard\Downloads\uploads_files_3533140_female_head.fbx"
DEST_PATH = "/Game/VexisAvatar"
DEST_NAME = "VexisHead"

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
editor_assets = unreal.EditorAssetLibrary

if not editor_assets.does_directory_exist(DEST_PATH):
    editor_assets.make_directory(DEST_PATH)

existing = f"{DEST_PATH}/{DEST_NAME}.{DEST_NAME}"
if editor_assets.does_asset_exist(existing):
    unreal.log(f"Asset already exists: {existing}")
else:
    task = unreal.AssetImportTask()
    task.filename = SOURCE
    task.destination_path = DEST_PATH
    task.destination_name = DEST_NAME
    task.replace_existing = True
    task.automated = True
    task.save = True

    options = unreal.FbxImportUI()
    options.import_mesh = True
    options.import_as_skeletal = False
    options.import_materials = False
    options.import_textures = False
    options.import_animations = False
    options.static_mesh_import_data.combine_meshes = True
    options.static_mesh_import_data.auto_generate_collision = False
    task.options = options

    asset_tools.import_asset_tasks([task])
    unreal.log(f"Imported asset to {DEST_PATH}")

editor_assets.save_directory(DEST_PATH, only_if_is_dirty=False, recursive=True)
for asset in editor_assets.list_assets(DEST_PATH, recursive=True, include_folder=False):
    unreal.log(asset)
