-- Dimensional Light: export the selected photo, open the live-preview app,
-- then import the relit result next to the original (stacked above it).

local LrApplication   = import "LrApplication"
local LrDialogs       = import "LrDialogs"
local LrExportSession = import "LrExportSession"
local LrFileUtils     = import "LrFileUtils"
local LrPathUtils     = import "LrPathUtils"
local LrTasks         = import "LrTasks"
local LrFunctionContext = import "LrFunctionContext"
local LrProgressScope = import "LrProgressScope"

local function q(s) return '"' .. s .. '"' end

local function uniquePath(dir, base, ext)
  local path = LrPathUtils.child(dir, base .. "-DimLight." .. ext)
  local n = 2
  while LrFileUtils.exists(path) do
    path = LrPathUtils.child(dir, base .. "-DimLight-" .. n .. "." .. ext)
    n = n + 1
  end
  return path
end

LrTasks.startAsyncTask(function()
  local catalog = LrApplication.activeCatalog()
  local photo = catalog:getTargetPhoto()
  if not photo then
    LrDialogs.message("Dimensional Light", "Select a photo first.", "info")
    return
  end

  -- locate the companion app (…/DimensionalLight/app) next to this plugin folder
  local root = LrPathUtils.parent(_PLUGIN.path)
  local appPy = LrPathUtils.child(LrPathUtils.child(root, "app"), "app.py")
  local python
  if WIN_ENV then
    python = LrPathUtils.child(LrPathUtils.child(LrPathUtils.child(root, ".venv"), "Scripts"), "pythonw.exe")
  else
    python = LrPathUtils.child(LrPathUtils.child(LrPathUtils.child(root, ".venv"), "bin"), "python3")
  end
  if not LrFileUtils.exists(python) or not LrFileUtils.exists(appPy) then
    LrDialogs.message("Dimensional Light",
      "Companion app not found.\n\nRun install.ps1 in:\n" .. root, "critical")
    return
  end

  local origPath = photo:getRawMetadata("path")
  local origDir  = LrPathUtils.parent(origPath)
  local base     = LrPathUtils.removeExtension(LrPathUtils.leafName(origPath))
  local outPath  = uniquePath(origDir, base, "tif")

  -- 1. render the photo (with its current Lightroom edits) to a temp 16-bit TIFF
  local tmpDir = LrPathUtils.child(LrPathUtils.getStandardFilePath("temp"), "DimensionalLight")
  LrFileUtils.createAllDirectories(tmpDir)

  local inPath
  local progress = LrProgressScope({ title = "Dimensional Light: preparing photo…" })
  local session = LrExportSession({
    photosToExport = { photo },
    exportSettings = {
      LR_format = "TIFF",
      LR_export_bitDepth = 16,
      LR_export_colorSpace = "sRGB",
      LR_export_destinationType = "specificFolder",
      LR_export_destinationPathPrefix = tmpDir,
      LR_export_useSubfolder = false,
      LR_collisionHandling = "overwrite",
      LR_renamingTokensOn = true,
      LR_tokens = "dimlight_input",
      LR_size_doNotEnlarge = true,
      LR_reimportExportedPhoto = false,
      LR_tiff_compressionMethod = "compressionMethod_None",
      LR_metadata_keywordOptions = "flat",
      LR_removeLocationMetadata = false,
    },
  })
  for _, rendition in session:renditions() do
    local ok, pathOrMsg = rendition:waitForRender()
    if ok then inPath = pathOrMsg else
      progress:done()
      LrDialogs.message("Dimensional Light", "Export failed:\n" .. tostring(pathOrMsg), "critical")
      return
    end
  end
  progress:done()
  if not inPath then return end

  -- 2. run the live-preview app (blocks until the user applies or cancels)
  local cmd = q(python) .. " " .. q(appPy) .. " --input " .. q(inPath) .. " --output " .. q(outPath)
  if WIN_ENV then cmd = '"' .. cmd .. '"' end
  LrTasks.execute(cmd)

  -- 3. import the result and stack it above the original
  if LrFileUtils.exists(outPath) then
    catalog:withWriteAccessDo("Dimensional Light import", function()
      local added = catalog:addPhoto(outPath, photo, "above")
      if added then catalog:setSelectedPhotos(added, { added }) end
    end)
  end

  LrFileUtils.delete(inPath)
  LrFileUtils.delete(inPath .. ".large.v2.depth.npy")
  LrFileUtils.delete(inPath .. ".depthpro.v2.depth.npy")
  LrFileUtils.delete(inPath .. ".ultra.v2.depth.npy")
end)
