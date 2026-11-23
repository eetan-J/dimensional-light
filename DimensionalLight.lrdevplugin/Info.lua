return {
  LrSdkVersion = 6.0,
  LrSdkMinimumVersion = 6.0,
  LrToolkitIdentifier = "com.dimensionallight.lightroom",
  LrPluginName = "Dimensional Light",

  -- Library > Plug-in Extras
  LrLibraryMenuItems = {
    {
      title = "Dimensional Light...",
      file = "Launch.lua",
    },
  },

  -- File > Plug-in Extras (works from any module)
  LrExportMenuItems = {
    {
      title = "Dimensional Light...",
      file = "Launch.lua",
    },
  },

  VERSION = { major = 1, minor = 0, revision = 1 },
}
