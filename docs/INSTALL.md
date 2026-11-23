# Installing Dimensional Light

This takes about 10-20 minutes, mostly downloading. You only do it once.

## 1. Check the requirements

| | |
|---|---|
| **Windows** | 10 or 11 (64-bit) |
| **Lightroom Classic** | Any recent version. The cloud-based "Lightroom" app doesn't support plugins. |
| **Python** | 3.10, 3.11, 3.12 or 3.13 |
| **GPU** | NVIDIA strongly recommended (GTX 10-series or newer, RTX 50-series supported). Without one it still works, but depth takes minutes instead of seconds. |
| **Disk space** | About 10 GB (Python packages plus about 6 GB of AI models) |

**Don't have Python?** Download it from <https://www.python.org/downloads/>. On the first screen of the installer, **tick "Add python.exe to PATH"**, then click *Install Now*.

**NVIDIA driver:** make sure it's reasonably up to date (GeForce Experience / NVIDIA App, or <https://www.nvidia.com/drivers>).

## 2. Download

1. Go to the [latest release](https://github.com/eetan-J/dimensional-light/releases/latest).
2. Download **`DimensionalLight-vX.Y.Z.zip`** under *Assets*.
3. **Extract it to a permanent folder**, for example `C:\Tools\DimensionalLight`.
   - Don't run it from inside the zip or from Downloads if you clean that folder; Lightroom keeps pointing at this location.
   - A path without OneDrive sync is best, because the environment folder gets large.

You should now have a folder that contains `Install.bat`, `app`, `DimensionalLight.lrdevplugin` and `docs`.

## 3. Run the installer

**Double-click `Install.bat`.** A window opens and shows each step:

1. It finds Python.
2. It creates a private environment (`.venv`) inside the folder, so your other Python setups aren't touched.
3. It installs PyTorch: the CUDA build if an NVIDIA GPU is found, the CPU build otherwise.
4. It installs the other packages.
5. It downloads the depth models: Fast (~1.3 GB), Pro (~1.9 GB) and Ultra (~2.5 GB).
6. It runs a self-check and prints **"Installed successfully."** with the plugin path.

If Windows SmartScreen warns about the file, click *More info > Run anyway*. The script is plain text, so you can open `install.ps1` in Notepad and read it first.

**Options** (run from a terminal in the folder):

```powershell
.\Install.bat -SkipUltra   # skip the 2.5 GB Ultra model (it downloads later if you pick Ultra)
.\Install.bat -Cpu         # force the CPU build of PyTorch
```

Running the installer again is safe. Finished steps are skipped, so it also works as a repair.

## 4. Add the plugin to Lightroom Classic

1. Open Lightroom Classic and go to **File > Plug-in Manager...**
2. Click **Add** (bottom left).
3. Select the **`DimensionalLight.lrdevplugin`** folder inside your install folder and click *Select Folder*.
4. "Dimensional Light" appears in the list with status **Installed and running**. Click **Done**.

Don't move or rename the install folder afterwards. The plugin finds the app through the folder next to it. If you do move it, remove the plugin in the Plug-in Manager and add it again from the new location.

## 5. First use

1. Select a photo (Library or Develop).
2. Choose **File > Plug-in Extras > Dimensional Light...** (also under **Library > Plug-in Extras**).
3. Lightroom renders the photo with your current edits, and the Dimensional Light window opens.
   - The first time, loading the AI model takes a few extra seconds.
4. Edit, then click **Apply to Lightroom**. A 16-bit TIFF named `yourphoto-DimLight.tif` is saved next to the original and added to your catalog, stacked above it.

You can try the app without Lightroom from a terminal in the install folder:

```powershell
.venv\Scripts\python.exe app\app.py --demo
.venv\Scripts\python.exe app\app.py --input C:\path\photo.jpg --output C:\path\photo-relit.tif
```

## Troubleshooting

| Problem | Fix |
|---|---|
| *"Python 3.10-3.13 was not found"* | Install Python (step 1) with **Add to PATH** ticked, close the window and run `Install.bat` again. |
| *"Companion app not found. Run install.ps1"* in Lightroom | The installer hasn't finished, or the folder was moved. Run `Install.bat` again, then re-add the plugin from its current location. |
| Plug-in Extras shows nothing | In the Plug-in Manager, select Dimensional Light and click **Reload Plug-in**. If it shows an error, remove it and add it again. |
| Very slow (minutes) | No GPU is being used. In the installer output, check that the self-check line shows your NVIDIA GPU. Update the NVIDIA driver and run `Install.bat` again. |
| Download stopped or failed | Run `Install.bat` again; it resumes. Hugging Face can rate-limit busy times, so wait a minute and retry. |
| `CUDA out of memory` with Ultra | Ultra needs about 6-8 GB of VRAM. Close other GPU apps or use Pro. |
| Window opens and closes, nothing imported | Check the log at `%TEMP%\dimlight.log` and include it if you open an issue. |

Still stuck? [Open an issue](https://github.com/eetan-J/dimensional-light/issues) with the installer output or `%TEMP%\dimlight.log`.

## Updating

Download the new release and extract it **over** your install folder, replacing files. Then run `Install.bat` again; it only installs what changed. Your `.venv` and models are reused.

## Uninstalling

1. Lightroom: **File > Plug-in Manager**, select Dimensional Light, click **Remove**.
2. Delete the install folder.
3. Optional: free about 6 GB by deleting the downloaded models in `%USERPROFILE%\.cache\huggingface\hub` (the `depth-anything`, `apple--DepthPro` and `prs-eth--marigold` folders).
