# Copilot / AI Agent Instructions for Qiaojia repo

Purpose: give an AI coding agent the immediate, actionable knowledge needed to work productively in this repository.

Big picture
- This repo builds a high-resolution earthquake catalog using a detection pipeline + relocation tools.
- Major components: detection/generation (`inference_final.py`, `generator/`), association (`GammaAssociation/`), conversion and HypoDD orchestration (`gamma2HypoDD/`), hypoinverse+HypoDD variants (`gamma2hypoinverse/`), and postprocessing notebooks (`postprocessing/`).

Key entrypoints & workflows
- Run detection (CPU/GPU): edit `inference_final.py` to point `input_hdf5`, `input_model`, and run `python inference_final.py` from repo root. The script calls `generator.predictor_final.predictor`.
- Convert picks -> HypoDD: see `gamma2HypoDD/run_gamma2hypodd.py` which runs `convert_stations.py`, `convert_picks()` and produces chunked `tmp_XX` folders and `hypoDD.pha` files.
- Build & run HypoDD binaries: compile with `make -C gamma2HypoDD/HYPODD/src` (Fortran toolchain required). The scripts expect `HYPODD/src/ph2dt/ph2dt` and `HYPODD/src/hypoDD/hypoDD` to be present and executable.
- Alternative hypoinverse flow: check `gamma2hypoinverse/run.sh` and `gamma2hypoinverse/gamma2hypoinverse.py` for the upstream hypoinverse → HypoDD steps.

Project-specific conventions & patterns
- Many scripts produce chunk directories named `tmp_00`, `tmp_01`, …; downstream processing expects `hypoDD_<idx>.reloc`, `dt_<idx>.ct`, `event_<idx>.sel` and then concatenates them into `hypoDD_catalog.txt`.
- Several scripts use absolute Windows-style paths (e.g. `D:/...`) — update to POSIX paths when running on macOS/Linux.
- External binaries are invoked via `os.system` and `cp`/`cat` shell calls; ensure binaries are built and available relative to script directories.
- Detection pipeline expects HDF5 inputs and Keras `.h5` models; see `inference_final.py` for example call signatures and parameters like `detection_threshold`, `P_threshold`, `S_threshold`, `batch_size`.

Integration points & external dependencies
- Python dependencies: `fastapi`, `obspy`, `uvicorn`, `pandas` (listed in `Readme.md`). Also common ML libs (TensorFlow/Keras or PyTorch) implied by `.h5` models in `generator/`.
- Native dependency: HypoDD (Fortran/C) — must be compiled and available at `gamma2HypoDD/HYPODD/src` or `gamma2hypoinverse/HYPODD/src`.
- Data: HDF5 detection inputs, CSV picks/catalogs under `GammaAssociation/` or `AssResults_*` external folders referenced in scripts.

Common pitfalls for an agent to avoid
- Do not assume POSIX paths; search for `D:/` and adjust when running on macOS.
- Do not attempt to run HypoDD steps unless `HYPODD` binaries exist; instead, surface the missing binary error and suggest `make -C gamma2HypoDD/HYPODD/src`.
- Many scripts modify cwd and rely on relative file layout; preserve `os.chdir` logic or update call sites when refactoring.

Small examples (copy-paste)
- Run detection (after editing paths):

```bash
python inference_final.py
```

- Compile HypoDD and run the orchestration:

```bash
make -C gamma2HypoDD/HYPODD/src
python gamma2HypoDD/run_gamma2hypodd.py
```

Files to inspect first
- `Readme.md` (repo overview)
- `inference_final.py` (detection example)
- `generator/predictor_final.py` (detection implementation)
- `gamma2HypoDD/run_gamma2hypodd.py` and `gamma2HypoDD/convert_stations.py` (HypoDD orchestration)
- `gamma2hypoinverse/gamma2hypoinverse.py` and `gamma2hypoinverse/run.sh` (alternate flow)

If you need to change behavior
- Prefer editing path constants in top of `inference_final.py` or provide a small CLI wrapper to avoid editing files in-place.
- When batching changes that affect working directories, run end-to-end on a small subset (`MAXEVENT=100`) before full runs.

Questions for the maintainers (ask the user)
- Where are the canonical input datasets located on the current machine (HDF5 and pick CSVs)?
- Should we normalize Windows paths to POSIX globally or wrap with a path-mapping helper?

End of instructions.
