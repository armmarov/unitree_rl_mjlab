NAME        ?=
ROBOT       ?= pm01
MOTIONS_DIR := src/assets/motions/$(ROBOT)
CSV         := $(MOTIONS_DIR)/$(NAME).csv
NPZ         := $(MOTIONS_DIR)/$(NAME).npz
FIXED_CSV   := $(MOTIONS_DIR)/$(NAME)_fixed.csv
FINAL_CSV   := $(MOTIONS_DIR)/$(NAME)_final.csv
FINAL_NPZ   := $(MOTIONS_DIR)/$(NAME)_final.npz

INPUT_FPS   ?= 30
OUTPUT_FPS  ?= 100
HEIGHT_OFFSET ?= 0.25
NUM_ENVS    ?= 4096
HOLD_SECS   ?= 0.5
TRANS_SECS  ?= 1.5
START_FRAME ?=
AUTO_BEST   ?=
ALIGN_YAW   ?=
PREPEND_FLAGS := --hold-seconds $(HOLD_SECS) --transition-seconds $(TRANS_SECS) --fps $(INPUT_FPS)
ifneq ($(strip $(START_FRAME)),)
PREPEND_FLAGS += --start-frame $(START_FRAME)
endif
ifneq ($(strip $(AUTO_BEST)),)
PREPEND_FLAGS += --auto-best-entry
endif
ifneq ($(strip $(ALIGN_YAW)),)
PREPEND_FLAGS += --align-yaw
endif
TASK        ?= EngineAI-PM01-Tracking
EXPORT_TASK ?= pm01_tracking
RUN_NAME    ?=
PY          ?= python

# Default motion file for train/play (use _final if exists, else NAME.npz).
MOTION_FILE ?= $(if $(wildcard $(FINAL_NPZ)),$(FINAL_NPZ),$(NPZ))

.PHONY: help fix-orientation prepend csv-to-npz prep visualize \
        train play export mnn check-name check-csv check-run

help:
	@echo "Targets (specify NAME=<motion_basename>):"
	@echo ""
	@echo "  make prep         NAME=Jog_3_stageii"
	@echo "      Fix orientation + height, prepend standing, then convert CSV->NPZ."
	@echo "      Output: $(MOTIONS_DIR)/<NAME>_final.npz"
	@echo ""
	@echo "  make fix-orientation NAME=<name>"
	@echo "      Fix root yaw and height offset only."
	@echo ""
	@echo "  make prepend NAME=<name>"
	@echo "      Prepend standing-to-motion transition only."
	@echo ""
	@echo "  make csv-to-npz NAME=<name>"
	@echo "      Convert CSV to NPZ (uses _final.csv if exists)."
	@echo ""
	@echo "  make visualize NAME=<name>"
	@echo "      Play with --agent zero to verify motion (no terminations)."
	@echo ""
	@echo "  make train    NAME=<name>"
	@echo "      Train tracking policy on the motion."
	@echo ""
	@echo "  make play     NAME=<name> RUN_NAME=<timestamp>"
	@echo "      Play the trained policy."
	@echo ""
	@echo "  make export   RUN_NAME=<timestamp>"
	@echo "      Export ONNX/MNN release for the run (full release folder)."
	@echo ""
	@echo "  make mnn      ONNX_FILE=<path/to/policy.onnx> [MNN_OUT=<output.mnn>]"
	@echo "      Convert a single ONNX file to MNN format."
	@echo ""
	@echo "Optional overrides:"
	@echo "  ROBOT=pm01|g1               (default: pm01)"
	@echo "  INPUT_FPS=30  OUTPUT_FPS=100"
	@echo "  HEIGHT_OFFSET=0.25"
	@echo "  HOLD_SECS=0.5  TRANS_SECS=1.5    Standing hold + transition seconds"
	@echo "  START_FRAME=<n>             Use motion frame n as entry (skip frames 0..n-1)"
	@echo "  AUTO_BEST=1                 Auto-pick motion frame closest to standing pose"
	@echo "  NUM_ENVS=4096"
	@echo "  TASK=EngineAI-PM01-Tracking"
	@echo "  EXPORT_TASK=pm01_tracking"

check-name:
	@if [ -z "$(NAME)" ]; then echo "ERROR: pass NAME=<motion_basename>"; exit 1; fi

check-csv: check-name
	@if [ ! -f "$(CSV)" ]; then echo "ERROR: CSV not found: $(CSV)"; exit 1; fi

check-run:
	@if [ -z "$(RUN_NAME)" ]; then echo "ERROR: pass RUN_NAME=<timestamp>"; exit 1; fi

fix-orientation: check-csv
	@echo "==> Fix orientation + height ($(HEIGHT_OFFSET)m): $(CSV) -> $(FIXED_CSV)"
	$(PY) scripts/fix_motion_orientation.py $(CSV) \
		--output-csv $(FIXED_CSV) \
		--height-offset $(HEIGHT_OFFSET)

prepend: check-name
	@if [ ! -f "$(FIXED_CSV)" ]; then echo "ERROR: $(FIXED_CSV) not found. Run 'make fix-orientation' first."; exit 1; fi
	@echo "==> Prepend standing transition: $(FIXED_CSV) -> $(FINAL_CSV)"
	$(PY) scripts/prepend_standing.py $(FIXED_CSV) \
		--output-csv $(FINAL_CSV) \
		$(PREPEND_FLAGS)

csv-to-npz: check-name
	@INPUT=$(if $(wildcard $(FINAL_CSV)),$(FINAL_CSV),$(CSV)); \
	OUTPUT=$(if $(wildcard $(FINAL_CSV)),$(NAME)_final.npz,$(NAME).npz); \
	echo "==> CSV -> NPZ: $$INPUT -> $(MOTIONS_DIR)/$$OUTPUT"; \
	$(PY) scripts/csv_to_npz.py \
		--robot $(ROBOT) \
		--input-file $$INPUT \
		--output-name $$OUTPUT \
		--input-fps $(INPUT_FPS) \
		--output-fps $(OUTPUT_FPS)

prep: fix-orientation prepend csv-to-npz
	@echo "==> Done: $(FINAL_NPZ)"

visualize: check-name
	@echo "==> Visualize motion (no policy): $(MOTION_FILE)"
	$(PY) scripts/play.py $(TASK) \
		--motion-file $(MOTION_FILE) \
		--agent zero \
		--no-terminations True

train: check-name
	@echo "==> Train: $(TASK) on $(MOTION_FILE)"
	$(PY) scripts/train.py $(TASK) \
		--motion-file $(MOTION_FILE) \
		--env.scene.num-envs $(NUM_ENVS)

play: check-name check-run
	@CKPT=$$(ls logs/rsl_rl/$(EXPORT_TASK)/$(RUN_NAME)/model_*.pt | awk -F'model_|\\.pt' '{printf "%010d %s\n", $$2, $$0}' | sort -n | tail -1 | cut -d' ' -f2-); \
	echo "==> Play: $$CKPT"; \
	$(PY) scripts/play.py $(TASK) \
		--motion-file $(MOTION_FILE) \
		--checkpoint-file $$CKPT

export: check-run
	@echo "==> Export release: $(EXPORT_TASK)/$(RUN_NAME)"
	$(PY) scripts/export_release.py $(EXPORT_TASK) $(RUN_NAME)

# Standalone MNN conversion (also runs as part of `export`).
MNN_CONVERT ?= $(shell command -v mnnconvert 2>/dev/null || echo /home/armmarov/work/robot/engineai/engineai_rl_workspace/venv/bin/mnnconvert)
ONNX_FILE   ?=
MNN_OUT     ?= $(basename $(ONNX_FILE)).mnn

mnn:
	@if [ -z "$(ONNX_FILE)" ]; then \
		echo "ERROR: pass ONNX_FILE=<path/to/policy.onnx>"; \
		echo "Example: make mnn ONNX_FILE=logs/rsl_rl/pm01_tracking/<run>/policy.onnx"; \
		exit 1; fi
	@if [ ! -f "$(ONNX_FILE)" ]; then echo "ERROR: ONNX not found: $(ONNX_FILE)"; exit 1; fi
	@if [ ! -x "$(MNN_CONVERT)" ] && ! command -v $(MNN_CONVERT) >/dev/null 2>&1; then \
		echo "ERROR: mnnconvert not found. Install via: pip install MNN"; exit 1; fi
	@echo "==> ONNX -> MNN: $(ONNX_FILE) -> $(MNN_OUT)"
	$(MNN_CONVERT) -f ONNX --modelFile $(ONNX_FILE) --MNNModel $(MNN_OUT) --bizCode biz
	@rm -f .__convert_external_data.bin
	@echo "==> Done: $(MNN_OUT)"
