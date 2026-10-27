# Laptop shortcuts. Everything is offline; nothing here needs a GPU or an API key.
PY ?= python
OUT ?= out

.PHONY: install test sim validate demo serve lint docs-check package-smoke clean site check-site train eval-policy serve-policy

install:
	$(PY) -m pip install -e ".[dev,sim,lint]"

test:
	$(PY) -m pytest -q -p no:cacheprovider

# Tiny physics sim (one person, 64 px frames); real runs use more episodes and --res 128.
sim:
	$(PY) -m arrival sim --episodes 2 --people 1 --res 64 --out $(OUT)

validate:
	$(PY) -m arrival validate $(OUT)/datasets/hero_kitchen_sim

# Offline end to end (about 20 s on a laptop, under 1 GB): scan, twin, sim dataset, validate, eval, contact
# sheet. Picks a working MuJoCo GL backend, refuses network access, ignores API keys, and writes
# $(OUT)/index.html. Exits 2 with one line if mujoco is missing; install it first with `make install`.
demo:
	$(PY) scripts/demo_all.py --quick --out $(OUT)

lint:
	$(PY) -m ruff check .

docs-check:
	$(PY) scripts/check_docs.py
	$(PY) scripts/check_links.py

# Build the sdist and wheel, install the wheel in a temp venv, run it from another directory.
package-smoke:
	$(PY) scripts/smoke_package.py

serve:
	$(PY) -m arrival serve --out $(OUT)

clean:
	rm -rf $(OUT)

# Static demo site: no build step. Open http://localhost:8765
site:
	$(PY) -m http.server 8765 --directory site

# Resolves every asset path the site references, against a running `make site` (or by file path with --offline)
check-site:
	$(PY) scripts/check_site.py

# GPU targets (G5). Never run on a GPU yet; see docs/BUILD-LOG.md. train needs HF_TOKEN in the environment.
DATASET ?= $(OUT)/datasets/hero_kitchen_sim
CKPT ?= $(OUT)/finetune_smoke/checkpoint-100
train:
	DATASET_DIR=$(DATASET) OUTPUT_DIR=$(OUT)/finetune_smoke bash scripts/run_finetune.sh

# Needs GR00T_POLICY_URL pointing at a running `make serve-policy`.
eval-policy:
	$(PY) -m arrival eval --policy vla --checkpoint-id "$(CKPT)" --out $(OUT)

serve-policy:
	$(PY) -m arrival.policy_server.gr00t_shim --model-path $(CKPT) --port 8000 --modality-config-path configs/gr00t/arrival_panda_config.py
