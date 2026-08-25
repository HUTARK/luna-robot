#!/usr/bin/env bash
# build_ctranslate2_cuda.sh — CUDA-enabled CTranslate2 for Jetson Orin.
#
#   chmod +x build_ctranslate2_cuda.sh
#   ./build_ctranslate2_cuda.sh            # run OUTSIDE the venv is fine;
#                                          # it installs into whatever python
#                                          # is active, so activate the venv
#                                          # first if that is where you want it
#
# WHY: PyPI has no CUDA aarch64 wheels for ctranslate2. `pip install
# faster-whisper` gives a CPU-only build, and passing device="cuda" then falls
# back to CPU SILENTLY — the Orin ends up no faster than a Pi 5 and nothing in
# the logs says why. Confirmed on this machine:
#     "This CTranslate2 package was not compiled with CUDA support"
#
# Expect 30-90 minutes. It is CPU-bound and memory-hungry.
#
# ALTERNATIVE: NVIDIA's jetson-containers project ships prebuilt faster-whisper
# images for JetPack 6. If this build fights you, that is the pragmatic route:
#     https://github.com/dusty-nv/jetson-containers

set -euo pipefail

# Orin = Ampere, compute capability 8.7. Building only this arch keeps the
# compile far shorter than the default multi-arch fat binary.
CUDA_ARCH=87
SRC="$HOME/src/CTranslate2"

echo "=== 0. Environment ==="
nvcc --version | tail -2 | head -1 || { echo "nvcc not on PATH."; \
  echo "  export PATH=/usr/local/cuda/bin:\$PATH"; exit 1; }
python3 -c "import sys; print('python', sys.version.split()[0])"

# Match the version already installed, so faster-whisper's dependency pin is
# still satisfied after we replace the wheel.
WANT=$(python3 -c "import ctranslate2; print(ctranslate2.__version__)" 2>/dev/null || echo "")
echo "currently installed ctranslate2: ${WANT:-<none>}"

echo
echo "=== 1. Performance mode ==="
# Benchmark on a throttled clock and you will draw the wrong conclusions.
sudo nvpmodel -m 0 || true          # MAXN
sudo jetson_clocks || true

echo
echo "=== 2. Memory headroom ==="
free -h
echo "CUDA compilation is memory-hungry. On an 8GB Orin Nano, nvcc can OOM"
echo "with high parallelism. Using -j4. If it still dies, add swap:"
echo "  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile"
echo "  sudo mkswap /swapfile && sudo swapon /swapfile"

echo
echo "=== 3. Build dependencies ==="
sudo apt-get update
# MKL is x86-only, so OpenBLAS is the BLAS backend on aarch64.
sudo apt-get install -y build-essential cmake git libopenblas-dev \
                        pkg-config python3-dev python3-pip

echo
echo "=== 4. Source ==="
mkdir -p "$(dirname "$SRC")"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch --tags --quiet
else
  git clone --recursive https://github.com/OpenNMT/CTranslate2.git "$SRC"
fi
cd "$SRC"
if [ -n "$WANT" ] && git rev-parse "v$WANT" >/dev/null 2>&1; then
  echo "checking out v$WANT to match the installed version"
  git checkout --quiet "v$WANT"
else
  TAG=$(git tag -l 'v*' --sort=-v:refname | head -1)
  echo "checking out latest tag $TAG"
  git checkout --quiet "$TAG"
fi
git submodule update --init --recursive --quiet

echo
echo "=== 5. Configure ==="
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
  -DBUILD_CLI=OFF

echo
echo "=== 6. Compile (this is the long part) ==="
make -j4

echo
echo "=== 7. Install the shared library ==="
sudo make install
sudo ldconfig

echo
echo "=== 8. Python wheel ==="
cd ../python
pip install -r install_requirements.txt
python3 setup.py bdist_wheel
pip install --force-reinstall dist/*.whl

echo
echo "=== 9. Verify ==="
cd "$HOME"
python3 - <<'PY'
import ctranslate2
print("ctranslate2:", ctranslate2.__version__)
try:
    types = ctranslate2.get_supported_compute_types("cuda")
    print("cuda compute types:", types)
    print("\nSUCCESS — CUDA is available." if types else "\nSTILL CPU-ONLY.")
    if types:
        print('Use: WhisperModel("small.en", device="cuda", compute_type="float16")')
except Exception as e:
    print("FAILED:", e)
PY

echo
echo "If step 9 reports CUDA types, benchmark next:"
echo "  python3 luna_stt_bench.py"
