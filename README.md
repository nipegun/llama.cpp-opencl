# llama.cpp-opencl

`llama.cpp-opencl` is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with its own implementation of the OpenCL backend, crafted with NiPeGunian precision to run GGUF LLMs on legacy GPUs.

It includes every improvement made in the official llama.cpp repository up to the day this repository was published.

## Only three backends

 - **OpenCL**: with OpenCL kernels written specifically for this project.
 - **CPU**: with one library per x86 generation, the best one being loaded at startup.
 - **RPC**: to be able to chain with other legacy hardware as well.

All the other backends were removed to clean up the code.

## Compatible GGUF models

For now, it is compatible with the following GGUF models (Q8_0, Q6_K, Q5_K_M and Q4_K_M):

### Qwen3.5

With its multi-token prediction (MTP) layer, but without multimodal input (for now).

 - **Qwen3.5 0.8B**.
 - **Qwen3.5 2B**.
 - **Qwen3.5 4B**.

## Build and install

As root on Debian 13:

```bash
# 1. Install git and the build dependencies
apt-get -y update
apt-get -y install git build-essential cmake ocl-icd-opencl-dev libssl-dev nodejs npm

# 2. Clone the repository
git clone https://github.com/nipegun/llama.cpp-opencl.git

# 3. Enter the folder and build
cd llama.cpp-opencl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_OPENCL=ON -DGGML_RPC=ON -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_TESTS_INSTALL=OFF
cmake --build build -j "$(nproc)"

# 4. Install system-wide (programs and backends in /usr/local/bin, libraries in /usr/local/lib)
cmake --install build
ldconfig
```

The web UI of `llama-server` is built from source with npm, so the first build needs an Internet connection to download its dependencies. If the web UI cannot be built, the build stops with an error.

The tests and micro-benchmarks (`test-backend-ops`, `llama-bench-dmmv` and `llama-bench-gemm`) are not installed; run them from `build/bin/`.

Instead of installing it, you can also copy the whole `build/bin/` folder, which holds all the programs and their shared libraries, to another computer with the GPU.

## Compatible GPUs

Any GPU whose driver exposes OpenCL 1.1 or later. The tables below group GPU series by the highest OpenCL version they can reach on Debian 13, counting every driver that can be installed on it, whether it comes from Debian 13 itself, from another Debian suite (backports, sid) or from the GPU vendor. Within each table, series are ordered by release date. Run `clinfo -l` to see what your own GPU reports.

Tested on an NVIDIA GeForce GT 720M (Fermi, NVIDIA 390) and an Intel HD Graphics 4000 (Ivy Bridge, Beignet). The rest of the list is based on the documentation and source code of each driver.

No GPU series tops out at OpenCL 2.1 or 2.2 on Debian 13, so those versions have no section of their own.

### Up to OpenCL 1.1

#### AMD

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| Radeon HD 5000 | Desktop | 2009/09 | HD 5450, 5570, 5670, 5770, 5870, 5970 | Clover | No |
| Mobility Radeon HD 5000 | Laptop | 2010/01 | Mobility HD 5470, 5650, 5870 | Clover | No |
| FirePro V (Evergreen), FireStream 9350/9370 | Pro/Server | 2010/04 | FirePro V3800 to V9800, FireStream 9370 | Clover | No |
| Radeon HD 6400 to 6800 | Desktop | 2010/10 | HD 6450, 6570, 6670, 6850, 6870 | Clover | No |
| Radeon HD 6900 | Desktop | 2010/12 | HD 6950, 6970, 6990 | Clover ¹ | No |
| E-Series, C-Series and Z-Series APUs (Brazos, Hondo) | Integrated | 2011/01 | E-350 (HD 6310), C-50 (HD 6250), E2-1800 (HD 7340), Z-60 | Clover | No |
| Radeon HD 6000M | Laptop | 2011/01 | HD 6470M, 6630M, 6770M, 6970M, 6990M | Clover | No |
| FirePro V (Northern Islands) | Pro | 2011 | FirePro V4900, V5900, V7900 | Clover ¹ | No |
| A-Series APUs (Llano) | Integrated | 2011/06 | A8-3850 (HD 6550D), A6-3420M (HD 6520G) | Clover | No |
| Radeon HD 7000 and HD 7000M (TeraScale models) | Desktop/Laptop | 2012/01 | HD 7450, 7470, 7570, 7670, HD 7470M, 7570M, 7670M | Clover | No |
| A-Series APUs (Trinity) | Integrated | 2012/05 | A10-4600M (HD 7660G), A10-5800K (HD 7660D) | Clover ¹ | No |
| Radeon HD 8000, R5 200 and R5 300 (TeraScale models) | Desktop | 2013/01 | HD 8350, 8450, 8470, 8490, R5 220, 230, 235, R5 310 | Clover | No |
| A-Series APUs (Richland) | Integrated | 2013/03 | A10-6800K (HD 8670D), A10-5750M (HD 8650G) | Clover ¹ | No |

#### NVIDIA

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| GeForce 400 | Desktop | 2010/03 | GTX 480, 470, 465, 460, GTS 450, GT 440, 430, 420 | NVIDIA 390 | No |
| Tesla 20 | Server | 2010/05 | C2050, C2070, M2050, M2070, M2090 | NVIDIA 390 | No |
| GeForce 400M | Laptop | 2010/05 | GTX 480M, 485M, GT 4xxM (except 405M) | NVIDIA 390 | No |
| Quadro (Fermi), NVS | Pro | 2010/07 | Quadro 600, 2000, 4000, 5000, 6000, 7000, Quadro 1000M to 5010M, NVS 310, 315, 4200M, 5200M, 5400M | NVIDIA 390 | No |
| GeForce 500 | Desktop | 2010/11 | GTX 580, 570, 560 Ti, 560, 550 Ti, GT 545, 530, 520, GeForce 510 | NVIDIA 390 | No |
| GeForce 500M | Laptop | 2011/01 | GTX 580M, 570M, 560M, GT 555M to 520M | NVIDIA 390 | No |
| GeForce 600M (Fermi models) | Laptop | 2012/03 | 610M, GT 620M, 625M, 630M, 635M, GTX 670M, 675M | NVIDIA 390 | No |
| GeForce 600 (Fermi models) | Desktop | 2012/04 | GT 605, 610, 615, 620, 625, GT 630 (GF108), GT 640 OEM (GF116), GT 645 | NVIDIA 390 | No |
| GeForce 700M (Fermi models) | Laptop | 2013/04 | 705M, 710M, GT 720M (GF117), 710A, 720A | NVIDIA 390 | No |
| GeForce 800M (Fermi models) | Laptop | 2014/03 | 800M, 810M, 820M (GF117) | NVIDIA 390 | No |
| GeForce 700 (Fermi models) | Desktop | 2014 | GT 705 (OEM), GT 730 96-core DDR3 (GF108) | NVIDIA 390 | No |

### Up to OpenCL 1.2

#### Intel

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| HD Graphics (Ivy Bridge, Gen7) | Integrated | 2012/04 | HD 2500, HD 4000, HD P4000 | Beignet ⁶ | 1.2 (hasvk) |
| HD, Iris and Iris Pro Graphics (Haswell, Gen7.5) | Integrated | 2013/06 | HD 4200, 4400, 4600, 5000, Iris 5100, Iris Pro 5200, P4600, P4700 | Beignet ⁶ | 1.2 (hasvk) |
| HD Graphics (Bay Trail) | Integrated | 2013/09 | Atom Z3000, Celeron and Pentium J1000, N2000, J2000, N3500 | Beignet ⁶ | 1.2 (hasvk) |
| HD Graphics 400 and 405 (Cherry Trail, Braswell) | Integrated | 2015/03 | Atom x5 and x7 Z8000, Celeron and Pentium N3000 | Beignet ⁶ | 1.2 (hasvk) |

### Up to OpenCL 2.0

#### AMD

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| Instinct MI100 | Server | 2020/11 | MI100 | ROCm (Debian) | No |
| Instinct MI200 | Server | 2021/11 | MI210, MI250, MI250X | ROCm (Debian) | No |
| Instinct MI300 | Server | 2023/12 | MI300X, MI300A, MI325X | ROCm (Debian) or ROCm (AMD) | No |
| Instinct MI350 | Server | 2025/06 | MI350X, MI355X | ROCm (AMD) and AMD's `amdgpu-dkms` kernel driver | No |

### Up to OpenCL 3.0

#### AMD

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| Radeon HD 7700 to 7900 | Desktop | 2012/01 | HD 7750, 7770, 7850, 7870, 7950, 7970, 7990 | Rusticl ² | 1.3 (RADV) ² |
| Radeon HD 7700M to 7900M | Laptop | 2012/04 | HD 7730M, 7850M, 7970M | Rusticl ² | 1.3 (RADV) ² |
| FirePro W and S (GCN 1), Radeon Sky | Pro/Server | 2012/08 | W600, W2100, W4100, W5000, W7000, W8000, W9000, S9000, S10000 | Rusticl ² | 1.3 (RADV) ² |
| Radeon HD 8000 and HD 8000M (GCN models) | Desktop/Laptop | 2013/01 | HD 8570, 8760, 8770, 8870, 8970, HD 8750M, 8970M | Rusticl ² ³ | 1.3 (RADV) ² ³ |
| Radeon HD 7790 | Desktop | 2013/03 | HD 7790 | Rusticl ³ | 1.3 (RADV) ³ |
| A-Series and E-Series APUs (Kabini, Temash) | Integrated | 2013/05 | A4-5000 (HD 8330), A6-5200 (HD 8400) | Rusticl ³ | 1.3 (RADV) ³ |
| Radeon R7 200 and R9 200 | Desktop | 2013/10 | R7 240, 250, 260X, 265, R9 270X, 280X, 285, 290, 290X, 295X2 | Rusticl ² ³ | 1.3 (RADV) ² ³, 1.4 on R9 285 |
| A-Series APUs (Kaveri, Godavari) | Integrated | 2014/01 | A10-7850K, A10-7870K, A10-7890K | Rusticl ³ | 1.3 (RADV) ³ |
| Radeon R5, R7 and R9 M200 | Laptop | 2014/01 | R5 M230, R7 M265, R9 M270X, R9 M280X, R9 M295X | Rusticl ² ³ | 1.3 (RADV) ² ³, 1.4 on R9 M295X |
| A-Series and E-Series APUs (Beema, Mullins) | Integrated | 2014/04 | A6-6310, A10 Micro-6700T | Rusticl ³ | 1.3 (RADV) ³ |
| FirePro (GCN 2) | Pro/Server | 2014/04 | W5100, W8100, W9100, S9150, S9170 | Rusticl ³ | 1.3 (RADV) ³ |
| FirePro (GCN 3), Radeon Pro Duo | Pro/Server | 2014/08 | W7100, S7150, S9300 x2, Radeon Pro Duo | Rusticl | 1.4 (RADV) |
| Radeon R5, R7 and R9 M300 | Laptop | 2015/05 | R5 M330, R7 M360, R9 M390X, R9 M395X | Rusticl ² | 1.3 (RADV) ², 1.4 on R9 M390X and M395X |
| Radeon R7 300, R9 300, R9 Fury and R9 Nano | Desktop | 2015/06 | R7 360, R7 370, R9 380, 380X, 390, 390X, Fury, Fury X, Nano | Rusticl ² ³ | 1.3 (RADV) ² ³, 1.4 on R9 380, 380X, Fury and Nano |
| A-Series APUs (Carrizo, Bristol Ridge, Stoney Ridge) | Integrated | 2015/06 | A10-8700P, A12-9800, A9-9420 | Rusticl | 1.4 (RADV) |
| Radeon RX 400 and R5 to R9 M400 | Desktop/Laptop | 2016/06 | RX 460, 470, 480, RX 470M | Rusticl | 1.4 (RADV) |
| Radeon Pro WX and Radeon Pro 400/500 | Pro | 2016/10 | WX 2100, 3100, 4100, 5100, 7100 | Rusticl | 1.4 (RADV) |
| Radeon RX 500 and 500X, Radeon 520 to 630 | Desktop/Laptop | 2017/04 | RX 550, 560, 570, 580, 590 | Rusticl | 1.4 (RADV) |
| Radeon Instinct MI6, MI8 and MI25 | Server | 2017/06 | MI6, MI8, MI25 | Rusticl | 1.4 (RADV) |
| Radeon RX Vega, Vega Frontier Edition, Radeon Pro WX 8200/9100 | Desktop/Pro/Laptop | 2017/06 | RX Vega 56, Vega 64, WX 9100, Pro SSG | Rusticl | 1.4 (RADV) |
| Ryzen with Radeon Vega Graphics | Integrated | 2017/10 | Ryzen 5 2500U, Ryzen 5 3400G, Ryzen 7 5700G | Rusticl | 1.4 (RADV) |
| Radeon VII, Radeon Pro VII, Instinct MI50 and MI60 | Desktop/Pro/Server | 2018/11 | Radeon VII, Pro VII, MI50, MI60 | Rusticl | 1.4 (RADV) |
| Radeon RX 5000 and 5000M, Radeon Pro W5000 | Desktop/Laptop/Pro | 2019/07 | RX 5500 XT, 5600 XT, 5700 XT, Pro W5700 | Rusticl | 1.4 (RADV) |
| Radeon RX 6000 and 6000M, Radeon Pro W6000 | Desktop/Laptop/Pro | 2020/11 | RX 6600, 6700 XT, 6800, 6900 XT, Pro W6800 | Rusticl | 1.4 (RADV) |
| Ryzen with RDNA 2 graphics | Integrated | 2022/01 | Radeon 680M, Steam Deck, Ryzen 7000 and 9000 desktop iGPU | Rusticl | 1.4 (RADV) |
| Radeon RX 7000 and 7000M, Radeon Pro W7000 | Desktop/Laptop/Pro | 2022/12 | RX 7600, 7800 XT, 7900 XTX, Pro W7900 | Rusticl | 1.4 (RADV) |
| Ryzen with RDNA 3 graphics | Integrated | 2023/05 | Radeon 780M (Ryzen 7040, 8040, 8000G, Z1) | Rusticl | 1.4 (RADV) |
| Ryzen AI with RDNA 3.5 graphics | Integrated | 2024/07 | Radeon 890M, 860M, 8060S | Rusticl | 1.4 (RADV) |
| Radeon RX 9000, Radeon AI PRO R9700 | Desktop/Pro | 2025/03 | RX 9060 XT, 9070, 9070 XT, AI PRO R9700 | Rusticl | 1.4 (RADV) |

#### Intel

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| HD and Iris Graphics (Broadwell, Gen8) | Integrated | 2014/09 | HD 5300, 5500, 5600, 6000, Iris 6100, Iris Pro 6200, P5700, P6300 | NEO legacy1 or Rusticl ⁴ | 1.3 (hasvk) |
| HD and Iris Graphics (Skylake, Gen9) | Integrated | 2015/08 | HD 510, 515, 520, 530, Iris 540, 550, Iris Pro 580, P530, P580 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| HD Graphics (Apollo Lake) | Integrated | 2016/08 | HD 500, 505 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| HD, UHD and Iris Plus Graphics (Kaby Lake, Amber Lake, Whiskey Lake) | Integrated | 2016/08 | HD 610, 615, 620, 630, UHD 615, 617, 620, Iris Plus 640, 650, P630 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| UHD and Iris Plus Graphics (Coffee Lake, Comet Lake) | Integrated | 2017/10 | UHD 610, 630, P630, Iris Plus 645, 655 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| UHD Graphics (Gemini Lake) | Integrated | 2017/12 | UHD 600, 605 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| Iris Plus and UHD Graphics (Ice Lake, Gen11) | Integrated | 2019/08 | Iris Plus G4, G7, UHD G1 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| UHD Graphics (Elkhart Lake) | Integrated | 2020/09 | Atom x6000E, Pentium and Celeron J6000/N6000 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| Iris Xe and UHD Graphics (Tiger Lake, Gen12) | Integrated | 2020/09 | Iris Xe (80 and 96 EU), UHD (48 EU) | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Iris Xe MAX, Iris Xe (DG1), Server GPU SG1 | Laptop/Desktop/Server | 2020/10 | Iris Xe MAX, Iris Xe (DG1), SG-18M | NEO or Rusticl ⁴ ⁵ | 1.4 (ANV) ⁵ |
| UHD Graphics (Jasper Lake) | Integrated | 2021/01 | Celeron N4500, N5100, Pentium Silver N6000 | NEO legacy1 or Rusticl ⁴ | 1.4 (ANV) |
| UHD Graphics (Rocket Lake) | Integrated | 2021/03 | UHD 730, 750 | NEO or Rusticl ⁴ | 1.4 (ANV) |
| UHD and Iris Xe Graphics (Alder Lake) | Integrated | 2021/11 | UHD 710, 730, 770, Iris Xe | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Arc A-Series (Alchemist) | Desktop/Laptop | 2022/03 | A310, A380, A580, A750, A770, A350M to A770M | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Arc Pro A-Series | Pro | 2022/08 | Arc Pro A30M, A40, A50, A60, A60M | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Data Center GPU Flex | Server | 2022/08 | Flex 140, Flex 170 | NEO or Rusticl ⁴ ⁵ | 1.4 (ANV) ⁵ |
| UHD and Iris Xe Graphics (Raptor Lake) | Integrated | 2022/10 | UHD 710, 730, 770, Iris Xe | NEO or Rusticl ⁴ | 1.4 (ANV) |
| UHD Graphics (Alder Lake-N, Twin Lake) | Integrated | 2023/01 | N100, N200, N97, Core i3-N305, N150, N250 | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Data Center GPU Max (Ponte Vecchio) | Server | 2023/01 | Max 1100, 1350, 1450, 1550 | NEO ⁸ | No |
| Arc Graphics (Meteor Lake) | Integrated | 2023/12 | Arc Graphics, Intel Graphics (Core Ultra 100) | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Arc 130V and 140V (Lunar Lake) | Integrated | 2024/09 | Arc 130V, Arc 140V | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Intel Graphics, Arc 130T and 140T (Arrow Lake) | Integrated | 2024/10 | Intel Graphics (Core Ultra 200S), Arc 130T, Arc 140T | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Arc B-Series (Battlemage) | Desktop | 2024/12 | Arc B570, B580 | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Arc Pro B60 | Pro | 2025/06 | Arc Pro B60 | NEO or Rusticl ⁴ ⁷ | 1.4 (ANV) ⁷ |
| Arc Pro B50 | Pro | 2025/09 | Arc Pro B50 | NEO or Rusticl ⁴ | 1.4 (ANV) |
| Arc B-Series and Intel Graphics (Panther Lake) | Integrated | 2026/01 | Arc B370, Arc B390, Intel Graphics | NEO or Rusticl ⁴ ⁷ | 1.4 (ANV) ⁷ |
| Arc Pro B65 and B70 | Pro | 2026/03 | Arc Pro B65, Arc Pro B70 | NEO or Rusticl ⁴ ⁷ | 1.4 (ANV) ⁷ |
| Intel Graphics (Wildcat Lake) | Integrated | 2026 | Intel Graphics | NEO or Rusticl ⁴ ⁷ | 1.4 (ANV) ⁷ |

#### NVIDIA

| Series | Type | Released | Examples | OpenCL driver | Vulkan |
|---|---|---|---|---|---|
| GeForce 600 (Kepler models) | Desktop | 2012/03 | GTX 690, 680, 670, 660 Ti, 660, 650 Ti, 650, GT 640, GT 635, GT 630 (GK107, GK208) | NVIDIA 470 | 1.2 (NVIDIA) |
| GeForce 600M (Kepler models) | Laptop | 2012/03 | GT 640M, 645M, 650M, GTX 660M, 670MX, 675MX, 680M, 680MX | NVIDIA 470 | 1.2 (NVIDIA) |
| Tesla K, GRID K | Server | 2012/05 | K10, K20, K20X, K40, K80, K520, GRID K1, K2 | NVIDIA 470 | Unverified |
| Quadro K (Kepler models), NVS 510 | Pro | 2012/08 | K600, K2000, K4000, K5000, K6000, K420, K4200, K5200, Quadro 410, K500M to K5100M, NVS 510 | NVIDIA 470 | 1.2 (NVIDIA) |
| GeForce GTX Titan, GeForce 700 (Kepler models) | Desktop | 2013/02 | GTX Titan, Titan Black, Titan Z, GTX 780 Ti, 780, 770, 760, GT 740, GT 730 (GK208), GT 720, GT 710 | NVIDIA 470 | 1.2 (NVIDIA) |
| GeForce 700M (Kepler models) | Laptop | 2013/04 | GT 730M to 755M, GTX 760M to 780M | NVIDIA 470 | 1.2 (NVIDIA) |
| GeForce 700 (Maxwell models) | Desktop | 2014/02 | GTX 750 Ti, 750, 745 | NVIDIA 550 | 1.3 (NVIDIA) |
| GeForce 800M | Laptop | 2014/03 | Kepler: GTX 880M, 870M, 825M. Maxwell: 830M, 840M, 845M, GTX 850M, 860M (GM107) | NVIDIA 470 (Kepler), NVIDIA 550 (Maxwell) | 1.2 (Kepler), 1.3 (Maxwell) |
| Quadro K (Maxwell models) | Pro | 2014/07 | K620, K1200, K2200, K620M, K2200M | NVIDIA 550 | 1.3 (NVIDIA) |
| GeForce 900, GTX Titan X | Desktop | 2014/09 | GTX 980 Ti, 980, 970, 960, 950, Titan X | NVIDIA 550 | 1.3 (NVIDIA) |
| GeForce 900M | Laptop | 2014/10 | Maxwell: GTX 980M, 970M, 965M, 960M, 950M, 920MX, 930M, 940M, 940MX, 945M. Kepler: 910M, 920M, 920A | NVIDIA 550 (Maxwell), NVIDIA 470 (Kepler) | 1.3 (Maxwell), 1.2 (Kepler) |
| Quadro M, NVS 810 | Pro | 2015/03 | M6000, M5000, M4000, M2000, M500M to M5500 | NVIDIA 550 | 1.3 (NVIDIA) |
| Tesla M | Server | 2015/08 | M60, M40, M10, M6, M4 | NVIDIA 550 | Unverified |
| GeForce 10 | Desktop | 2016/05 | GTX 1080 Ti, 1080, 1070 Ti, 1070, 1060, 1050 Ti, 1050, GT 1030, GT 1010, Titan X (Pascal), Titan Xp | NVIDIA 550 | 1.3 (NVIDIA) |
| Tesla P | Server | 2016/06 | P100, P40, P4, P6 | NVIDIA 550 | Unverified |
| GeForce 10 laptop | Laptop | 2016/08 | GTX 1080, 1070, 1060, 1050 Ti, 1050 (laptop and Max-Q) | NVIDIA 550 | 1.3 (NVIDIA) |
| Quadro P | Pro | 2016/10 | P6000, P5000, P4000, P2200, P2000, P1000, P620, P600, P400, GP100, P500 to P5200 (laptop) | NVIDIA 550 | 1.3 (NVIDIA) |
| GeForce MX | Laptop | 2017/05 | Pascal: MX150, MX230, MX250, MX330, MX350. Maxwell: MX110, MX130 | NVIDIA 550 | 1.3 (NVIDIA) |
| Tesla V100 | Server | 2017/06 | V100, V100S | NVIDIA 550 | Unverified |
| Titan V, Quadro GV100 | Desktop/Pro | 2017/12 | Titan V, Quadro GV100 | NVIDIA 550 | 1.3 (NVIDIA) |
| Quadro RTX | Pro | 2018/08 | Quadro RTX 8000, 6000, 5000, 4000, RTX 3000 and 5000 (laptop) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce RTX 20, Titan RTX | Desktop | 2018/09 | RTX 2080 Ti, 2080, 2070, 2060 (and Super), Titan RTX | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| Tesla T4 | Server | 2018/09 | T4 | NVIDIA 550 | Unverified |
| GeForce RTX 20 laptop | Laptop | 2019/01 | RTX 2080, 2070, 2060 (laptop and Max-Q) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce GTX 16 | Desktop | 2019/02 | GTX 1660 Ti, 1660 Super, 1660, 1650 Super, 1650, 1630 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce GTX 16 laptop | Laptop | 2019/04 | GTX 1660 Ti, 1650 Ti, 1650 (laptop) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| Quadro T, NVIDIA T | Pro | 2019/05 | T400, T600, T1000, T1200, T2000, T500, T550 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| A100, A800, A30 | Server | 2020/05 | A100, A800, A30 | NVIDIA 550 | Unverified |
| GeForce MX450 and MX550 | Laptop | 2020/08 | MX450, MX550 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce RTX 30 | Desktop | 2020/09 | RTX 3090 Ti, 3090, 3080 Ti, 3080, 3070 Ti, 3070, 3060 Ti, 3060, 3050 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| RTX A-Series, A40, A10, A16, A2 | Pro/Server | 2020/10 | RTX A6000, A5500, A5000, A4500, A4000, A2000, A1000, A400, A40, A10, A16, A2 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce RTX 30 laptop | Laptop | 2021/01 | RTX 3080 Ti to 3050 (laptop) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| RTX A-Series laptop | Laptop | 2021/04 | RTX A5000, A4000, A3000, A2000, A1000, A500 (laptop) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce RTX 2050 and MX570 | Laptop | 2021/12 | RTX 2050, MX570 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| H100, H800, H200, H20 | Server | 2022/09 | H100, H800, H200, H20 | NVIDIA 550 | Unverified |
| GeForce RTX 40 | Desktop | 2022/10 | RTX 4090, 4080 (and Super), 4070 Ti (and Super), 4070 (and Super), 4060 Ti, 4060 | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| L40, L40S, L4, L20 | Server | 2022/10 | L40, L40S, L4, L20 | NVIDIA 550 | Unverified |
| RTX Ada | Pro | 2022/12 | RTX 6000, 5000, 4500, 4000, 4000 SFF, 2000 Ada | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| GeForce RTX 40 laptop | Laptop | 2023/02 | RTX 4090 to 4050 (laptop), RTX 3050 A (laptop) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| RTX Ada laptop | Laptop | 2023/03 | RTX 5000, 3500, 3000, 2000 Ada (laptop) | NVIDIA 550 | 1.3 (NVIDIA), 1.4 (NVK) |
| B100, B200, B300 | Server | 2024/12 | B100, B200, B300 | NVIDIA 570+ | Unverified |
| GeForce RTX 50 | Desktop | 2025/01 | RTX 5090, 5080, 5070 Ti, 5070, 5060 Ti, 5060, 5050 | NVIDIA 570+ | 1.4 (NVIDIA) |
| GeForce RTX 50 laptop | Laptop | 2025/03 | RTX 5090 to 5050 (laptop) | NVIDIA 570+ | 1.4 (NVIDIA) |
| RTX PRO Blackwell | Pro | 2025/03 | RTX PRO 6000, 5000, 4500, 4000, 2000 | NVIDIA 570+ | 1.4 (NVIDIA) |

### OpenCL drivers

| Key | Driver | Where to get it | Notes |
|---|---|---|---|
| Clover | Mesa clover | `mesa-opencl-icd` (Debian 13 main) | Do not use the Mesa from trixie-backports: it no longer includes clover. |
| Rusticl | Mesa rusticl | `mesa-opencl-icd` (Debian 13 main) | Disabled by default: set `RUSTICL_ENABLE=radeonsi` (AMD) or `RUSTICL_ENABLE=iris` (Intel). |
| ROCm (Debian) | AMD ROCm 5.7 | `rocm-opencl-icd` (Debian 13 main) | |
| ROCm (AMD) | AMD ROCm | AMD's repository (repo.radeon.com) | AMD supports Debian 13 only on Instinct MI300X, MI325X, MI350X and MI355X. |
| NEO | Intel compute-runtime | `.deb` packages from github.com/intel/compute-runtime/releases | Not in the Debian 13 repositories. |
| NEO legacy1 | Intel compute-runtime, legacy1 branch (24.35.30872.36) | Same place, legacy1 release | For Gen8 to Gen11. Can be installed together with NEO. |
| Beignet | Intel Beignet 1.3.2 | `beignet-opencl-icd` (1.3.2-8) from Debian 11 (bullseye) | Its dependencies are all in Debian 13. |
| NVIDIA 390 | `nvidia-legacy-390xx-driver` (390.157) | Debian sid (non-free) | Last branch for Fermi. Installs and builds on Debian 13's kernel 6.12. On CPUs with Indirect Branch Tracking (IBT), add `ibt=off` to the kernel command line. |
| NVIDIA 470 | `nvidia-tesla-470-driver` (470.256.02) | Debian bookworm-backports (pinned) or sid (non-free) | Last branch for Kepler. Reports OpenCL 3.0. |
| NVIDIA 550 | `nvidia-driver` (550) or `nvidia-tesla-535-driver` | Debian 13 non-free | NVIDIA's 580 installer is also an option (Vulkan 1.4). |
| NVIDIA 570+ | NVIDIA 570 or newer, open kernel modules | NVIDIA's `.run` installer or its CUDA `debian13` repository | Required for Blackwell. |

### Not supported

These GPUs either have no OpenCL driver that can be installed on Debian 13, or only reach OpenCL 1.0.

| Vendor | Series | Reason | Vulkan |
|---|---|---|---|
| AMD | Radeon HD 2000, 3000 and 4000, Mobility Radeon HD 4000, Radeon HD 3200 to 4290 (integrated), FirePro V3700 to V8750, FireStream 9170, 9250 and 9270 | Their only OpenCL driver (ATI Stream with fglrx) is dead, and Mesa has no compute support for them. | No |
| Intel | GMA, GMA X3000 to X4500, HD Graphics (Ironlake), HD 2000 and 3000 (Sandy Bridge) | They never had OpenCL on Linux. | No |
| Intel | UHD Graphics (Lakefield) | No Linux kernel driver supports it, not even in newer kernels. | No |
| NVIDIA | GeForce 8, 9, 100, 200 and 300, ION, Quadro FX, Tesla C870 and C1060, GeForce 405, GT 415 (OEM) and 405M | Only OpenCL 1.0. Their driver (`nvidia-legacy-340xx-driver` from Debian sid) installs on Debian 13, but these GPUs lack the 32 KB of local memory that OpenCL 1.1 requires. | No |

### Notes

1. TeraScale 3 GPUs (Radeon HD 6900, FirePro V5900 and V7900, Trinity and Richland APUs) are affected by an open LLVM bug that can make large OpenCL kernels fail to compile.
2. GCN 1 GPUs use the old `radeon` kernel driver by default. Add `radeon.si_support=0 amdgpu.si_support=1` to the kernel command line (required for Vulkan, recommended for OpenCL).
3. GCN 2 GPUs use the old `radeon` kernel driver by default. Add `radeon.cik_support=0 amdgpu.cik_support=1` to the kernel command line (required for Vulkan, recommended for OpenCL).
4. Prefer NEO on Intel: with rusticl, a single buffer cannot exceed 1 GiB and there is no half-precision (fp16) support.
5. Kernel 6.12 only enables these GPUs with `i915.force_probe=<PCI ID>` on the kernel command line.
6. Beignet installs on Debian 13 from the Debian 11 package. Tested on an Intel HD Graphics 4000.
7. Kernel 6.12 does not support these GPUs: install the kernel, `firmware-intel-graphics` and Mesa from trixie-backports.
8. Needs Intel's out-of-tree kernel driver (`intel-gpu-i915-backports`, as DKMS) and its firmware from github.com/intel-gpu/intel-gpu-firmware. Intel has built it for kernel 6.12 but has not tested it there.

