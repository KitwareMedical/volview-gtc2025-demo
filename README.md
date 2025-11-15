# VolView + NVIDIA Integration for GTC 2025

This repository showcases an enhanced version of [**Kitware's
VolView**](https://github.com/Kitware/VolView), extended with cutting-edge
healthcare foundation models from **NVIDIA Clara**.

This special build integrates three powerful AI capabilities into the VolView
interface, each running on a scalable, independent backend server. For general
VolView features, please see the [official
repository](https://github.com/Kitware/VolView).

## NVIDIA Clara Features

This version of VolView adds three new tabs: **Curate**, **Reason**, and
**Generate**.

### 🧠 Segment (Curation)

The Segment tab uses the **NVIDIA Clara NV-Curate-CTMR-v2** model to perform
automatic 3D segmentation of anatomical structures.

* **How to Use:**

  1. Load a compatible 3D CT or MR dataset.

  2. Navigate to the **Segment** tab.

  3. Click **Run Segmentation**.

* **Output:** A new segmentation layer is automatically added to the scene.

![A screenshot of the Curate tab showing a segmented
CT](docs/assets/curate_tab_example.jpeg)

### 💬 Reason (Multimodal Chat)

The Reason tab integrates a multimodal chatbot powered by the **NVIDIA Clara
NV-Reason-CXR-3B** model, allowing you to have a text-based conversation about
the loaded medical image.

* **How to Use:**

  1. Load a medical image.

  2. Navigate to the **Reason** tab and select **Clara NV-Reason-CXR-3B**.

  3. Type your question (e.g., "Are there any visible fractures?") into the
     chat box.

* **Output:** The model's text response appears directly in the chat window.

![A screenshot of the Reason tab with an active chat
session](docs/assets/reason_tab_example.jpeg)

### 🎲 Generate (Synthetic Data)

The Generate tab uses the **NVIDIA Clara NV-Generate-CTMR-v2** model to create
synthetic 3D CT scans based on your specifications.

* **How to Use:**

  1. Navigate to the **Generate** tab.

  2. Configure the desired parameters (body region, anatomy, resolution, etc.).

  3. Click **Generate CT Scan**.

* **Output:** A new, realistic 3D volume is generated and loaded into VolView.

![A screenshot of a synthetically generated CT scan in
VolView](docs/assets/generate_tab_example_v2.jpeg)


## Software Requirements

Before getting started, ensure your system meets the following requirements:

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+ recommended), macOS, or Windows with WSL2
- **GPU**: NVIDIA GPU with at least 24GB VRAM (RTX 3090, RTX 4090, or equivalent)
- **CUDA**: CUDA 11.8+ or CUDA 12.x
- **System RAM**: At least 32GB recommended
- **Storage**: At least 50GB free space for models and data

### Software Dependencies

#### Frontend Requirements
- **Node.js**: Version 18.0+ 
- **npm**: Version 8.0+ (comes with Node.js)

#### Backend Requirements  
- **Python**: Version 3.11-3.13
- **Poetry**: Latest version for Python dependency management
- **PyTorch**: 2.0+ with CUDA support
- **MONAI**: For medical AI model execution

### Installation Prerequisites

1. **Install Node.js and npm**:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # macOS with Homebrew
   brew install node
   
   # Windows: Download from https://nodejs.org
   ```

2. **Install Python 3.11+**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.11 python3.11-venv
   
   # macOS with Homebrew  
   brew install python@3.11
   ```

3. **Install Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

4. **Verify CUDA Installation**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

## 📦 Model Setup

Before running the application, you need to download the required NVIDIA medical AI models. This section provides step-by-step instructions for downloading each model.

### Prerequisites

1. **Install Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Install Git** (for cloning GitHub repositories):
   ```bash
   # Ubuntu/Debian
   sudo apt install git

   # macOS
   brew install git
   ```

### Model Directory Structure

Models are stored directly in the `models/` directory with a flat structure:
```
models/
├── NV-Reason-CXR-3B/              # Chest X-ray reasoning model
├── NV-Segment-CT/                 # CT segmentation (132 classes)
├── NV-Segment-CTMR/               # CT & MR segmentation (345+ classes)
├── NV-Generate-CT/                # Synthetic CT generation
│   ├── configs/                   # Configuration files from GitHub
│   ├── scripts/                   # Inference scripts from GitHub
│   └── models/                    # CT-specific model weights
└── NV-Generate-MR/                # Synthetic MR generation
    ├── configs/                   # Configuration files from GitHub
    ├── scripts/                   # Inference scripts from GitHub
    └── models/                    # MR-specific model weights

reference/                          # Reference repositories (for setup only)
├── NV-Segment-CTMR-github/        # NVIDIA-Medtech segmentation repo
└── NV-Generate-CTMR-github/       # NVIDIA-Medtech generation repo
```

### Downloading Models

First, create the required directories:

```bash
mkdir -p models reference
```

#### 1. NV-Reason-CXR-3B (Chest X-ray Analysis)

Download the multimodal reasoning model for chest X-rays:

```bash
# From project root
huggingface-cli download nvidia/NV-Reason-CXR-3B \
  --local-dir models/NV-Reason-CXR-3B \
  --local-dir-use-symlinks False
```

**Model Info:**
- Size: ~6GB
- Purpose: Interactive chat-based analysis of chest X-rays
- Architecture: Vision-Language Model (3B parameters)

#### 2. NV-Segment Models (3D Segmentation)

Both segmentation models require MONAI bundle configs from GitHub + model weights from HuggingFace.

**Step 1: Clone the segmentation models repository (one-time setup)**

```bash
git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git reference/NV-Segment-CTMR-github
```

**Step 2a: Setup NV-Segment-CT (132 classes, CT only)**

```bash
# Copy bundle structure (configs, scripts, docs)
cp -r reference/NV-Segment-CTMR-github/NV-Segment-CT models/NV-Segment-CT

# Ensure models directory exists
mkdir -p models/NV-Segment-CT/models

# Download model weights directly (CT config expects model_mrct.pt)
wget "https://huggingface.co/nvidia/NV-Segment-CT/resolve/main/vista3d_pretrained_model/model.pt" \
  -O models/NV-Segment-CT/models/model_mrct.pt
```

**Step 2b: Setup NV-Segment-CTMR (345+ classes, CT & MR)**

```bash
# Copy bundle structure
cp -r reference/NV-Segment-CTMR-github/NV-Segment-CTMR models/NV-Segment-CTMR

# Ensure models directory exists
mkdir -p models/NV-Segment-CTMR/models

# Download model weights directly (CTMR config expects model.pt)
wget "https://huggingface.co/nvidia/NV-Segment-CTMR/resolve/main/vista3d_pretrained_model/model.pt" \
  -O models/NV-Segment-CTMR/models/model.pt
```

**Model Info:**
- **NV-Segment-CT**: 132 classes, CT scans only
- **NV-Segment-CTMR**: 345+ classes, supports CT_BODY, MRI_BODY, MRI_BRAIN modalities
- Size: ~2GB each
- Architecture: VISTA-3D transformer-based (218M parameters)

#### 3. NV-Generate Models (Synthetic CT & MR Generation)

NV-Generate requires **separate directories** for CT and MR because they are trained on different base modalities and require different datasets.

Each model requires manual assembly from two sources:
- **GitHub**: Configuration files and inference scripts (NVIDIA-Medtech/NV-Generate-CTMR)
- **HuggingFace**: Model weights and modality-specific datasets

**Step 1: Clone the configuration repository (one-time setup)**

```bash
git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git \
  reference/NV-Generate-CTMR-github
```

**Step 2a: Setup NV-Generate-CT**

```bash
# Create directory structure
mkdir -p models/NV-Generate-CT

# Copy configuration files and scripts from GitHub
cp -r reference/NV-Generate-CTMR-github/configs models/NV-Generate-CT/
cp -r reference/NV-Generate-CTMR-github/scripts models/NV-Generate-CT/

# Download CT model weights and CT-specific datasets from HuggingFace
huggingface-cli download nvidia/NV-Generate-CT \
  --local-dir models/NV-Generate-CT \
  --local-dir-use-symlinks False
```

**Step 2b: Setup NV-Generate-MR**

```bash
# Create directory structure
mkdir -p models/NV-Generate-MR

# Copy configuration files and scripts from GitHub
cp -r reference/NV-Generate-CTMR-github/configs models/NV-Generate-MR/
cp -r reference/NV-Generate-CTMR-github/scripts models/NV-Generate-MR/

# Download MR model weights and MR-specific datasets from HuggingFace
huggingface-cli download nvidia/NV-Generate-MR \
  --local-dir models/NV-Generate-MR \
  --local-dir-use-symlinks False
```

**Model Info:**
- **Architecture**: Rectified Flow (RFLOW) diffusion models - 30 steps, ~2-3 minutes per generation
- **NV-Generate-CT**:
  - Required models: autoencoder_v1.pt, diff_unet_3d_rflow-ct.pt
  - Required configs: config_maisi_diff_model_rflow-ct.json, environment_maisi_diff_model_rflow-ct.json
  - Size: ~4GB (models only, datasets not required for diff_model_infer workflow)
- **NV-Generate-MR**:
  - Required models: autoencoder_v2.pt, diff_unet_3d_rflow-mr.pt
  - Required configs: config_maisi_diff_model_rflow-mr.json, environment_maisi_diff_model_rflow-mr.json
  - Size: ~4GB (models only, datasets not required for diff_model_infer workflow)
- **Resolution**: Up to 512×512×768 voxels
- **Spacing**: 0.5mm to 5.0mm
- **Usage**: Uses simplified `scripts/diff_model_infer.py` workflow for image generation

**Important**: CT and MR models are kept separate because they are trained on different base modalities.

### Verify Model Installation

After downloading, verify that all models are correctly installed:

```bash
cd server
python3 scripts/verify_models.py
```

Expected output:
```
======================================================================
NVIDIA Medical AI Models - Installation Verification
======================================================================

✅ NV-Reason-CXR-3B
✅ NV-Segment-CT (132 classes)
✅ NV-Segment-CTMR (345+ classes)
✅ NV-Generate-CT
✅ NV-Generate-MR

Models verified: 5/5
✅ All models are properly installed and configured!
```

**Note:** The verification script checks for:
- **NV-Segment models**: `configs/inference.json` and `models/model.pt` (or `model_mrct.pt` for CT)
- **NV-Generate-CT**: Separate directory with:
  - `scripts/diff_model_infer.py` (from GitHub)
  - `configs/environment_maisi_diff_model_rflow-ct.json`
  - `configs/config_maisi_diff_model_rflow-ct.json`
  - CT-specific models: autoencoder_v1.pt, diff_unet_3d_rflow-ct.pt
- **NV-Generate-MR**: Separate directory with:
  - `scripts/diff_model_infer.py` (from GitHub)
  - `configs/environment_maisi_diff_model_rflow-mr.json`
  - `configs/config_maisi_diff_model_rflow-mr.json`
  - MR-specific models: autoencoder_v2.pt, diff_unet_3d_rflow-mr.pt

### Custom Model Paths (Optional)

You can override default model paths using environment variables. Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` to specify custom model locations:

```bash
# Custom model paths (optional)
NV_REASON_PATH=/custom/path/to/NV-Reason-CXR-3B
NV_SEGMENT_CT_PATH=/custom/path/to/NV-Segment-CT
NV_SEGMENT_CTMR_PATH=/custom/path/to/NV-Segment-CTMR
NV_GENERATE_CT_PATH=/custom/path/to/NV-Generate-CT
NV_GENERATE_MR_PATH=/custom/path/to/NV-Generate-MR
```

### Troubleshooting

**Issue:** `FileNotFoundError: Bundle not found`
- **Solution:** Run the verification script (`python config/model_paths.py`) to identify missing models, then re-download them.

**Issue:** HuggingFace CLI download fails
- **Solution:** Check your internet connection and ensure you have enough disk space (~15GB total).

**Issue:** Model path not recognized
- **Solution:** Ensure you're running commands from the project root directory and the `models/` directory exists.

---

## 🚀 Getting Started

Follow these steps to set up the front-end and back-end services.

### 1. Run the Front-End (VolView)

The VolView interface is a Node.js web app. From the project root, run:

```sh
npm install
npm run build
npm run serve
```

You can now access the VolView interface at `http://localhost:5173`.

> **Tip:** Use `npm run serve --host` to make the app accessible from other
> devices on your local network.

### 2. Run the Back-End (NVIDIA Models)

Each NVIDIA model runs in its own Python server. From the `server` directory,
install dependencies and launch each service in a separate terminal.

```sh
cd server
poetry install
```

* **Curate (Segmentation)**

  ```ph
  poetry run python -m volview_server -P 4014 -H 0.0.0.0 2025_nvidiagtcdc/nv_segment.py
  ```

* **Reason (Chat)**

  ```sh
  poetry run python -m volview_server -P 4015 -H 0.0.0.0 2025_nvidiagtcdc/chat.py
  ```

  **Note:** The Reason server preloads the NV-Reason-CXR-3B model on startup (~1 minute on first run). You'll see:
  ```
  ======================================================================
  PRELOADING MODELS ON SERVER STARTUP
  ======================================================================
  This may take ~1 minute on first run...

  [GPU detection and model loading output]

  ✅ Model preloaded successfully in 56.1s
  Server is ready to accept requests!
  ======================================================================

  ======== Running on http://0.0.0.0:4015 ========
  ```

  Wait for `"Server is ready to accept requests!"` before connecting from the UI. This ensures immediate response to your first chat message.

* **Generate (Synthetic Data)**

  ```sh
  poetry run python -m volview_server -P 4016 -H 0.0.0.0 2025_nvidiagtcdc/nv_generate.py
  ```

### 3. Connect Front-End to Back-End

Finally, connect the VolView front-end to your running model servers.

1. In the VolView UI, click the **Settings (gear) icon** to open the server
   configuration panel.

   ![A screenshot of the settings icon in the VolView
   UI](docs/assets/volview-server-config-1.png)

2. Update the URL for each service to point to the correct IP address and port
   where your Python servers are running. The panel will show a "Connected"
   status for each successful connection.

   ![A screenshot of the server configuration panel in
   VolView](docs/assets/volview-server-config-2.png)

### (Optional) 4. Saving Configs for Remote Servers

To save IP addresses for the three backend servers in the VolView Settings panel
, you can pre-configure them in the `.env` file.

1. Copy the env file local configuration file from the template:

   ```sh
   cp .env.example .env
   ```

2. Edit the **.env.** file and update the following variables with the correct
   remote IP addresses (and ports) of your running back-end servers:

   ```bash
   VITE_REMOTE_SERVER_1_URL=
   VITE_REMOTE_SERVER_2_URL=
   VITE_REMOTE_SERVER_3_URL=
   ```

   Example:

   ```bash
   VITE_REMOTE_SERVER_1_URL=http://localhost:4014
   VITE_REMOTE_SERVER_2_URL=http://localhost:4015
   VITE_REMOTE_SERVER_3_URL=http://10.50.56.30:9003

---

That's it! You are now ready to use the integrated NVIDIA models within VolView.

## Disclaimer

This software is provided **solely for research and educational purposes**.  It
is a research platform and **is not intended for clinical
use**.  

- This software has **not been reviewed or approved by the U.S. Food and Drug
  Administration (FDA)** or any other regulatory authority.  
- It must **not be used for diagnosis, treatment, or any clinical
  decision-making**.  
- No warranties or guarantees of performance, safety, or fitness for medical
  purposes are provided.  

By using this software, you acknowledge that it is for **non-clinical,
investigational research only**.

## Licenses & Attribution

This repository (`volview-gtc2025-demo`) is released under the **Apache License 2.0**.  
You are free to use, modify, and distribute this code, provided you comply with the terms of the license.  

### External Models

This demo integrates external AI models that are **not part of this repository**.  
Each model has its own license and usage terms. You are responsible for reviewing and complying with these terms when downloading or using the models.

### Important Notes

- **This repository does not redistribute model weights.**  
  Instead, it provides integration points to download and use them directly from their official sources.

- **Model license terms vary.**  
  Some models are Apache 2.0, while others (e.g., NVIDIA-hosted) may have research-only or commercial-use restrictions.  
  Always check the model card or repository for the current license.

- **Attribution.**  
  If you use this demo in your work, please attribute the external models according to their license terms (e.g., Apache NOTICE requirements, CC-BY citation, NVIDIA terms of use).

---

