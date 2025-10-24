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
   sudo apt install python3.11 python3.11-pip python3.11-venv
   
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

