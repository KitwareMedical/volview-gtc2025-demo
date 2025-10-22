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

### 🧠 Curate (Segmentation)

The Curate tab uses the **NVIDIA Clara NV-Curate-CTMR-v2** model to perform
automatic 3D segmentation of anatomical structures.

* **How to Use:**

  1. Load a compatible 3D CT or MR dataset.

  2. Navigate to the **Curate** tab.

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
VolView](docs/assets/generate_tab_example.jpeg)

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

That's it! You are now ready to use the integrated NVIDIA models within VolView.
