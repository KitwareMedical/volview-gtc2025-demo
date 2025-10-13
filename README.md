# VolView + NVIDIA Integration for GTC 2025

This repository is a fork of [Kitware's
VolView](https://github.com/Kitware/VolView) customized to showcase VolView's
extensibility with cutting-edge healthcare models from NVIDIA.

---

## Overview

This fork integrates three NVIDIA Clara foundation models into the VolView
interface, each accessible via a dedicated tab. Each of the models are connected
to VolView via compute-independent and scalable backend servers.

For general VolView features, documentation, and issue tracking, please refer to
the official [Kitware/VolView repository](https://github.com/Kitware/VolView).
This README focuses exclusively on the NVIDIA integrations.

---

## NVIDIA Clara Model Integration

This version of VolView includes three additional tabs: Curate, Reason, and
Generate.

### Curate Tab (Segmentation)

The Curate tab uses the NVIDIA Clara NV-Curate-CTMR-v2 model for automatic 3D
segmentation.

- What it does: Automatically identifies and segments anatomical structures
  within a loaded 3D CT or MR volume.
- How to use:
  - Load a compatible 3D dataset.
  - Navigate to the "Curate" tab.
  - Click "Run Segmentation".
- Output: A new segmentation layer is added to the scene with labeled anatomical
  structures.
- Parameters: This is a zero-shot model with no user-configurable parameters in
  the UI.

### Reason Tab (Multimodal Chat)

The Reason tab integrates a multimodal chatbot powered by the NVIDIA Clara
NV-Reason-CXR-3B model.

- What it does: Allows text-based conversation about the loaded medical image.
  Ask it to identify findings, describe features, or answer questions.
- How to use:
  - Load an image.
  - Navigate to the "Reason" tab.
  - Type your question into the message box and send.
- Output: The model's text response appears in the chat window.
- Parameters: The primary input is the text prompt. Different chat models can be
  selected from the dropdown menu.

### Generate Tab (Synthetic Data Generation)

The Generate tab uses the NVIDIA Clara NV-Generate-CTMR-v2 model to create
synthetic 3D CT scans.

- What it does: Generates a new, realistic 3D CT volume based on user-defined
  parameters.
- How to use:
  - Navigate to the "Generate" tab.
  - Configure the desired parameters.
  - Click "Generate CT Scan".
- Output: A new 3D volume is generated and loaded into VolView.
- Parameters:
  - Body Region: Choose between 'Abdomen' or 'Chest'.
  - Anatomy Part: Select a specific primary organ (e.g., 'liver', 'spleen',
    'aorta').
  - Resolution:
    - XY Resolution: Set the resolution for the axial plane (256 or 512).
    - Z Resolution: Set the number of slices in the volume (128, 256, or 512).
  - Spacing (mm):
    - Coronal/Sagittal Spacing: Adjust pixel spacing in X and Y.
    - Axial Spacing: Adjust slice spacing in Z.
    > Note: For 512 XY resolution, spacing is locked to 1mm.

---

## Setting up VolView w/ NVIDIA Models

### Start Front-End Server

VolView is served as a Node.js web app. To serve the web app, in the project
root, run the following:

```sh
npm install
npm run build
npm run serve
```

Then, you should be able to use the web app by connecting to
`http://localhost:5174/` if run on your local machine (note: if the port is
used, VITE will serve on a different port).

> If you run `npm run serve --host`, the web app should be accessible via any
> device (e.g. mobile phone) at `http://:machineIp:5174/`, as long as
> `machineIp` is routable.

### Start Back-End Servers

The "back-end" refers to the NVIDIA models to be run on independent,
scalable machines. They can also be run on the same machine. The application
code assumes that:

- "Server 1" is running the "Curate" segmentation model.
- "Server 2" is running the "Reason" reasoning model.
- "Server 3" is running the "Generate" 3D CT generation model.

#### Curate

To start the python server running the "Curate" segmentation model on port
`4014`:

```sh
cd server
poetry install
poetry run python -m volview_server -P 4014 -H 0.0.0.0 2025_nvidiagtcdc/vista3d.py
```

#### Reason

To start the python server running the "Reason" reasoning model on port `4015`:

```sh
cd server
poetry install
poetry run python -m volview_server -P 4015 -H 0.0.0.0 2025_nvidiagtcdc/chat.py
```

#### Generate

To start the python server running the "Generate" 3D CT generation model on port
`4016`:

```sh
cd server
poetry install
poetry run python -m volview_server -P 4016 -H 0.0.0.0 2025_nvidiagtcdc/maisi.py
```

### Connecting Front-End to Back-End

You configure VolView to point to each server via the configuration icon:

![A screenshot of the configuration
icon](./docs/assets/volview-server-config-1.png)

Then, you point each of the servers to the machines running the Python servers
by modifying the links. In the example below, Curate and Reason are being run on
my machine, but Generate is being run on `10.50.56.30` on port `9003`, which is
a cluster accessible on my local intranet:

![A screenshot of the configuration
icon](./docs/assets/volview-server-config-2.png)

For each server, you can independently try "connect" and the configuration
interface will inform you if the server is connected. By default, the servers
are all configured to be run on localhost on different ports, but you can
replace the links with arbitrary machines and arbitrary ports, given that you
have run the server on the appropriate corresponding port above.

---
