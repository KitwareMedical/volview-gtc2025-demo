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
