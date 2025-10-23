<script setup lang="ts">
import { computed, ref } from 'vue';
import { useServerStore, ConnectionState } from '@/src/store/server-3';
import { useNVGenerateStore } from '@/src/store/nv-generate';
import { useImageStore } from '@/src/store/datasets-images';
import { useDatasetStore } from '@/src/store/datasets';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtk from '@kitware/vtk.js/vtk';

const serverStore = useServerStore();
const nvGenerateStore = useNVGenerateStore();
const imageStore = useImageStore();
const dataStore = useDatasetStore();

const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

const generationLoading = ref(false);

// Resolution options
const resolutionXYOptions = [256, 384, 512];
const resolutionZOptions = [128, 256, 384, 512, 640, 768];
const selectedResolutionXY = ref(512);
const selectedResolutionZ = ref(512);

// Spacing options (dropdowns instead of sliders)
const spacingXYOptions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const spacingZOptions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
const selectedSpacingXY = ref(1.0);
const selectedSpacingZ = ref(1.0);

// Model info expansion state
const modelInfoExpanded = ref<number[]>([]);

const doGenerateWithNVGenerateCT = async () => {
  generationLoading.value = true;
  const generationId = `nv-generate-ct-${Date.now()}`;
  try {
    // Construct the parameters object from the UI state
    const params = {
      output_size: [selectedResolutionXY.value, selectedResolutionXY.value, selectedResolutionZ.value],
      spacing: [selectedSpacingXY.value, selectedSpacingXY.value, selectedSpacingZ.value]
    };
    await client.call('generateWithMAISI', [generationId, params]);
    const generatedImageObject = nvGenerateStore.getNVGenerateResult(generationId);

    if (!generatedImageObject) {
      console.error(`No NV-Generate-CT data found for generation ID: ${generationId}`);
      return;
    }

    // Convert the plain JS object from the store into a vtkImageData object
    const generatedImageData = vtk(generatedImageObject) as vtkImageData;

    // Add the generated data as a new image layer
    const newImageId = imageStore.addVTKImageData('NV-Generate-CT Synthetic CT', generatedImageData);

    // Set as the current/active image to display it automatically
    dataStore.setPrimarySelection(newImageId);

    // Clean up the result from the temporary store
    nvGenerateStore.removeNVGenerateResult(generationId);

    console.log('NV-Generate-CT: Synthetic CT scan successfully generated and loaded!');
  } catch (error) {
    console.error('An error occurred during generation:', error);
  } finally {
    generationLoading.value = false;
  }
};
</script>

<template>
  <div class="overflow-y-auto overflow-x-hidden ma-2 fill-height">
    <v-alert v-if="!ready" color="info" class="mb-4">
      Not connected to the server.
    </v-alert>

    <v-card class="mb-4">
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">mdi-creation</v-icon>
        <span class="text-h6">NV-Generate-CT</span>
        <v-chip size="small" color="info" variant="outlined" class="ml-3">
          <v-icon start size="small">mdi-image-plus</v-icon>
          Synthetic CT Generation
        </v-chip>
      </v-card-title>

      <v-card-text>
        <div class="text-body-2 mb-4">
          Generate high-quality synthetic 3D CT images. Configure output size and voxel spacing below.
        </div>

        <div class="text-subtitle-2 mb-2">Output Size</div>
        <v-row>
          <v-col cols="6">
            <v-select
              v-model="selectedResolutionXY"
              :items="resolutionXYOptions"
              label="X/Y Resolution"
              variant="outlined"
              density="compact"
              :disabled="generationLoading"
            />
          </v-col>
          <v-col cols="6">
            <v-select
              v-model="selectedResolutionZ"
              :items="resolutionZOptions"
              label="Z Resolution"
              variant="outlined"
              density="compact"
              :disabled="generationLoading"
            />
          </v-col>
        </v-row>

        <div class="text-subtitle-2 mb-2 mt-2">Voxel Spacing (mm)</div>
        <v-row>
          <v-col cols="6">
            <v-select
              v-model="selectedSpacingXY"
              :items="spacingXYOptions"
              label="X/Y Spacing"
              variant="outlined"
              density="compact"
              :disabled="generationLoading"
              suffix="mm"
            />
          </v-col>
          <v-col cols="6">
            <v-select
              v-model="selectedSpacingZ"
              :items="spacingZOptions"
              label="Z Spacing"
              variant="outlined"
              density="compact"
              :disabled="generationLoading"
              suffix="mm"
            />
          </v-col>
        </v-row>

        <v-btn
          color="primary"
          size="x-large"
          block
          @click="doGenerateWithNVGenerateCT"
          :loading="generationLoading"
          :disabled="!ready"
          class="mb-3 mt-4"
        >
          <v-icon left>mdi-play</v-icon>
          {{ generationLoading ? 'Generating...' : 'Generate CT Scan' }}
        </v-btn>
      </v-card-text>
    </v-card>

    <v-expansion-panels v-model="modelInfoExpanded" variant="accordion">
      <v-expansion-panel>
        <v-expansion-panel-title>
          <v-icon class="mr-2">mdi-information-outline</v-icon>
          Model Information
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div class="text-body-2 mb-3">
            NV-Generate-CT is a 3D Latent Diffusion Model designed for generating high-quality synthetic CT images.
            This model excels in data augmentation and creating realistic medical imaging data up to 512×512×768 voxels
            with variable voxel sizes ranging from 0.5mm to 5.0mm.
          </div>

          <v-list density="compact" lines="two">
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Architecture</v-list-item-title>
              <v-list-item-subtitle>3D UNet + Attention Blocks</v-list-item-subtitle>
            </v-list-item>
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Model Version</v-list-item-title>
              <v-list-item-subtitle>v1.0</v-list-item-subtitle>
            </v-list-item>
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Runtime Engine</v-list-item-title>
              <v-list-item-subtitle>MONAI Core v1.5</v-list-item-subtitle>
            </v-list-item>
          </v-list>

          <div class="text-subtitle-2 mt-4 mb-2">Recommended Settings</div>
          <v-table density="compact">
            <thead>
              <tr>
                <th>Output Size</th>
                <th>Spacing</th>
                <th>Approx. GPU Mem.</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>256 × 256 × 256</td>
                <td>1.5 × 1.5 × 1.5 mm</td>
                <td>15.4 GB</td>
              </tr>
              <tr>
                <td>512 × 512 × 128</td>
                <td>0.8 × 0.8 × 2.5 mm</td>
                <td>15.7 GB</td>
              </tr>
              <tr>
                <td>512 × 512 × 512</td>
                <td>1.0 × 1.0 × 1.0 mm</td>
                <td>22.8 GB</td>
              </tr>
            </tbody>
          </v-table>
          </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>
