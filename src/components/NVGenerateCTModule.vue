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

// Model selection
const selectedModel = ref<string>('NV-Generate-CT');

// Available model options
const modelOptions = [
  {
    value: 'NV-Generate-CT',
    title: 'NV-Generate-CT',
    subtitle: 'Synthetic CT generation with anatomical annotations',
  },
  {
    value: 'NV-Generate-MR',
    title: 'NV-Generate-MR',
    subtitle: 'Synthetic MR generation with anatomical annotations',
  },
];

// Computed values based on selected model
const modalityLabel = computed(() =>
  selectedModel.value === 'NV-Generate-CT' ? 'CT' : 'MR'
);

const modelDescription = computed(() =>
  selectedModel.value === 'NV-Generate-CT'
    ? '3D Latent Diffusion Model optimized for chest and abdomen CT generation'
    : '3D Latent Diffusion Model optimized for brain and body MR generation'
);

// Resolution options
const resolutionXYOptions = [128, 256, 384, 512];
const resolutionZOptions = [128, 256, 384, 512, 640, 768];
const selectedResolutionXY = ref(256);
const selectedResolutionZ = ref(128);

// Spacing options (dropdowns instead of sliders)
const spacingXYOptions = [0.5, 0.75, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0];
const spacingZOptions = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
const selectedSpacingXY = ref(1.5);
const selectedSpacingZ = ref(1.5);

// Model info expansion state (expanded by default)
const modelInfoExpanded = ref<number[]>([0]);

const doGenerateWithNVGenerate = async () => {
  generationLoading.value = true;
  const generationId = `${selectedModel.value.toLowerCase()}-${Date.now()}`;
  try {
    // Construct the parameters object from the UI state
    const params = {
      output_size: [selectedResolutionXY.value, selectedResolutionXY.value, selectedResolutionZ.value],
      spacing: [selectedSpacingXY.value, selectedSpacingXY.value, selectedSpacingZ.value],
      // Autoencoder memory optimization settings
      autoencoder_sliding_window_infer_size: [48, 48, 48],
      autoencoder_sliding_window_infer_overlap: 0.5,
      num_splits: 16
    };

    // Call the new unified endpoint with model parameter
    await client.call('generateWithNVGenerate', [generationId, params, selectedModel.value]);
    const generatedImageObject = nvGenerateStore.getNVGenerateResult(generationId);

    if (!generatedImageObject) {
      console.error(`No ${selectedModel.value} data found for generation ID: ${generationId}`);
      return;
    }

    // Convert the plain JS object from the store into a vtkImageData object
    const generatedImageData = vtk(generatedImageObject) as vtkImageData;

    // Add the generated data as a new image layer with dynamic name
    const imageName = `${selectedModel.value} Synthetic ${modalityLabel.value}`;
    const newImageId = imageStore.addVTKImageData(imageName, generatedImageData);

    // Set as the current/active image to display it automatically
    dataStore.setPrimarySelection(newImageId);

    // Clean up the result from the temporary store
    nvGenerateStore.removeNVGenerateResult(generationId);

    console.log(`${selectedModel.value}: Synthetic ${modalityLabel.value} scan successfully generated and loaded!`);
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
      <!-- Header with Title and Chip -->
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">mdi-creation</v-icon>
        <span class="text-h6">{{ selectedModel }}</span>
        <v-chip size="small" color="info" variant="outlined" class="ml-3">
          <v-icon start size="small">mdi-image-plus</v-icon>
          Synthetic {{ modalityLabel }} Generation
        </v-chip>
      </v-card-title>

      <v-card-text>
        <div class="text-body-2 mb-4">
          Generate high-quality synthetic 3D {{ modalityLabel }} images. Configure output size and voxel spacing below.
        </div>

        <!-- Model Selector -->
        <div class="mb-4">
          <v-select
            v-model="selectedModel"
            :items="modelOptions"
            label="Select Model"
            item-title="title"
            item-value="value"
            variant="outlined"
            density="compact"
            :disabled="generationLoading"
          >
            <template #item="{ props, item }">
              <v-list-item v-bind="props">
                <template #subtitle>
                  <span class="text-caption">{{ item.raw.subtitle }}</span>
                </template>
              </v-list-item>
            </template>
          </v-select>
        </div>

        <!-- Output Size Selection -->
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

        <!-- Spacing Selection -->
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
          @click="doGenerateWithNVGenerate"
          :loading="generationLoading"
          :disabled="!ready"
          class="mb-3 mt-4"
        >
          <v-icon left>mdi-play</v-icon>
          {{ generationLoading ? 'Generating...' : `Generate ${modalityLabel} Scan` }}
        </v-btn>
      </v-card-text>
    </v-card>

    <!-- Collapsible Model Information -->
    <v-expansion-panels v-model="modelInfoExpanded" variant="accordion">
      <v-expansion-panel>
        <v-expansion-panel-title>
          <v-icon class="mr-2">mdi-information-outline</v-icon>
          Model Information
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div class="text-body-2 mb-3">
            {{ modelDescription }}
          </div>

          <div class="text-subtitle-2 mb-2">Recommended Settings</div>
          <!-- CT Recommended Settings -->
          <v-table v-if="selectedModel === 'NV-Generate-CT'" density="compact">
            <thead>
              <tr>
                <th>Anatomy</th>
                <th>Output Size</th>
                <th>Spacing (mm)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Chest</td>
                <td>512 × 512 × 128</td>
                <td>0.8 × 0.8 × 2.5</td>
              </tr>
              <tr>
                <td>Abdomen</td>
                <td>512 × 512 × 128</td>
                <td>0.75 × 0.75 × 3.0</td>
              </tr>
              <tr>
                <td>Whole Body</td>
                <td>512 × 512 × 256</td>
                <td>1.0 × 1.0 × 1.75</td>
              </tr>
            </tbody>
          </v-table>
          <!-- MR Recommended Settings -->
          <v-table v-else density="compact">
            <thead>
              <tr>
                <th>Anatomy</th>
                <th>Output Size</th>
                <th>Spacing (mm)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Brain</strong></td>
                <td>256 × 256 × 256</td>
                <td>1.0 × 1.0 × 1.0</td>
              </tr>
              <tr>
                <td>Body</td>
                <td>256 × 256 × 128</td>
                <td>1.5 × 1.5 × 1.5</td>
              </tr>
            </tbody>
          </v-table>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>
