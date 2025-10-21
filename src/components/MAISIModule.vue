<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useServerStore, ConnectionState } from '@/src/store/server-3';
import { useNVGenerateStore } from '@/src/store/nv-generate';
import { useImageStore } from '@/src/store/datasets-images';
import { useDatasetStore } from '@/src/store/datasets';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtk from '@kitware/vtk.js/vtk';
import NVIDIAModelCard from './NVIDIAModelCard.vue';

const serverStore = useServerStore();
const nvGenerateStore = useNVGenerateStore();
const imageStore = useImageStore();
const datasetStore = useDatasetStore();

const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

// --- State for Configurable Parameters ---

const generationLoading = ref(false);

// Anatomy selections based on your provided list
const anatomyData = {
  abdomen: ['liver', 'spleen', 'pancreas', 'right kidney', 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney', 'bladder', 'small bowel', 'hepatic vessel', 'colon'],
  chest: ['aorta', 'inferior vena cava', 'esophagus', 'portal vein and splenic vein', 'left lung upper lobe', 'left lung lower lobe', 'right lung upper lobe', 'right lung middle lobe', 'right lung lower lobe', 'left iliac artery', 'right iliac artery', 'left iliac vena', 'right iliac vena']
};
const bodyRegions = Object.keys(anatomyData);

const selectedRegion = ref<'abdomen' | 'chest'>('abdomen');
const selectedPart = ref<string>('liver');

// MAISI Model Card Data
const maisiCard = {
  modelName: 'NVIDIA MAISI',
  subtitle: 'Medical AI for Synthetic Imaging',
  icon: 'mdi-creation',
  chips: [
    { text: 'Synthetic CT Generation', icon: 'mdi-image-plus', color: 'info' },
    { text: 'Research Use Only', icon: 'mdi-flask', color: 'warning' },
  ],
  description: 'NVIDIA MAISI is a state-of-the-art 3D Latent Diffusion Model designed for generating high-quality synthetic CT images with or without anatomical annotations. This AI model excels in data augmentation and creating realistic medical imaging data.',
  details: [
    { key: 'Architecture', value: 'ControlNet + 3D UNet + attention blocks' },
    { key: 'Model Version', value: '0.3.1' },
    { key: 'Runtime Engine', value: 'MONAI Core v.1.3.2' },
    { key: 'Supported Hardware', value: 'NVIDIA Ampere, NVIDIA Hopper' },
  ],
  references: [
    { text: 'Guo, P., et al. (2024). MAISI: Medical AI for Synthetic Imaging.', url: 'https://arxiv.org/abs/2409.11169' },
    { text: 'Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models.', url: 'https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf' },
  ]
};

// Computed property to dynamically update anatomy parts based on selected region
const anatomyParts = computed(() => {
  return anatomyData[selectedRegion.value];
});

// Watch for region changes to reset the selected anatomy part to the first in the new list
watch(selectedRegion, (newRegion) => {
  if (newRegion) {
    selectedPart.value = anatomyData[newRegion][0];
  } else {
    selectedPart.value = '';
  }
});

// Resolution selections
const resolutionXYOptions = [256, 512];
const resolutionZOptions = [128, 256, 512];
const selectedResolutionXY = ref(256);
const selectedResolutionZ = ref(128);

// Spacing selections
const coronalSagittalSpacing = ref(1.5);
const axialSpacing = ref(1.5);

// Rule: When XY resolution is 512, lock spacing to 1
const areSpacingsLocked = computed(() => selectedResolutionXY.value === 512);

watch(selectedResolutionXY, (newVal) => {
  if (newVal === 512) {
    coronalSagittalSpacing.value = 1;
    axialSpacing.value = 1;
  }
}, { immediate: true });

const doGenerateWithMAISI = async () => {
  generationLoading.value = true;
  const generationId = `maisi-gen-${Date.now()}`;
  try {
    // Construct the parameters object from the UI state
    const params = {
      anatomy_list: [selectedPart.value],
      output_size: [selectedResolutionXY.value, selectedResolutionXY.value, selectedResolutionZ.value],
      spacing: [coronalSagittalSpacing.value, coronalSagittalSpacing.value, axialSpacing.value]
    };
    await client.call('generateWithMAISI', [generationId, params]);
    const generatedImageObject = nvGenerateStore.getNVGenerateResult(generationId);

    if (!generatedImageObject) {
      console.error(`No MAISI data found for generation ID: ${generationId}`);
      return;
    }

    // Convert the plain JS object from the store into a vtkImageData object
    const generatedImageData = vtk(generatedImageObject) as vtkImageData;

    // get new image id and set
    const newImageId = imageStore.addVTKImageData('MAISI Generated CT', generatedImageData);
    datasetStore.setPrimarySelection(newImageId);

    // Clean up the result from the temporary store
    nvGenerateStore.removeNVGenerateResult(generationId);

    console.log('MAISI CT Scan successfully generated and loaded!');
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
      <v-card-title class="text-h6">
        <v-icon class="mr-2">mdi-creation</v-icon>
        MAISI Image Generation
      </v-card-title>
      <v-card-text>
        <div class="text-body-2 mb-4">
          Configure the parameters below to generate a new synthetic 3D CT scan.
        </div>

        <v-select
          v-model="selectedRegion"
          :items="bodyRegions"
          label="Body Region"
          variant="outlined"
          density="compact"
          class="mb-4"
          :disabled="generationLoading"
        />

        <v-select
          v-model="selectedPart"
          :items="anatomyParts"
          label="Anatomy Part"
          variant="outlined"
          density="compact"
          class="mb-4"
          :disabled="!selectedRegion || generationLoading"
        />

        <v-row>
          <v-col cols="6">
            <v-select
              v-model="selectedResolutionXY"
              :items="resolutionXYOptions"
              label="XY Resolution"
              variant="outlined"
              density="compact"
              class="mb-4"
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
              class="mb-4"
              :disabled="generationLoading"
            />
          </v-col>
        </v-row>

        <v-slider
          v-model="coronalSagittalSpacing"
          label="Coronal/Sagittal Spacing (mm)"
          :min="1"
          :max="3"
          :step="0.5"
          thumb-label
          class="mb-2"
          :disabled="areSpacingsLocked || generationLoading"
        />

        <v-slider
          v-model="axialSpacing"
          label="Axial Spacing (mm)"
          :min="1"
          :max="3"
          :step="0.5"
          thumb-label
          class="mb-2"
          :disabled="areSpacingsLocked || generationLoading"
        />

        <v-alert v-if="areSpacingsLocked" type="info" variant="tonal" density="compact" class="mb-4">
          Spacings are locked to 1mm for 512 XY resolution.
        </v-alert>

        <v-btn
          color="primary"
          size="x-large"
          block
          @click="doGenerateWithMAISI"
          :loading="generationLoading"
          :disabled="!ready || !selectedPart"
          class="mb-3 mt-4"
        >
          <v-icon left>mdi-play</v-icon>
          {{ generationLoading ? 'Generating...' : 'Generate CT Scan' }}
        </v-btn>
      </v-card-text>
    </v-card>

     <NVIDIAModelCard
      :model-name="maisiCard.modelName"
      :subtitle="maisiCard.subtitle"
      :icon="maisiCard.icon"
      :chips="maisiCard.chips"
      :description="maisiCard.description"
      :details="maisiCard.details"
      :references="maisiCard.references"
    />
  </div>
</template>
