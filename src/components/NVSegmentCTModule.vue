<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server-1';
import { useNVSegmentStore } from '@/src/store/nv-segment';
import { useImageStore } from '@/src/store/datasets-images';
import { useSegmentGroupStore } from '@/src/store/segmentGroups';
import { useImageCacheStore } from '@/src/store/image-cache';
import { ensureSameSpace } from '@/src/io/resample/resample';
import vtkLabelMap from '@/src/vtk/LabelMap';
import {
  CATEGORICAL_COLORS,
} from '@/src/config';
import { normalizeForStore } from '@/src/utils';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtk from '@kitware/vtk.js/vtk';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import type { TypedArray } from '@kitware/vtk.js/types';
import {
  SEGMENT_MODEL_CONFIGS,
  getClassItems,
  getClassName,
} from '@/src/store/segment-classes';

// Model selection
const selectedModel = ref<string>('NV-Segment-CT');
const selectedModality = ref<string>('CT');

// Available model options
const modelOptions = [
  {
    value: 'NV-Segment-CT',
    title: 'NV-Segment-CT',
    subtitle: '132 classes - CT only (best for tumor segmentation)',
  },
  {
    value: 'NV-Segment-CTMR',
    title: 'NV-Segment-CTMR',
    subtitle: '345 classes - CT & MR (includes brain structures)',
  },
];

// Modality options (only relevant for CTMR)
const modalityOptions = [
  { value: 'CT', title: 'CT Body' },
  { value: 'MR', title: 'MR Body' },
  { value: 'MR_BRAIN', title: 'MR Brain' },
];

// Show modality selector only for CTMR model
const showModalitySelector = computed(() => selectedModel.value === 'NV-Segment-CTMR');

// Get available classes based on selected model
const availableClasses = computed(() => getClassItems(selectedModel.value));

// Current model config
const currentModelConfig = computed(() => SEGMENT_MODEL_CONFIGS[selectedModel.value]);

// Modality support text for chip display
const modalitySupport = computed(() =>
  selectedModel.value === 'NV-Segment-CT' ? 'CT Only' : 'CT & MR'
);

// Selected classes for segmentation (empty = segment everything)
const selectedClasses = ref<number[]>([]);

// Reset selected classes when model changes
watch(selectedModel, () => {
  selectedClasses.value = [];
});

const serverStore = useServerStore();
const nvSegmentStore = useNVSegmentStore();
const imageStore = useImageStore();
const segmentGroupStore = useSegmentGroupStore();
const imageCacheStore = useImageCacheStore();

const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

const segmentationLoading = ref(false);
const { currentImageID } = useCurrentImage();

// Model info expansion state
const modelInfoExpanded = ref<number[]>([]);

// --- Helpers copied from segmentGroupStore ---

const LabelmapArrayType = Uint8Array;

function convertToUint8(array: number[] | TypedArray): Uint8Array {
  const uint8Array = new Uint8Array(array.length);
  for (let i = 0; i < array.length; i++) {
    const value = array[i];
    uint8Array[i] = value < 0 || value > 255 ? 0 : value;
  }
  return uint8Array;
}

function getLabelMapScalars(imageData: vtkImageData) {
  const scalars = imageData.getPointData().getScalars();
  let values = scalars.getData();

  if (!(values instanceof LabelmapArrayType)) {
    values = convertToUint8(values);
  }

  return vtkDataArray.newInstance({
    numberOfComponents: scalars.getNumberOfComponents(),
    values,
  });
}

function toLabelMap(imageData: vtkImageData) {
  const labelmap = vtkLabelMap.newInstance(
    imageData.get('spacing', 'origin', 'direction', 'extent', 'dataDescription')
  );

  labelmap.setDimensions(imageData.getDimensions());
  labelmap.computeTransforms();

  // outline rendering only supports UInt8Array image types
  const scalars = getLabelMapScalars(imageData);
  labelmap.getPointData().setScalars(scalars);

  return labelmap;
}

// Helper function to get a color (copied from segmentGroupStore)
let nextColorIndex = 0;
const getNextColor = () => {
  const color = CATEGORICAL_COLORS[nextColorIndex];
  nextColorIndex = (nextColorIndex + 1) % CATEGORICAL_COLORS.length;
  return [...color, 255] as const;
};

// --- Component Logic ---

const doSegmentWithNVSegment = async () => {
  const baseImageId = currentImageID.value;
  if (!baseImageId) return;

  segmentationLoading.value = true;
  try {
    // Prepare the label_prompt parameter
    // Empty array means "segment everything"
    const labelPrompt = selectedClasses.value.length > 0
      ? selectedClasses.value
      : [];

    // Call the new unified endpoint with model and modality parameters
    await client.call('segmentWithNVSegment', [
      baseImageId,
      labelPrompt,
      selectedModel.value,
      selectedModality.value
    ]);
    const labelmapObject = nvSegmentStore.getNVSegmentResult(baseImageId);

    if (!labelmapObject) {
      console.error(`No ${selectedModel.value} data found for ID: ${baseImageId}`);
      return;
    }

    const labelmapImageData = vtk(labelmapObject) as vtkImageData;

    // 1. Get the parent image
    const parentImage = imageCacheStore.getVtkImageData(baseImageId);
    if (!parentImage) {
      throw new Error(`Could not find parent image data for ${baseImageId}`);
    }
    const parentName = imageStore.metadata[baseImageId]?.name ?? 'Image';
    const newGroupName = `${selectedModel.value} Result for ${parentName}`;

    // 2. Ensure segmentation is in the same space as the parent
    const matchingParentSpace = await ensureSameSpace(
      parentImage,
      labelmapImageData,
      true // true for labelmap interpolation (nearest neighbor)
    );

    // 3. Convert the vtkImageData to a vtkLabelMap
    const labelmapImage = toLabelMap(matchingParentSpace);

    // 4. Find unique values and map them to the correct segment names
    const scalarData = labelmapImage.getPointData().getScalars().getData();
    const uniqueValues = new Set<number>(scalarData);
    uniqueValues.delete(0); // 0 is always background

    const segments = Array.from(uniqueValues).map((value) => ({
      value,
      name: getClassName(selectedModel.value, value),
      color: [...getNextColor()],
      visible: true,
    }));

    const { order, byKey } = normalizeForStore(segments, 'value');

    // 5. Find and remove existing group, then add the new one

    // Check if a group with the same name and parent already exists
    const existingGroupID = Object.keys(segmentGroupStore.metadataByID).find(
      (id) => {
        const meta = segmentGroupStore.metadataByID[id];
        return meta.parentImage === baseImageId && meta.name === newGroupName;
      }
    );

    if (existingGroupID) {
      // Overwrite: Remove the old group first
      segmentGroupStore.removeGroup(existingGroupID);
    }

    // Add the new (or replacement) label map to the store
    segmentGroupStore.addLabelmap(labelmapImage, {
      name: newGroupName,
      parentImage: baseImageId,
      segments: { order, byValue: byKey },
    });
  } catch (error) {
    console.error('An error occurred during segmentation:', error);
  } finally {
    segmentationLoading.value = false;
  }
};

const hasCurrentImage = computed(() => !!currentImageID.value);

const deselectAllClasses = () => {
  selectedClasses.value = [];
};

const selectionHintText = computed(() => {
  if (selectedClasses.value.length === 0) {
    return 'No classes selected - will segment all available classes';
  }
  return `${selectedClasses.value.length} class(es) selected`;
});
</script>

<template>
  <div class="overflow-y-auto overflow-x-hidden ma-2 fill-height">
    <v-alert v-if="!ready" color="info" class="mb-4">
      Not connected to the server.
    </v-alert>

    <v-card class="mb-4">
      <!-- Merged Header with Title and Chips -->
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">mdi-auto-fix</v-icon>
        <span class="text-h6">{{ selectedModel }}</span>
        <v-chip size="small" color="info" variant="outlined" class="ml-3">
          <v-icon start size="small">mdi-cube-scan</v-icon>
          {{ currentModelConfig?.numClasses }} Classes | {{ modalitySupport }}
        </v-chip>
      </v-card-title>

      <v-card-text>
        <div class="text-body-2 mb-4">
          {{ currentModelConfig?.description }}. Foundation model for 3D medical image segmentation.
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
            :disabled="segmentationLoading"
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

        <!-- Modality Selector (only for CTMR) -->
        <div v-if="showModalitySelector" class="mb-4">
          <v-select
            v-model="selectedModality"
            :items="modalityOptions"
            label="Image Modality"
            variant="outlined"
            density="compact"
            :disabled="segmentationLoading"
          >
            <template #prepend>
              <v-icon>mdi-image-filter-center-focus</v-icon>
            </template>
          </v-select>
          <div class="text-caption text-medium-emphasis mt-1">
            <div class="mb-1">Note: NV-Segment-CTMR also works with CT images (all 132 CT classes included)</div>
            <div v-if="selectedModality === 'MR_BRAIN'" class="text-warning">
              ⚠️ Brain segmentation requires skull-stripped, normalized T1 MRI. See <a href="https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing" target="_blank" class="text-info">preprocessing guide</a>.
            </div>
          </div>
        </div>

        <!-- Class Selector -->
        <div class="mb-4">
          <v-select
            v-model="selectedClasses"
            :items="availableClasses"
            label="Select Classes to Segment"
            multiple
            chips
            closable-chips
            variant="outlined"
            density="compact"
            :disabled="segmentationLoading || !hasCurrentImage"
            class="mb-2"
          >
            <template #prepend-item>
              <v-list-item
                title="Deselect All"
                @click.stop="deselectAllClasses"
              >
                <template #prepend>
                  <v-icon>mdi-close-circle-outline</v-icon>
                </template>
              </v-list-item>
              <v-divider class="mb-2"></v-divider>
            </template>
          </v-select>

          <div class="text-caption text-medium-emphasis">
            {{ selectionHintText }}
          </div>
        </div>

        <v-btn
          color="primary"
          size="x-large"
          block
          @click="doSegmentWithNVSegment"
          :loading="segmentationLoading"
          :disabled="!ready || !hasCurrentImage"
          class="mb-3"
        >
          <v-icon left>mdi-play</v-icon>
          {{ segmentationLoading ? 'Running Segmentation...' : 'Run Segmentation' }}
        </v-btn>

        <div class="text-center text-caption" v-if="!hasCurrentImage">
          Load an image to enable segmentation.
        </div>
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
            VISTA-3D is a specialized interactive foundation model for 3D medical imaging segmentation.
            It provides accurate and adaptable analysis across anatomies, with support for segment everything,
            segment by class, and interactive point-based segmentation.
          </div>

          <v-list density="compact" lines="two">
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Architecture</v-list-item-title>
              <v-list-item-subtitle>Transformer (SAM-like)</v-list-item-subtitle>
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
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>

<style scoped>
.gap-2 {
  gap: 0.5rem;
}
</style>
