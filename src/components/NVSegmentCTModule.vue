<script setup lang="ts">
import { computed, ref } from 'vue';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server-1';
import { useNVSegmentStore } from '@/src/store/nv-segment';
import { useImageStore } from '@/src/store/datasets-images';
import { useSegmentGroupStore } from '@/src/store/segmentGroups';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtk from '@kitware/vtk.js/vtk';

// --- Segmentation Class Mapping ---
// Maps class index to class name for VISTA-3D
const SEGMENT_CLASSES: Record<number, string> = {
  1: "liver",
  2: "kidney",
  3: "spleen",
  4: "pancreas",
  5: "right kidney",
  6: "aorta",
  7: "inferior vena cava",
  8: "right adrenal gland",
  9: "left adrenal gland",
  10: "gallbladder",
  11: "esophagus",
  12: "stomach",
  13: "duodenum",
  14: "left kidney",
  15: "bladder",
  16: "prostate or uterus",
  17: "portal vein and splenic vein",
  18: "rectum",
  19: "small bowel",
  20: "lung",
  21: "bone",
  22: "brain",
  23: "lung tumor",
  24: "pancreatic tumor",
  25: "hepatic vessel",
  26: "hepatic tumor",
  27: "colon cancer primaries",
  28: "left lung upper lobe",
  29: "left lung lower lobe",
  30: "right lung upper lobe",
  31: "right lung middle lobe",
  32: "right lung lower lobe",
  33: "vertebrae L5",
  34: "vertebrae L4",
  35: "vertebrae L3",
  36: "vertebrae L2",
  37: "vertebrae L1",
  38: "vertebrae T12",
  39: "vertebrae T11",
  40: "vertebrae T10",
  41: "vertebrae T9",
  42: "vertebrae T8",
  43: "vertebrae T7",
  44: "vertebrae T6",
  45: "vertebrae T5",
  46: "vertebrae T4",
  47: "vertebrae T3",
  48: "vertebrae T2",
  49: "vertebrae T1",
  50: "vertebrae C7",
  51: "vertebrae C6",
  52: "vertebrae C5",
  53: "vertebrae C4",
  54: "vertebrae C3",
  55: "vertebrae C2",
  56: "vertebrae C1",
  57: "trachea",
  58: "left iliac artery",
  59: "right iliac artery",
  60: "left iliac vena",
  61: "right iliac vena",
  62: "colon",
  63: "left rib 1",
  64: "left rib 2",
  65: "left rib 3",
  66: "left rib 4",
  67: "left rib 5",
  68: "left rib 6",
  69: "left rib 7",
  70: "left rib 8",
  71: "left rib 9",
  72: "left rib 10",
  73: "left rib 11",
  74: "left rib 12",
  75: "right rib 1",
  76: "right rib 2",
  77: "right rib 3",
  78: "right rib 4",
  79: "right rib 5",
  80: "right rib 6",
  81: "right rib 7",
  82: "right rib 8",
  83: "right rib 9",
  84: "right rib 10",
  85: "right rib 11",
  86: "right rib 12",
  87: "left humerus",
  88: "right humerus",
  89: "left scapula",
  90: "right scapula",
  91: "left clavicula",
  92: "right clavicula",
  93: "left femur",
  94: "right femur",
  95: "left hip",
  96: "right hip",
  97: "sacrum",
  98: "left gluteus maximus",
  99: "right gluteus maximus",
  100: "left gluteus medius",
  101: "right gluteus medius",
  102: "left gluteus minimus",
  103: "right gluteus minimus",
  104: "left autochthon",
  105: "right autochthon",
  106: "left iliopsoas",
  107: "right iliopsoas",
  108: "left atrial appendage",
  109: "brachiocephalic trunk",
  110: "left brachiocephalic vein",
  111: "right brachiocephalic vein",
  112: "left common carotid artery",
  113: "right common carotid artery",
  114: "costal cartilages",
  115: "heart",
  116: "left kidney cyst",
  117: "right kidney cyst",
  118: "prostate",
  119: "pulmonary vein",
  120: "skull",
  121: "spinal cord",
  122: "sternum",
  123: "left subclavian artery",
  124: "right subclavian artery",
  125: "superior vena cava",
  126: "thyroid gland",
  127: "vertebrae S1",
  128: "bone lesion",
  129: "kidney mass",
  130: "liver tumor",
  131: "vertebrae L6",
  132: "airway"
};

// Create items for the select dropdown
interface ClassItem {
  title: string;
  value: number;
}

const availableClasses: ClassItem[] = Object.entries(SEGMENT_CLASSES).map(
  ([index, name]) => ({
    title: name,
    value: parseInt(index, 10),
  })
);

const serverStore = useServerStore();
const nvSegmentStore = useNVSegmentStore();
const imageStore = useImageStore();
const segmentGroupStore = useSegmentGroupStore();

const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

const segmentationLoading = ref(false);
const { currentImageID } = useCurrentImage();

// Selected classes for segmentation (empty = segment everything)
const selectedClasses = ref<number[]>([]);

// Model info expansion state
const modelInfoExpanded = ref<number[]>([]);

const doSegmentWithNVSegmentCT = async () => {
  const baseImageId = currentImageID.value;
  if (!baseImageId) return;

  segmentationLoading.value = true;
  try {
    // Prepare the label_prompt parameter
    // Empty array means "segment everything"
    const labelPrompt = selectedClasses.value.length > 0
      ? selectedClasses.value
      : [];

    await client.call('segmentWithNVSegmentCT', [baseImageId, labelPrompt]);
    const labelmapObject = nvSegmentStore.getNVSegmentResult(baseImageId);

    if (!labelmapObject) {
      console.error(`No NV-Segment-CT data found for ID: ${baseImageId}`);
      return;
    }

    // Convert the plain JS object and assert its type to vtkImageData
    const labelmapImageData = vtk(labelmapObject) as vtkImageData;

    // Add the data as a new image layer
    const newImageId = imageStore.addVTKImageData(
      'NV-Segment-CT Segmentation Result',
      labelmapImageData
    );

    // Convert the new image layer to a labelmap
    segmentGroupStore.convertImageToLabelmap(newImageId, baseImageId);

    console.log('Segmentation successfully loaded and converted to labelmap!');
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
        <span class="text-h6">NV-Segment-CT</span>
        <v-chip size="small" color="info" variant="outlined" class="ml-3">
          <v-icon start size="small">mdi-cube-scan</v-icon>
          Whole Body Segmentation
        </v-chip>
      </v-card-title>

      <v-card-text>
        <div class="text-body-2 mb-4">
          Foundation model for 3D CT segmentation. Automatically segment anatomical structures in the loaded CT image.
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
          @click="doSegmentWithNVSegmentCT"
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
