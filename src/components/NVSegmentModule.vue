<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore as useServerStore1, ConnectionState } from '@/src/store/server-1';
import { useServerStore as useServerStore4 } from '@/src/store/server-4';
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

// --- Segmentation Class Mappings ---

// Maps class index to class name for VISTA-3D (CT)
const CT_SEGMENT_CLASSES: Record<number, string> = {
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

// MRI body classes (subset of 345+ classes for MRI_BODY modality)
// This is a representative set - in practice, you'd get the full mapping from the model metadata
const MRI_SEGMENT_CLASSES: Record<number, string> = {
  1: "liver",
  2: "spleen", 
  3: "left kidney",
  4: "right kidney",
  5: "stomach",
  6: "gallbladder",
  7: "esophagus",
  8: "pancreas",
  9: "left adrenal gland",
  10: "right adrenal gland",
  11: "left lung",
  12: "right lung",
  13: "heart",
  14: "aorta",
  15: "inferior vena cava",
  16: "portal vein and splenic vein",
  17: "left iliac artery",
  18: "right iliac artery",
  19: "left iliac vena",
  20: "right iliac vena",
  21: "bladder",
  22: "prostate",
  23: "vertebrae L5",
  24: "vertebrae L4",
  25: "vertebrae L3",
  26: "vertebrae L2",
  27: "vertebrae L1",
  28: "vertebrae T12",
  29: "vertebrae T11",
  30: "vertebrae T10",
  31: "sacrum",
  32: "left femur",
  33: "right femur",
  34: "left hip",
  35: "right hip",
  36: "spinal cord",
  37: "brain stem",
  38: "cerebellum",
  39: "left cerebral hemisphere",
  40: "right cerebral hemisphere",
  41: "left thalamus",
  42: "right thalamus",
  43: "left caudate",
  44: "right caudate",
  45: "left putamen",
  46: "right putamen",
  47: "left hippocampus",
  48: "right hippocampus",
  49: "left amygdala",
  50: "right amygdala"
};

// Create items for the select dropdown
interface ClassItem {
  title: string;
  value: number;
}

type SegmentationModality = 'CT' | 'MRI';

const ctServerStore = useServerStore1();
const mriServerStore = useServerStore4();
const nvSegmentStore = useNVSegmentStore();
const imageStore = useImageStore();
const segmentGroupStore = useSegmentGroupStore();
const imageCacheStore = useImageCacheStore();

const { currentImageID } = useCurrentImage();

// Modality selection
const selectedModality = ref<SegmentationModality>('CT');

// Selected classes for segmentation (empty = segment everything)
const selectedClasses = ref<number[]>([]);

// Model info expansion state
const modelInfoExpanded = ref<number[]>([]);

// Loading state
const segmentationLoading = ref(false);

// Computed properties based on selected modality
const currentServerStore = computed(() => 
  selectedModality.value === 'CT' ? ctServerStore : mriServerStore
);

const currentClasses = computed(() => 
  selectedModality.value === 'CT' ? CT_SEGMENT_CLASSES : MRI_SEGMENT_CLASSES
);

const availableClasses = computed((): ClassItem[] => 
  Object.entries(currentClasses.value).map(([index, name]) => ({
    title: name,
    value: parseInt(index, 10),
  }))
);

const ready = computed(() => 
  currentServerStore.value.connState === ConnectionState.Connected
);

const hasCurrentImage = computed(() => !!currentImageID.value);

const modalityConfig = computed(() => {
  if (selectedModality.value === 'CT') {
    return {
      title: 'NV-Segment-CT',
      subtitle: 'Whole Body CT Segmentation',
      description: 'Foundation model for 3D CT segmentation. Automatically segment anatomical structures in the loaded CT image.',
      icon: 'mdi-auto-fix',
      chipIcon: 'mdi-cube-scan',
      modelName: 'VISTA-3D',
      modelInfo: {
        architecture: 'Transformer (SAM-like)',
        version: 'v1.0',
        runtime: 'MONAI Core v1.5',
        description: 'VISTA-3D is a specialized interactive foundation model for 3D medical imaging segmentation. It provides accurate and adaptable analysis across anatomies, with support for segment everything, segment by class, and interactive point-based segmentation.'
      }
    };
  }
  return {
    title: 'NV-Segment-CTMR',
    subtitle: 'Multi-Modal MRI Segmentation',
    description: 'Foundation model for 3D MRI segmentation. Automatically segment anatomical structures in the loaded MRI image using the unified CT/MR model.',
    icon: 'mdi-brain',
    chipIcon: 'mdi-medical-bag',
    modelName: 'NV-Segment-CTMR',
    modelInfo: {
      architecture: 'Transformer (SAM-like)',
      version: 'v0.1',
      runtime: 'MONAI Core v1.5 + HuggingFace',
      description: 'NV-Segment-CTMR is a unified foundation model for 3D medical image segmentation that excels at accurate, adaptable, automatic segmentation across anatomies and modalities, including CT and MR imaging. Supports 345+ anatomical classes.'
    }
  };
});

const selectionHintText = computed(() => {
  if (selectedClasses.value.length === 0) {
    const modalityText = selectedModality.value === 'MRI' ? 'MRI_BODY preset' : 'all available classes';
    return `No classes selected - will segment ${modalityText}`;
  }
  return `${selectedClasses.value.length} class(es) selected`;
});

// Watch modality changes to clear selection
watch(selectedModality, () => {
  selectedClasses.value = [];
});

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

// Helper for default names
const makeDefaultSegmentName = (value: number) => `Segment ${value}`;

/**
 * Picks a unique name for a segment group with a given prefix.
 */
function pickUniqueSegmentGroupName(
  baseName: string,
  parentID: string,
  prefix: string = 'Segment Group'
) {
  // Get all existing names for this parent image
  const existingNames = new Set(
    Object.values(segmentGroupStore.metadataByID)
      .filter((meta) => meta.parentImage === parentID)
      .map((meta) => meta.name)
  );

  let index = 1;
  let name = `${prefix} for ${baseName}`; // Initial name suggestion

  // Keep checking for a unique name, appending (index) if needed
  while (existingNames.has(name)) {
    index++;
    name = `${prefix} for ${baseName} (${index})`;
  }

  return name;
}

// --- Component Logic ---

const deselectAllClasses = () => {
  selectedClasses.value = [];
};

const doSegmentation = async () => {
  const baseImageId = currentImageID.value;
  if (!baseImageId) return;

  segmentationLoading.value = true;
  try {
    // Prepare the parameters based on modality
    const labelPrompt = selectedClasses.value.length > 0 ? selectedClasses.value : [];
    
    let rpcCall: string;
    let rpcArgs: any[];
    
    if (selectedModality.value === 'CT') {
      rpcCall = 'segmentWithNVSegmentCT';
      rpcArgs = [baseImageId, labelPrompt];
    } else {
      rpcCall = 'segmentWithNVSegmentMRI';
      rpcArgs = [baseImageId, labelPrompt, 'MRI_BODY']; // Add modality parameter for MRI
    }

    await currentServerStore.value.client.call(rpcCall, rpcArgs);
    const labelmapObject = nvSegmentStore.getNVSegmentResult(baseImageId);

    if (!labelmapObject) {
      console.error(`No segmentation data found for ID: ${baseImageId}`);
      return;
    }

    const labelmapImageData = vtk(labelmapObject) as vtkImageData;

    // 1. Get the parent image
    const parentImage = imageCacheStore.getVtkImageData(baseImageId);
    if (!parentImage) {
      throw new Error(`Could not find parent image data for ${baseImageId}`);
    }
    const parentName = imageStore.metadata[baseImageId]?.name ?? 'Image';

    // Generate a unique name for this segment group
    const modelPrefix = selectedModality.value === 'CT' ? 'NV-Segment-CT' : 'NV-Segment-CTMR';
    const newGroupName = pickUniqueSegmentGroupName(
      parentName,
      baseImageId,
      modelPrefix
    );

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
      name: currentClasses.value[value] || makeDefaultSegmentName(value),
      color: [...getNextColor()],
      visible: true,
    }));

    const { order, byKey } = normalizeForStore(segments, 'value');

    // 5. Add the new segment group to the store
    segmentGroupStore.addLabelmap(labelmapImage, {
      name: newGroupName,
      parentImage: baseImageId,
      segments: { order, byValue: byKey },
    });

    console.log(`Segmentation successfully created: ${newGroupName}`);
  } catch (error) {
    console.error('An error occurred during segmentation:', error);
  } finally {
    segmentationLoading.value = false;
  }
};
</script>

<template>
  <div class="overflow-y-auto overflow-x-hidden ma-2 fill-height">
    <v-alert v-if="!ready" color="info" class="mb-4">
      Not connected to the {{ selectedModality === 'CT' ? 'CT segmentation' : 'MRI segmentation' }} server.
    </v-alert>

    <v-card class="mb-4">
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2">{{ modalityConfig.icon }}</v-icon>
        <span class="text-h6">{{ modalityConfig.title }}</span>
        <v-chip size="small" color="info" variant="outlined" class="ml-3">
          <v-icon start size="small">{{ modalityConfig.chipIcon }}</v-icon>
          {{ modalityConfig.subtitle }}
        </v-chip>
      </v-card-title>

      <v-card-text>
        <div class="text-body-2 mb-4">
          {{ modalityConfig.description }}
        </div>

        <!-- Modality Selection -->
        <div class="mb-4">
          <v-radio-group 
            v-model="selectedModality" 
            inline
            :disabled="segmentationLoading"
            class="mb-3"
          >
            <template #label>
              <div class="text-subtitle-2 font-weight-medium">Imaging Modality</div>
            </template>
            <v-radio label="CT Segmentation" value="CT" class="mr-4"></v-radio>
            <v-radio label="MRI Segmentation" value="MRI"></v-radio>
          </v-radio-group>
        </div>

        <!-- Class Selection -->
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

        <!-- Run Segmentation Button -->
        <v-btn
          color="primary"
          size="x-large"
          block
          @click="doSegmentation"
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

    <!-- Model Information -->
    <v-expansion-panels v-model="modelInfoExpanded" variant="accordion">
      <v-expansion-panel>
        <v-expansion-panel-title>
          <v-icon class="mr-2">mdi-information-outline</v-icon>
          Model Information
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div class="text-body-2 mb-3">
            {{ modalityConfig.modelInfo.description }}
          </div>

          <v-list density="compact" lines="two">
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Model Name</v-list-item-title>
              <v-list-item-subtitle>{{ modalityConfig.modelName }}</v-list-item-subtitle>
            </v-list-item>
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Architecture</v-list-item-title>
              <v-list-item-subtitle>{{ modalityConfig.modelInfo.architecture }}</v-list-item-subtitle>
            </v-list-item>
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Model Version</v-list-item-title>
              <v-list-item-subtitle>{{ modalityConfig.modelInfo.version }}</v-list-item-subtitle>
            </v-list-item>
            <v-list-item>
              <v-list-item-title class="font-weight-bold">Runtime Engine</v-list-item-title>
              <v-list-item-subtitle>{{ modalityConfig.modelInfo.runtime }}</v-list-item-subtitle>
            </v-list-item>
            <v-list-item v-if="selectedModality === 'MRI'">
              <v-list-item-title class="font-weight-bold">Supported Classes</v-list-item-title>
              <v-list-item-subtitle>345+ anatomical structures</v-list-item-subtitle>
            </v-list-item>
            <v-list-item v-else>
              <v-list-item-title class="font-weight-bold">Supported Classes</v-list-item-title>
              <v-list-item-subtitle>132 anatomical structures</v-list-item-subtitle>
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