<script setup lang="ts">
import { computed, ref } from 'vue';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server-1';
import { useVista3dStore } from '@/src/store/vista3d';
import { useImageStore } from '@/src/store/datasets-images';
import { useSegmentGroupStore } from '@/src/store/segmentGroups';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtk from '@kitware/vtk.js/vtk';

const serverStore = useServerStore();
const vista3dStore = useVista3dStore();
const imageStore = useImageStore();
const segmentGroupStore = useSegmentGroupStore();

const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

const segmentWithVista3DLoading = ref(false);
const { currentImageID } = useCurrentImage();

const doSegmentWithVista3D = async () => {
  const baseImageId = currentImageID.value;
  if (!baseImageId) return;

  segmentWithVista3DLoading.value = true;
  try {
    await client.call('segmentWithVista3D', [baseImageId]);
    const labelmapObject = vista3dStore.getVista3dResult(baseImageId);

    if (!labelmapObject) {
      console.error(`No vista3d data found for ID: ${baseImageId}`);
      return;
    }

    // Convert the plain JS object and assert its type to vtkImageData
    const labelmapImageData = vtk(labelmapObject) as vtkImageData;

    // Add the data as a new image layer with corrected arguments.
    const newImageId = imageStore.addVTKImageData(
      'Clara NV-Curate-CTMR-v2 Segmentation Result',
      labelmapImageData
    );

    // Convert the new image layer to a labelmap.
    segmentGroupStore.convertImageToLabelmap(newImageId, baseImageId);

    console.log('Segmentation successfully loaded and converted to labelmap!');
  } catch (error) {
    console.error('An error occurred during segmentation:', error);
  } finally {
    segmentWithVista3DLoading.value = false;
  }
};

const hasCurrentImage = computed(() => !!currentImageID.value);
</script>

<template>
  <div class="overflow-y-auto overflow-x-hidden ma-2 fill-height">
    <v-alert v-if="!ready" color="info" class="mb-4">
      Not connected to the server.
    </v-alert>

    <v-card>
      <v-card-title class="text-h6">
        <v-icon class="mr-2">mdi-auto-fix</v-icon>
        Clara NV-Curate-CTMR-v2
      </v-card-title>
      <v-card-text>
        <div class="text-body-2 mb-4">
          Automatically run segmentation on the currently loaded image.
        </div>

        <v-btn
          color="primary"
          size="x-large"
          block
          @click="doSegmentWithVista3D"
          :loading="segmentWithVista3DLoading"
          :disabled="!ready || !hasCurrentImage"
          class="mb-3"
        >
          <v-icon left>mdi-play</v-icon>
          {{ segmentWithVista3DLoading ? 'Running Segmentation...' : 'Run Segmentation' }}
        </v-btn>

        <div class="text-center text-caption" v-if="!hasCurrentImage">
          Load an image to enable segmentation.
        </div>
      </v-card-text>
    </v-card>
  </div>
</template>
