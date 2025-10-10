<script setup lang="ts">
import { computed, ref } from 'vue';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server';
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
    <v-alert v-if="!ready" color="info">Not connected to the server.</v-alert>
    <v-divider />
    <v-list-subheader>Segment With Clara NV-Curate-CTMR-v2</v-list-subheader> <div>
      <v-row>
        <v-col>
          <v-btn
            @click="doSegmentWithVista3D"
            :loading="segmentWithVista3DLoading"
            :disabled="!ready || !hasCurrentImage"
          >
            Run Segmentation with Clara </v-btn>
          <span v-if="!hasCurrentImage" class="ml-4 body-2">
            No image loaded
          </span>
        </v-col>
      </v-row>
    </div>
  </div>
</template>
