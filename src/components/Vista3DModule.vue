<script setup lang="ts">
import { computed, ref } from 'vue';
import pako from 'pako';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server';
import { useVista3dStore } from '@/src/store/vista3d';
import { loadFiles } from '@/src/actions/loadUserFiles';

const serverStore = useServerStore();
const vista3dStore = useVista3dStore();
const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

// --- run vista 3d --- //

const segmentWithMONAILoading = ref(false);
const { currentImageID } = useCurrentImage();

const doSegmentWithMONAI = async () => {
  const id = currentImageID.value;
  if (!id) return;

  segmentWithMONAILoading.value = true;
  try {
    await client.call('segmentWithMONAI', [id]);
    const blobResult = vista3dStore.getVista3dResult(id);

    if (!blobResult) {
      console.error(`No vista3d data found for ID: ${id}`);
      return;
    }

    // 1. Get the raw gzipped bytes from the blob
    const compressedBytes = await blobResult.arrayBuffer();

    // 2. Decompress the bytes using pako
    const decompressedBytes = pako.ungzip(new Uint8Array(compressedBytes));

    // 3. Create a new File from the DECOMPRESSED data with a .nii extension
    const segmentationFile = new File([decompressedBytes], 'segmentation.nii', {
      type: 'application/octet-stream', // Use a generic binary type
    });

    // 4. Load the raw .nii file
    await loadFiles([segmentationFile]);

    console.log('Segmentation loading initiated!');

  } catch (error) {
    console.error('An error occurred during segmentation:', error);
  } finally {
    segmentWithMONAILoading.value = false;
  }
};

const hasCurrentImage = computed(() => !!currentImageID.value);
</script>

<template>
  <div class="overflow-y-auto overflow-x-hidden ma-2 fill-height">
    <v-alert v-if="!ready" color="info">Not connected to the server.</v-alert>
    <v-divider />
    <v-list-subheader>Segment With MONAI</v-list-subheader>
    <div>
      <v-row>
        <v-col>
          <v-btn
            @click="doSegmentWithMONAI"
            :loading="segmentWithMONAILoading"
            :disabled="!ready || !hasCurrentImage"
          >
            Run Segment With MONAI
          </v-btn>
          <span v-if="!hasCurrentImage" class="ml-4 body-2">
            No image loaded
          </span>
        </v-col>
      </v-row>
    </div>
  </div>
</template>
