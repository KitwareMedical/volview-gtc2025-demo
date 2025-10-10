<script setup lang="ts">
import { computed, ref } from 'vue';
import { useServerStore, ConnectionState } from '@/src/store/server-3';
import { useMAISIStore } from '@/src/store/maisi';
import { useImageStore } from '@/src/store/datasets-images';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtk from '@kitware/vtk.js/vtk';

const serverStore = useServerStore();
const maisiStore = useMAISIStore();
const imageStore = useImageStore();

const { client } = serverStore;
const ready = computed(
  () => serverStore.connState === ConnectionState.Connected
);

const generationLoading = ref(false);
const magicNumber = ref(42); // Dummy parameter for the UI

const doGenerateWithMAISI = async () => {
  generationLoading.value = true;
  // Create a unique ID for this generation job to retrieve the result later
  const generationId = `maisi-gen-${Date.now()}`;
  try {
    const params = {
      magicNumber: magicNumber.value,
    };
    await client.call('generateWithMAISI', [generationId, params]);
    const generatedImageObject = maisiStore.getMAISIResult(generationId);

    if (!generatedImageObject) {
      console.error(`No MAISI data found for generation ID: ${generationId}`);
      return;
    }

    // Convert the plain JS object from the store into a vtkImageData object
    const generatedImageData = vtk(generatedImageObject) as vtkImageData;

    // Add the generated data as a new image layer
    imageStore.addVTKImageData('MAISI Generated CT', generatedImageData);

    // Clean up the result from the temporary store
    maisiStore.removeMAISIResult(generationId);

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

    <v-card>
      <v-card-title class="text-h6">
        <v-icon class="mr-2">mdi-creation</v-icon>
        MAISI Image Generation
      </v-card-title>
      <v-card-text>
        <div class="text-body-2 mb-4">
          Generate a new synthetic 3D CT scan using the MAISI model.
        </div>

        <v-text-field
          v-model.number="magicNumber"
          label="Magic Number"
          type="number"
          variant="outlined"
          density="compact"
          class="mb-4"
          :disabled="generationLoading"
        />

        <v-btn
          color="primary"
          size="x-large"
          block
          @click="doGenerateWithMAISI"
          :loading="generationLoading"
          :disabled="!ready"
          class="mb-3"
        >
          <v-icon left>mdi-play</v-icon>
          {{ generationLoading ? 'Generating...' : 'Generate CT Scan' }}
        </v-btn>
      </v-card-text>
    </v-card>
  </div>
</template>
