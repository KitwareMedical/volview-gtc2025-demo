import { defineStore } from 'pinia';
import { reactive, ref } from 'vue';
import { Maybe } from '@/src/types';

/**
 * A cache for vista3d results, which are treated as opaque .nii.gz blobs.
 * The blobs are indexed by the image ID.
 */
export const useVista3dStore = defineStore('vista3d', () => {
  const vista3dIds = ref<string[]>([]);
  const resultsById = reactive<Record<string, Blob>>({});

  /**
   * Sets the vista3d result for a given image ID.
   *
   * This function now accepts raw binary data from the backend (ArrayBuffer,
   * Uint8Array) and automatically converts it to a Blob for consistent storage.
   *
   * @param id The image ID.
   * @param result The .nii.gz data, as a Blob or raw binary buffer.
   */
  function setVista3dResult(id: string, result: Blob | ArrayBuffer | Uint8Array) {
    let dataAsBlob: Blob;

    // Check if the incoming data is already a Blob.
    if (result instanceof Blob) {
      dataAsBlob = result;
    } else {
      // If not, create a new Blob from the raw binary data (ArrayBuffer/Uint8Array).
      dataAsBlob = new Blob([result]);
    }

    if (!(id in resultsById)) {
      vista3dIds.value.push(id);
    }
    // Store the data, now guaranteed to be a Blob.
    resultsById[id] = dataAsBlob;
  }

  /**
   * Retrieves the vista3d blob for a given image ID.
   *
   * @param id The image ID.
   * @returns The blob if it exists, otherwise null.
   */
  function getVista3dResult(id: Maybe<string>): Maybe<Blob> {
    if (!id) return null;
    return resultsById[id] ?? null;
  }

  /**
   * Removes a vista3d result associated with an image ID from the store.
   *
   * @param id The image ID of the result to remove.
   */
  function removeVista3dResult(id: string) {
    if (!(id in resultsById)) return;

    const index = vista3dIds.value.indexOf(id);
    if (index > -1) {
      vista3dIds.value.splice(index, 1);
    }
    delete resultsById[id];
  }

  return {
    vista3dIds,
    setVista3dResult,
    getVista3dResult,
    removeVista3dResult,
  };
});
