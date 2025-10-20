import { defineStore } from 'pinia';
import { reactive, ref } from 'vue';
import { Maybe } from '@/src/types';

/**
 * A cache for NV-Segment results, which are treated as vtk.js image data objects.
 * The objects are indexed by the image ID.
 */
export const useNVSegmentStore = defineStore('nv-segment', () => {
  const nvSegmentIds = ref<string[]>([]);
  const resultsById = reactive<Record<string, any>>({});

  /**
   * Sets the NV-Segment result for a given image ID.
   *
   * @param id The image ID.
   * @param result The vtk.js object from the backend.
   */
  function setNVSegmentResult(id: string, result: any) {
    if (!(id in resultsById)) {
      nvSegmentIds.value.push(id);
    }
    resultsById[id] = result;
  }

  /**
   * Retrieves the NV-Segment result for a given image ID.
   */
  function getNVSegmentResult(id: Maybe<string>): Maybe<any> {
    if (!id) return null;
    return resultsById[id] ?? null;
  }

  /**
   * Removes an NV-Segment result associated with an image ID from the store.
   *
   * @param id The image ID of the result to remove.
   */
  function removeNVSegmentResult(id: string) {
    if (!(id in resultsById)) return;

    const index = nvSegmentIds.value.indexOf(id);
    if (index > -1) {
      nvSegmentIds.value.splice(index, 1);
    }
    delete resultsById[id];
  }

  return {
    nvSegmentIds,
    setNVSegmentResult,
    getNVSegmentResult,
    removeNVSegmentResult,
  };
});
