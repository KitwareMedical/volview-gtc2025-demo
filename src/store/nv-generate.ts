import { defineStore } from 'pinia';
import { reactive, ref } from 'vue';
import { Maybe } from '@/src/types';

/**
 * A cache for NV-Generate generation results, which are treated as vtk.js image data objects.
 * The objects are indexed by a unique generation ID.
 */
export const useNVGenerateStore = defineStore('nv-generate', () => {
  const nvGenerateIds = ref<string[]>([]);
  const resultsById = reactive<Record<string, any>>({});

  /**
   * Sets the NV-Generate result for a given generation ID.
   *
   * @param id The generation ID.
   * @param result The vtk.js object from the backend.
   */
  function setNVGenerateResult(id: string, result: any) {
    if (!(id in resultsById)) {
      nvGenerateIds.value.push(id);
    }
    resultsById[id] = result;
  }

  /**
   * Retrieves the NV-Generate result for a given generation ID.
   */
  function getNVGenerateResult(id: Maybe<string>): Maybe<any> {
    if (!id) return null;
    return resultsById[id] ?? null;
  }

  /**
   * Removes a NV-Generate result associated with a generation ID from the store.
   *
   * @param id The generation ID of the result to remove.
   */
  function removeNVGenerateResult(id: string) {
    if (!(id in resultsById)) return;

    const index = nvGenerateIds.value.indexOf(id);
    if (index > -1) {
      nvGenerateIds.value.splice(index, 1);
    }
    delete resultsById[id];
  }

  return {
    nvGenerateIds,
    setNVGenerateResult,
    getNVGenerateResult,
    removeNVGenerateResult,
  };
});
