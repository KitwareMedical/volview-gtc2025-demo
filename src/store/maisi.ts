import { defineStore } from 'pinia';
import { reactive, ref } from 'vue';
import { Maybe } from '@/src/types';

/**
 * A cache for MAISI generation results, which are treated as vtk.js image data objects.
 * The objects are indexed by a unique generation ID.
 */
export const useMAISIStore = defineStore('maisi', () => {
  const maisiIds = ref<string[]>([]);
  const resultsById = reactive<Record<string, any>>({});

  /**
   * Sets the MAISI result for a given generation ID.
   *
   * @param id The generation ID.
   * @param result The vtk.js object from the backend.
   */
  function setMAISIResult(id: string, result: any) {
    if (!(id in resultsById)) {
      maisiIds.value.push(id);
    }
    resultsById[id] = result;
  }

  /**
   * Retrieves the MAISI result for a given generation ID.
   */
  function getMAISIResult(id: Maybe<string>): Maybe<any> {
    if (!id) return null;
    return resultsById[id] ?? null;
  }

  /**
   * Removes a MAISI result associated with a generation ID from the store.
   *
   * @param id The generation ID of the result to remove.
   */
  function removeMAISIResult(id: string) {
    if (!(id in resultsById)) return;

    const index = maisiIds.value.indexOf(id);
    if (index > -1) {
      maisiIds.value.splice(index, 1);
    }
    delete resultsById[id];
  }

  return {
    maisiIds,
    setMAISIResult,
    getMAISIResult,
    removeMAISIResult,
  };
});
