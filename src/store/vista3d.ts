import { defineStore } from 'pinia';
import { reactive, ref } from 'vue';
import { Maybe } from '@/src/types';

/**
 * A cache for vista3d results, which are treated as vtk.js image data objects.
 * The objects are indexed by the image ID.
 */
export const useVista3dStore = defineStore('vista3d', () => {
  const vista3dIds = ref<string[]>([]);
  const resultsById = reactive<Record<string, any>>({});

  /**
   * Sets the vista3d result for a given image ID.
   *
   * @param id The image ID.
   * @param result The vtk.js object from the backend.
   */
  function setVista3dResult(id: string, result: any) {
    if (!(id in resultsById)) {
      vista3dIds.value.push(id);
    }
    resultsById[id] = result;
  }

  /**
   * Retrieves the vista3d JSON string for a given image ID.
   */
  function getVista3dResult(id: Maybe<string>): Maybe<any> {
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
