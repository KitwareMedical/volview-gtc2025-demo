import { defineStore } from 'pinia';

/**
 * Interface for the MedGemma store's state.
 */
interface State {
  selectedModel: string; // The currently selected backend model
  analysisIdList: string[]; // list of analysis ids
  analysisInput: Record<string, Record<string, any>>; // inputs are string-to-array-of-numbers dictionaries
  analysisOutput: Record<string, string | null>; // output is a string (e.g., the result of MedGemma)
}

export const useBackendModelStore = defineStore('backend-model-store', {
  /**
   * Defines the initial state of the store.
   */
  state: (): State => ({
    selectedModel: 'medgemma', // Default model on initialization
    analysisIdList: [],
    analysisInput: Object.create(null),
    analysisOutput: Object.create(null),
  }),

  /**
   * Actions to modify the store's state.
   */
  actions: {
    /**
     * Sets the backend model to use for analysis.
     * @param model - The name of the model (e.g., 'medgemma-4b').
     */
    setModel(model: string) {
      this.selectedModel = model;
    },

    addAnalysisId(id: string) {
      if (!this.analysisIdList.includes(id)) {
        this.analysisIdList.push(id);
      }
    },

    /**
     * Sets the input data for the analysis.
     * @param input - A dictionary where keys are strings and values are arrays of numbers.
     */
    setAnalysisInput(id: string, input: Record<string, any>) {
      this.addAnalysisId(id);
      this.analysisInput[id] = input;
    },

    /**
     * Stores the result of the analysis.
     * @param result - A string representing the MedGemma model's output.
     */
    setAnalysisResult(id: string, result: string) {
      this.analysisOutput[id] = result;
    },

    /**
     * Resets the entire analysis state to its initial values.
     */
    clearAnalysis(id: string) {
      if (id in this.analysisInput) {
        this.analysisInput[id] = {};
      }
      if (id in this.analysisOutput) {
        this.analysisOutput[id] = null;
      }
    },
  },
});
