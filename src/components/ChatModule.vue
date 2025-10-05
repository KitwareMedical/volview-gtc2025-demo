<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import { storeToRefs } from 'pinia';
import MarkdownIt from 'markdown-it';

import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { usePatientStore } from '../store/patient-store';
import { useBackendModelStore } from '../store/backend-model-store';
import { useServerStore, ConnectionState } from '@/src/store/server';
import useViewSliceStore from '@/src/store/view-configs/slicing';

// --- Configuration ---
/** Identifier for the view whose slice we are interested in. */
const TARGET_VIEW_ID = 'Axial';

/** List of vital sign fields to extract from the backend model store. */
const VITAL_FIELDS = ['heart_rate', 'respiratory_rate', 'spo2'] as const;

/** Available backend models for selection. */
const AVAILABLE_MODELS = ['MedGemma', 'Clara NV-Reason-CXR-3B'] as const;
type ModelName = typeof AVAILABLE_MODELS[number];

// --- Store and Composables Setup ---
const patientStore = usePatientStore();
const backendModelStore = useBackendModelStore();
const serverStore = useServerStore();
const viewSliceStore = useViewSliceStore();
const md = new MarkdownIt({ breaks: true });

const { selectedPatient, vitals } = storeToRefs(patientStore);
const { selectedModel } = storeToRefs(backendModelStore);
const { client } = serverStore;
const { currentImageID } = useCurrentImage();

// --- Component State ---
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

/** Stores chat histories for all models, keyed by model name. */
const chatHistories = ref<Record<ModelName, Message[]>>({} as Record<ModelName, Message[]>);
const newMessage = ref('');
const isTyping = ref(false);

// --- Computed Properties ---
const isConnected = computed(() => serverStore.connState === ConnectionState.Connected);

/** The current slice being viewed in the target 2D view. */
const currentSlice = computed(() => {
  if (!currentImageID.value) return null;
  const config = viewSliceStore.getConfig(TARGET_VIEW_ID, currentImageID.value);
  return config?.slice ?? null;
});

/**
 * Dynamically returns the message history for the currently selected model.
 */
const currentMessages = computed(() => {
    const model = selectedModel.value as ModelName;
    return chatHistories.value[model] ?? [];
});

// --- Watchers ---

/**
 * Reset all chat histories when the selected patient changes.
 */
watch(selectedPatient, () => {
    resetAllChats();
});


// --- Utility Functions ---

/**
 * Selects a model and sets it in the backend store.
 * @param model The name of the model to select.
 */
const selectModel = (model: ModelName) => {
  backendModelStore.setModel(model);
};

/**
 * A convenience function to reset all chat histories.
 */
const resetAllChats = () => {
  chatHistories.value = {} as Record<ModelName, Message[]>;
};

/**
 * Extracts the numerical value from a list of FHIR Observation resources for a given vital field.
 * @param field The vital field name (e.g., 'heart_rate').
 * @returns An array of numerical vital sign values.
 */
function extractVitals(field: typeof VITAL_FIELDS[number]): (number | undefined)[] {
  const observations = vitals.value[field];
  return (
    observations
      ?.map((obs) => obs?.valueQuantity?.value)
      .filter((v): v is number | undefined => v != null) ?? []
  );
}

/**
 * Appends a new message to the chat log for the currently active model.
 */
const appendMessage = (text: string, sender: 'user' | 'bot') => {
  const model = selectedModel.value as ModelName;
  // Initialize history for the model if it doesn't exist
  if (!chatHistories.value[model]) {
    chatHistories.value[model] = [];
  }
  chatHistories.value[model].push({ id: Date.now(), text, sender });
};

// --- Main Method ---
const sendMessage = async () => {
  const text = newMessage.value.trim();
  if (!text || isTyping.value) return;

  // Initial setup and validation
  const patientID = selectedPatient.value?.id;
  if (!patientID) {
    console.error('No patient is selected.');
    return;
  }

  appendMessage(text, 'user');
  newMessage.value = '';
  isTyping.value = true;

  try {
    // Dynamically build the vital signs part of the payload
    const vitalPayload = VITAL_FIELDS.reduce(
      (acc, field) => {
        acc[field] = extractVitals(field);
        return acc;
      },
      {} as Record<typeof VITAL_FIELDS[number], (number | undefined)[]>
    );

    // Define the complete payload
    const payload = {
      prompt: text,
      ...vitalPayload,
    };

    backendModelStore.setAnalysisInput(patientID, payload);

    await client.call('multimodalLlmAnalysis', [
      patientID,
      currentImageID.value ?? null,
      currentSlice.value,
    ]);

    // Get the data outputs from the store
    const botResponse = backendModelStore.analysisOutput[patientID];

    if (!botResponse || typeof botResponse !== 'string') {
      throw new Error('Received an invalid or malformed response from the server.');
    }

    appendMessage(botResponse, 'bot');
  } catch (error) {
    console.error('Error calling multimodalLlmAnalysis:', error);
    const errorMessage = (error as Error).message.includes('No patient')
      ? 'Please select a patient before chatting.'
      : 'Sorry, an error occurred while processing your request. Please try again.';
    appendMessage(errorMessage, 'bot');
  } finally {
    isTyping.value = false;
  }
};
</script>

<template>
  <v-container fluid class="fill-height pa-0">
    <v-card v-if="selectedPatient" class="chat-card">
      <v-card-title class="d-flex align-center py-2">
        <span class="text-subtitle-1">AI Assistant</span>
        <v-spacer></v-spacer>

        <v-menu offset-y>
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              variant="tonal"
              color="primary"
              size="small"
              class="mr-2"
            >
              {{ selectedModel }}
              <v-icon end>mdi-menu-down</v-icon>
            </v-btn>
          </template>
          <v-list dense>
            <v-list-item
              v-for="model in AVAILABLE_MODELS"
              :key="model"
              @click="selectModel(model)"
              :active="model === selectedModel"
            >
              <v-list-item-title>{{ model }}</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-menu>

        <v-btn
            icon="mdi-delete-sweep"
            variant="text"
            @click="resetAllChats"
            size="small"
        >
          <v-icon>mdi-delete-sweep</v-icon>
          <v-tooltip activator="parent" location="bottom">Clear All Chats</v-tooltip>
        </v-btn>
      </v-card-title>

      <v-divider />

      <v-card-text class="chat-log">
        <div
          v-for="message in currentMessages"
          :key="message.id"
          :class="['d-flex', message.sender === 'user' ? 'justify-end' : 'justify-start']"
          class="mb-4"
        >
          <div :class="['message-bubble', `message-${message.sender}`]">
            <div v-if="message.sender === 'bot'" v-html="md.render(message.text)"></div>
            <div v-else class="message-text-user">{{ message.text }}</div>
          </div>
        </div>
      </v-card-text>

      <v-progress-linear
        v-if="isTyping"
        indeterminate
        color="primary"
      ></v-progress-linear>

      <v-card-actions class="pa-4">
        <v-text-field
          v-model="newMessage"
          @keydown.enter="sendMessage"
          @keydown.stop
          :disabled="isTyping"
          label="Type your message..."
          variant="solo"
          hide-details
          clearable
          rounded
        >
          <template #append-inner>
            <v-btn
              @click="sendMessage"
              :disabled="isTyping || !newMessage"
              icon="mdi-send"
              variant="text"
              color="primary"
            ></v-btn>
          </template>
        </v-text-field>
      </v-card-actions>
    </v-card>

    <v-alert
      v-else
      type="info"
      variant="tonal"
      border="start"
      icon="mdi-account-search-outline"
    >
      Please select a patient to begin a chat session.
    </v-alert>
  </v-container>
</template>

<style scoped>
.chat-card {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
}

.chat-log {
  flex-grow: 1;
  overflow-y: auto;
  padding: 16px;
}

.message-bubble {
  padding: 10px 16px;
  border-radius: 18px;
  max-width: 70%;
  line-height: 1.5;
  word-wrap: break-word;
}

.message-user {
  background-color: rgb(var(--v-theme-primary));
  color: rgb(var(--v-theme-on-primary));
  border-bottom-right-radius: 4px;
}

.message-bot {
  background-color: rgb(var(--v-theme-surface-variant));
  color: rgb(var(--v-theme-on-surface-variant));
  border-bottom-left-radius: 4px;
}

.message-text-user {
  /* This ensures user-typed newlines are respected */
  white-space: pre-wrap;
}

/* Styles for rendered markdown content from the bot.
   ':deep()' is used to apply styles to the v-html content,
   which is not processed by Vue's scoped styles otherwise.
*/
.message-bot :deep(p) {
  margin-bottom: 0.5em;
}
.message-bot :deep(p:last-child) {
  margin-bottom: 0;
}
.message-bot :deep(ul),
.message-bot :deep(ol) {
  padding-left: 20px;
  margin-bottom: 0.5em;
}
.message-bot :deep(li) {
  margin-bottom: 0.25em;
}
.message-bot :deep(strong) {
  font-weight: 600;
}
</style>
