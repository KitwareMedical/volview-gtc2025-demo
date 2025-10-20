<script setup lang="ts">
import { ref, computed } from 'vue';
import { storeToRefs } from 'pinia';
import MarkdownIt from 'markdown-it';

import useViewSliceStore from '@/src/store/view-configs/slicing';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server-2';
import { useBackendModelStore } from '../store/backend-model-store';
import NVIDIAModelCard from './NVIDIAModelCard.vue';

// --- Configuration ---
const TARGET_VIEW_ID = 'Axial';
const AVAILABLE_MODELS = ['Clara NV-Reason-CXR-3B'] as const;
type ModelName = (typeof AVAILABLE_MODELS)[number];

/** Suggested prompts for quick access */
const SUGGESTED_PROMPTS = [
  'Provide a comprehensive image analysis, and list all abnormalities.',
  'Provide differentials',
  'Write a structured report',
] as const;


// --- Store and Composables Setup ---
const backendModelStore = useBackendModelStore();
const serverStore = useServerStore();
const viewSliceStore = useViewSliceStore();
const md = new MarkdownIt({ breaks: true });

const { selectedModel } = storeToRefs(backendModelStore);
const { client } = serverStore;
const { currentImageID } = useCurrentImage();

// --- Component State ---
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

// Placeholder Model Card Data
const mistyCard = {
  modelName: 'Clara NV-Reason-CXR-3B (Misty)',
  subtitle: 'Conversational AI for Medical Imaging Analysis',
  icon: 'mdi-chat-question',
  chips: [
    { text: 'Multimodal LLM', icon: 'mdi-text-box-search', color: 'info' },
    { text: 'Placeholder', icon: 'mdi-flask', color: 'warning' },
  ],
  description:
    'This is a placeholder description for the Misty model. It is designed to understand and respond to queries about medical images.',
  details: [
    { key: 'Architecture', value: 'Placeholder Architecture' },
    { key: 'Model Version', value: '0.1.0 (Placeholder)' },
    { key: 'Intended Use', value: 'Radiology report generation and analysis' },
  ],
};

const chatHistories = ref<Record<ModelName, Message[]>>(
  {} as Record<ModelName, Message[]>
);
const newMessage = ref('');
const isTyping = ref(false);

// --- Computed Properties ---
const isConnected = computed(
  () => serverStore.connState === ConnectionState.Connected
);
const hasCurrentImage = computed(() => !!currentImageID.value);

const isInputDisabled = computed(
  () => isTyping.value || !isConnected.value || !hasCurrentImage.value
);

const inputPlaceholder = computed(() => {
  if (!isConnected.value) return 'Not connected to server';
  if (!hasCurrentImage.value) return 'Load an image to start chatting';
  return 'Type your message...';
});

const currentSlice = computed(() => {
  if (!currentImageID.value) return null;
  const config = viewSliceStore.getConfig(TARGET_VIEW_ID, currentImageID.value);
  return config?.slice ?? null;
});

const currentMessages = computed(() => {
  const model = selectedModel.value as ModelName;
  return chatHistories.value[model] ?? [];
});

// --- Methods ---
const selectModel = (model: ModelName) => {
  backendModelStore.setModel(model);
};

const resetAllChats = () => {
  chatHistories.value = {} as Record<ModelName, Message[]>;
};

const appendMessage = (text: string, sender: 'user' | 'bot') => {
  const model = selectedModel.value as ModelName;
  if (!chatHistories.value[model]) {
    chatHistories.value[model] = [];
  }
  chatHistories.value[model].push({ id: Date.now(), text, sender });
};

/**
 * Sets the input field to the selected suggested prompt.
 */
const useSuggestedPrompt = (prompt: string) => {
  newMessage.value = prompt;
};

const sendMessage = async () => {
  if (isInputDisabled.value) return;

  const text = newMessage.value.trim();
  if (!text || !currentImageID.value) return;

  appendMessage(text, 'user');
  newMessage.value = '';
  isTyping.value = true;

  try {
    // Convert chat history to the format expected by the backend
    // Format: [{ role: 'user' | 'assistant', content: string }, ...]
    const history = currentMessages.value.map((msg) => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.text,
    }));

    // Define the complete payload with prompt and history
    const payload = {
      prompt: text,
      history,
    };

    backendModelStore.setAnalysisInput(currentImageID.value, payload);

    await client.call('multimodalLlmAnalysis', [
      currentImageID.value,
      currentSlice.value,
    ]);

    const botResponse = backendModelStore.analysisOutput[currentImageID.value];
    if (typeof botResponse !== 'string') {
      throw new Error('Received an invalid response from the server.');
    }
    appendMessage(botResponse, 'bot');
  } catch (error) {
    console.error('Error calling multimodalLlmAnalysis:', error);
    appendMessage('Sorry, an error occurred. Please try again.', 'bot');
  } finally {
    isTyping.value = false;
  }
};
</script>

<template>
  <div class="fill-height d-flex flex-column">
    <v-alert v-if="!isConnected" type="info" variant="tonal" class="ma-2 flex-shrink-0">
      Not connected to the server.
    </v-alert>

    <v-card class="ma-2 d-flex flex-column flex-grow-1 overflow-hidden">
      <div class="flex-shrink-0">
        <v-card-title class="d-flex align-center py-2">
          <span class="text-subtitle-1">AI Assistant</span>
          <v-spacer></v-spacer>

          <v-menu offset-y>
            <template #activator="{ props }">
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
            <v-tooltip activator="parent" location="bottom">
              Clear All Chats
            </v-tooltip>
          </v-btn>
        </v-card-title>
        <v-divider />

        <NVIDIAModelCard
          v-if="selectedModel === 'Clara NV-Reason-CXR-3B'"
          :model-name="mistyCard.modelName"
          :subtitle="mistyCard.subtitle"
          :icon="mistyCard.icon"
          :chips="mistyCard.chips"
          :description="mistyCard.description"
          :details="mistyCard.details"
          class="ma-2"
        />
      </div>

      <v-card-text class="chat-log flex-grow-1">
        <div
          v-for="message in currentMessages"
          :key="message.id"
          :class="[
            'd-flex',
            message.sender === 'user' ? 'justify-end' : 'justify-start',
          ]"
          class="mb-4"
        >
          <div :class="['message-bubble', `message-${message.sender}`]">
            <div
              v-if="message.sender === 'bot'"
              v-html="md.render(message.text)"
            ></div>
            <div v-else class="message-text-user">{{ message.text }}</div>
          </div>
        </div>
      </v-card-text>

      <div class="flex-shrink-0">
        <v-progress-linear
          v-if="isTyping"
          indeterminate
          color="primary"
        ></v-progress-linear>

        <v-card-text class="pa-3">
          <div class="text-caption text-medium-emphasis mb-2">Suggested prompts:</div>
          <div class="d-flex flex-wrap ga-2">
            <v-chip
              v-for="(prompt, index) in SUGGESTED_PROMPTS"
              :key="index"
              @click="useSuggestedPrompt(prompt)"
              variant="outlined"
              size="small"
              class="suggested-prompt-chip"
            >
              {{ prompt }}
            </v-chip>
          </div>
        </v-card-text>

        <v-card-actions class="pa-4">
          <v-text-field
            v-model="newMessage"
            @keydown.enter.prevent="sendMessage"
            @keydown.stop
            :disabled="isInputDisabled"
            :label="inputPlaceholder"
            variant="solo"
            hide-details
            clearable
            rounded
          >
            <template #append-inner>
              <v-btn
                @click="sendMessage"
                :disabled="isInputDisabled || !newMessage"
                icon="mdi-send"
                variant="text"
                color="primary"
              ></v-btn>
            </template>
          </v-text-field>
        </v-card-actions>
      </div>
    </v-card>
  </div>
</template>

<style scoped>
.chat-log {
  overflow-y: auto;
  padding: 16px;
}

.message-bubble {
  padding: 10px 16px;
  border-radius: 18px;
  max-width: 85%; /* Adjusted for potentially narrower sidebars */
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
  white-space: pre-wrap;
}

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

.suggested-prompt-chip {
  cursor: pointer;
  transition: all 0.2s ease;
}

.suggested-prompt-chip:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>
