<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import MarkdownIt from 'markdown-it';

import useViewSliceStore from '@/src/store/view-configs/slicing';
import { useCurrentImage } from '@/src/composables/useCurrentImage';
import { useServerStore, ConnectionState } from '@/src/store/server-2';
import { useBackendModelStore } from '../store/backend-model-store';

// --- Configuration ---
const TARGET_VIEW_ID = 'Axial';

/** Suggested prompts for quick access */
const SUGGESTED_PROMPTS = [
  'Find abnormalities and support devices',
  'Examine the chest X-ray',
  'Provide differentials'
] as const;

// --- Store and Composables Setup ---
const backendModelStore = useBackendModelStore();
const serverStore = useServerStore();
const viewSliceStore = useViewSliceStore();
const md = new MarkdownIt({ breaks: true });

const { client } = serverStore;
const { currentImageID } = useCurrentImage();

// Set the model to NV-Reason-CXR-3B when component mounts
onMounted(() => {
  backendModelStore.setModel('Clara NV-Reason-CXR-3B');
});

// --- Component State ---
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

const chatHistory = ref<Message[]>([]);
const newMessage = ref('');
const isTyping = ref(false);
const chatLogRef = ref<HTMLElement | null>(null);

// Watch chat history and scroll after DOM updates
watch(
  chatHistory,
  () => {
    if (chatLogRef.value) {
      const scrollToMax = () => {
        if (chatLogRef.value) {
          chatLogRef.value.scrollTop = chatLogRef.value.scrollHeight;
        }
      };

      // Scroll immediately and after delays to handle markdown rendering
      scrollToMax();
      setTimeout(scrollToMax, 50);
      setTimeout(scrollToMax, 200);
      setTimeout(scrollToMax, 500);
    }
  },
  { deep: true, flush: 'post' }
);

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

// --- Methods ---
const resetAllChats = () => {
  chatHistory.value = [];
};

const appendMessage = (text: string, sender: 'user' | 'bot') => {
  chatHistory.value.push({ id: Date.now(), text, sender });
};

const useSuggestedPrompt = (prompt: string) => {
  newMessage.value = prompt;
};

const sendMessage = async () => {
  if (isInputDisabled.value) return;

  const text = newMessage.value.trim();
  if (!text || !currentImageID.value) return;

  // Convert chat history to the format expected by the backend
  // Map frontend sender ('user' | 'bot') to backend role ('user' | 'assistant')
  // IMPORTANT: Map history BEFORE appending the current message to avoid duplication
  const history = chatHistory.value.map((msg) => ({
    role: msg.sender === 'user' ? 'user' : 'assistant',
    content: msg.text,
  }));

  appendMessage(text, 'user');
  newMessage.value = '';
  isTyping.value = true;

  try {
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
        <!-- Header with Title and Chip -->
        <v-card-title class="d-flex align-center py-3">
          <v-icon class="mr-2">mdi-chat-question</v-icon>
          <span class="text-h6 flex-shrink-0">NV-Reason-CXR-3B</span>
          <v-chip size="small" color="info" variant="outlined" class="ml-3 flex-shrink-0">
            <v-icon start size="small">mdi-clipboard-text-search</v-icon>
            Chest X-Ray Analysis
          </v-chip>
          <v-spacer></v-spacer>
          <v-btn
            icon="mdi-delete-sweep"
            variant="text"
            @click="resetAllChats"
            size="small"
          >
            <v-tooltip activator="parent" location="bottom">
              Clear Chat
            </v-tooltip>
          </v-btn>
        </v-card-title>

        <!-- Model Description -->
        <v-card-text class="text-body-2 pt-0 pb-3">
          A specialized vision-language model for medical reasoning and interpretation of chest X-ray images.
          Combines visual understanding with medical reasoning to provide comprehensive analysis and detailed
          explanations.
        </v-card-text>

        <v-divider />
      </div>

      <div ref="chatLogRef" class="chat-log flex-grow-1">
        <div
          v-for="message in chatHistory"
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
      </div>

      <div class="flex-shrink-0">
        <v-progress-linear
          v-if="isTyping"
          indeterminate
          color="primary"
        ></v-progress-linear>

        <v-divider />

        <v-card-text class="px-4 py-2">
          <div class="text-caption text-medium-emphasis mb-1">Suggested prompts:</div>
          <div class="d-flex flex-wrap ga-1">
            <v-chip
              v-for="(prompt, index) in SUGGESTED_PROMPTS"
              :key="index"
              @click="useSuggestedPrompt(prompt)"
              variant="outlined"
              size="x-small"
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
  max-width: 85%;
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
