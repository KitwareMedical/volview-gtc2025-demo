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
  'Find abnormalities',
  'Examine the chest X-ray',
  'Provide differentials',
  'Write a Structured Report',
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
  thinkContent?: string;
  answerContent?: string;
  isThinkCollapsed?: boolean;
}

const chatHistory = ref<Message[]>([]);
const newMessage = ref('');
const isTyping = ref(false);
const chatLogRef = ref<HTMLElement | null>(null);

// --- Helper Functions ---
const scrollToMax = () => {
  if (chatLogRef.value) {
    chatLogRef.value.scrollTop = chatLogRef.value.scrollHeight;
  }
};

/**
 * Parses a response text to extract <think> and <answer> sections.
 * Handles incomplete tags gracefully during streaming.
 */
const parseThinkAndAnswer = (text: string): {
  thinkContent: string;
  answerContent: string;
  remainingText: string;
} => {
  let thinkContent = '';
  let answerContent = '';
  let remainingText = text;

  // Extract <think> content
  const thinkMatch = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
  if (thinkMatch) {
    thinkContent = thinkMatch[1].trim();
    // Check if tag is closed
    if (text.includes('</think>')) {
      remainingText = text.replace(/<think>[\s\S]*?<\/think>/, '');
    } else {
      // Tag is not closed yet, keep remainder for next update
      remainingText = '';
    }
  }

  // Extract <answer> content from remaining text
  const answerMatch = remainingText.match(/<answer>([\s\S]*?)(?:<\/answer>|$)/);
  if (answerMatch) {
    answerContent = answerMatch[1].trim();
  } else if (remainingText && !remainingText.includes('<think>')) {
    // If no <answer> tag but we have content outside <think>, treat it as answer
    answerContent = remainingText.replace(/<\/?answer>/g, '').trim();
  }

  return { thinkContent, answerContent, remainingText };
};

/**
 * Updates a message with parsed think/answer content
 */
const updateMessageContent = (message: Message) => {
  const parsed = parseThinkAndAnswer(message.text);
  // eslint-disable-next-line no-param-reassign
  message.thinkContent = parsed.thinkContent;
  // eslint-disable-next-line no-param-reassign
  message.answerContent = parsed.answerContent;
};

// --- Watchers ---
// Watch chat history and scroll after DOM updates
watch(
  chatHistory,
  () => {
    // Scroll immediately (after DOM update thanks to flush: 'post')
    // and after delays to handle markdown rendering
    scrollToMax();
    setTimeout(scrollToMax, 50);
    setTimeout(scrollToMax, 200);
    setTimeout(scrollToMax, 500);
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
  return (
    viewSliceStore.getConfig(TARGET_VIEW_ID, currentImageID.value)?.slice ?? null
  );
});

// --- Methods ---
const resetAllChats = () => {
  chatHistory.value = [];
};

const appendMessage = (text: string, sender: 'user' | 'bot') => {
  const message: Message = {
    id: Date.now(),
    text,
    sender,
    isThinkCollapsed: false // Start expanded by default
  };
  if (sender === 'bot') {
    updateMessageContent(message);
  }
  chatHistory.value.push(message);
  return message;
};

const toggleThinkCollapse = (messageId: number) => {
  const message = chatHistory.value.find((m) => m.id === messageId);
  if (message) {
    message.isThinkCollapsed = !message.isThinkCollapsed;
  }
};

const useSuggestedPrompt = (prompt: string) => {
  newMessage.value = prompt;
};

const sendMessage = async () => {
  if (isInputDisabled.value) return;

  const promptText = newMessage.value.trim();
  const imageId = currentImageID.value;
  if (!promptText || !imageId) return;

  // Convert chat history to the format expected by the backend
  // Map frontend sender ('user' | 'bot') to backend role ('user' | 'assistant')
  // IMPORTANT: Map history BEFORE appending the current message to avoid duplication
  const history = chatHistory.value.map(({ sender, text }) => ({
    role: sender === 'user' ? 'user' : 'assistant',
    content: text,
  }));

  appendMessage(promptText, 'user');
  newMessage.value = '';
  isTyping.value = true;

  // Create a placeholder bot message for streaming
  const botMessage = appendMessage('', 'bot');

  try {
    const payload = {
      prompt: promptText,
      history,
    };
    backendModelStore.setAnalysisInput(imageId, payload);

    // Use streaming endpoint
    const botMessageIndex = chatHistory.value.length - 1;

    await client.stream(
      'multimodalLlmAnalysisStream',
      [imageId, currentSlice.value],
      (data: { token: string }) => {
        // Get the current message from the array
        const message = chatHistory.value[botMessageIndex];

        // Append token to text
        message.text += data.token;

        // Update parsed content in real-time
        updateMessageContent(message);

        // Trigger Vue reactivity by creating a new object with all properties preserved
        chatHistory.value[botMessageIndex] = {
          ...message,
          thinkContent: message.thinkContent,
          answerContent: message.answerContent,
        };
      }
    ).catch((streamError) => {
      // Handle stream-specific errors without disconnecting
      console.error('Stream error (handled):', streamError);

      // If we have partial content, keep it; otherwise show error
      if (!botMessage.text || botMessage.text.length === 0) {
        botMessage.text = 'Sorry, streaming was interrupted. Please try again.';
        updateMessageContent(botMessage);
      }

      // Don't re-throw - this prevents disconnection
    });

    // After streaming completes successfully, do a final parse
    const finalMessage = chatHistory.value[botMessageIndex];
    updateMessageContent(finalMessage);

    console.log('Stream completed successfully');
  } catch (error) {
    console.error('Error calling multimodalLlmAnalysisStream:', error);

    // Only show error if we have no content
    if (!botMessage.text || botMessage.text.length === 0) {
      botMessage.text = 'Sorry, an error occurred. Please try again.';
      updateMessageContent(botMessage);
    }
  } finally {
    isTyping.value = false;
  }
};
</script>

<template>
  <div class="fill-height d-flex flex-column">
    <v-card class="ma-2 d-flex flex-column flex-grow-1 overflow-hidden">
      <div class="flex-shrink-0">
        <v-card-title class="d-flex align-center py-3">
          <v-icon class="mr-2">mdi-chat-question</v-icon>
          <span class="text-h6 flex-shrink-0">NV-Reason-CXR-3B</span>
          <v-chip
            size="small"
            color="info"
            variant="outlined"
            class="ml-3 flex-shrink-0"
          >
            <v-icon start size="small">mdi-clipboard-text-search</v-icon>
            Chest X-Ray Analysis
          </v-chip>
          <v-spacer></v-spacer>
          <v-btn
          icon variant="text"
          @click="resetAllChats"
          size="small"
          >
            <v-icon>mdi-delete-sweep</v-icon> <v-tooltip activator="parent" location="bottom">
              Clear Chat
            </v-tooltip>
          </v-btn>
        </v-card-title>

        <v-card-text class="text-body-2 pt-0 pb-3">
          A specialized vision-language model for medical reasoning and
          interpretation of chest X-ray images. Combines visual understanding
          with medical reasoning to provide comprehensive analysis and detailed
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
            <!-- User message: simple text -->
            <div v-if="message.sender === 'user'" class="message-text-user">
              {{ message.text }}
            </div>

            <!-- Bot message: two-section layout with think and answer -->
            <div v-else class="bot-message-content">
              <!-- Thinking Section -->
              <div v-if="message.thinkContent" class="think-section">
                <div
                  class="section-header"
                  @click="toggleThinkCollapse(message.id)"
                  @keydown.enter="toggleThinkCollapse(message.id)"
                  @keydown.space.prevent="toggleThinkCollapse(message.id)"
                  tabindex="0"
                  role="button"
                  :aria-expanded="!message.isThinkCollapsed"
                >
                  <v-icon size="small" class="section-icon">mdi-brain</v-icon>
                  <span class="section-title">Reasoning</span>
                  <v-spacer></v-spacer>
                  <v-btn
                    icon
                    size="x-small"
                    variant="text"
                    class="collapse-btn"
                  >
                    <v-icon size="small">
                      {{ message.isThinkCollapsed ? 'mdi-chevron-down' : 'mdi-chevron-up' }}
                    </v-icon>
                  </v-btn>
                </div>
                <v-expand-transition>
                  <div v-show="!message.isThinkCollapsed" class="section-content">
                    <div v-html="md.render(message.thinkContent)"></div>
                  </div>
                </v-expand-transition>
              </div>

              <!-- Answer Section -->
              <div v-if="message.answerContent" class="answer-section">
                <div class="section-header">
                  <v-icon size="small" class="section-icon">mdi-check-circle</v-icon>
                  <span class="section-title">Analysis</span>
                </div>
                <div class="section-content">
                  <div v-html="md.render(message.answerContent)"></div>
                </div>
              </div>

              <!-- Fallback: show raw text if no sections parsed yet -->
              <div v-if="!message.thinkContent && !message.answerContent && message.text" class="section-content">
                <div v-html="md.render(message.text)"></div>
              </div>
            </div>
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
          <div class="text-caption text-medium-emphasis mb-1">
            Suggested prompts:
          </div>
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
  padding: 12px 16px;
  border-radius: 10px;
  max-width: 85%;
  line-height: 1.5;
  word-wrap: break-word;
  font-size: 0.875em;
}

.message-user {
  background: linear-gradient(135deg, #1e3d5c 0%, #2d4a66 100%);
  color: #b3d4f5;
  border: 1px solid #3d5a7a;
  border-bottom-right-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* Bot message: remove bubble background and padding */
.message-bot {
  background-color: transparent;
  padding: 0;
}

.message-text-user {
  white-space: pre-wrap;
}

/* Bot message content container */
.bot-message-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
}

/* Section base styles */
.think-section,
.answer-section {
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* Thinking section - dark purple/indigo theme */
.think-section {
  border: 1px solid #7e57c2;
  background: linear-gradient(135deg, #2c2540 0%, #3d2f52 100%);
}

/* Answer section - dark teal/cyan theme */
.answer-section {
  border: 1px solid #26a69a;
  background: linear-gradient(135deg, #1e3a38 0%, #2d4a48 100%);
}

/* Section headers */
.section-header {
  display: flex;
  align-items: center;
  padding: 10px 14px;
  font-weight: 600;
  font-size: 0.85em;
  cursor: pointer;
  user-select: none;
  transition: background-color 0.2s ease;
}

.think-section .section-header {
  background: linear-gradient(135deg, #4a3a5c 0%, #5d4a70 100%);
  border-bottom: 1px solid #7e57c2;
  color: #d1c4e9;
}

.think-section .section-header:hover {
  background: linear-gradient(135deg, #564466 0%, #69567a 100%);
}

.think-section .section-header:focus {
  outline: 2px solid #b39ddb;
  outline-offset: 2px;
}

.answer-section .section-header {
  background: linear-gradient(135deg, #2d5450 0%, #3d6460 100%);
  border-bottom: 1px solid #26a69a;
  cursor: default;
  color: #b2dfdb;
}

.section-icon {
  margin-right: 8px;
  opacity: 0.9;
}

.think-section .section-icon {
  color: #b39ddb;
}

.answer-section .section-icon {
  color: #80cbc4;
}

.section-title {
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-size: 0.75em;
}

.collapse-btn {
  opacity: 0.7;
  transition: opacity 0.2s ease;
  color: #d1c4e9;
}

.collapse-btn:hover {
  opacity: 1;
}

/* Section content */
.section-content {
  padding: 14px;
  font-size: 0.95em;
  line-height: 1.6;
}

.think-section .section-content {
  color: #d1c4e9;
  background-color: rgba(0, 0, 0, 0.2);
}

.answer-section .section-content {
  color: #b2dfdb;
  background-color: rgba(0, 0, 0, 0.2);
}

/* Markdown content styles for bot messages */
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
