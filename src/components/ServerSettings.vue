<script setup lang="ts">
import { computed } from 'vue';
import { storeToRefs } from 'pinia';
import { ConnectionState, useServerStore as useServerStore1 } from '@/src/store/server-1';
import { useServerStore as useServerStore2 } from '@/src/store/server-2';
import { useServerStore as useServerStore3 } from '@/src/store/server-3';

const StatusToText: Record<ConnectionState, string> = {
  [ConnectionState.Connected]: 'Connected',
  [ConnectionState.Disconnected]: 'Disconnected',
  [ConnectionState.Pending]: 'Connecting...',
};

// Create a reactive configuration for each server store
const servers = [useServerStore1, useServerStore2, useServerStore3].map((useStore) => {
  const store = useStore();
  const { connState } = storeToRefs(store);

  return {
    url: computed({
      get: () => store.url,
      set: (url: string) => store.setUrl(url),
    }),
    connState,
    connStatus: computed(() => StatusToText[connState.value]),
    connectBtnText: computed(() =>
      connState.value === ConnectionState.Connected ? 'Disconnect' : 'Connect'
    ),
    toggleConnection: () => {
      if (connState.value === ConnectionState.Connected) {
        store.disconnect();
      } else if (connState.value === ConnectionState.Disconnected) {
        store.connect();
      }
    },
  };
});
</script>

<template>
  <div class="ma-2">
    <div
      v-for="(server, index) in servers"
      :key="index"
      :class="{ 'mt-6': index > 0 }"
    >
      <h3>
        <span>Remote Server {{ index + 1 }}</span>
        <v-btn
          icon="mdi-help-circle"
          variant="flat"
          size="x-small"
          href="https://kitware.github.io/VolView/doc/server.html"
          target="_blank"
          class="ml-1"
          style="color: #777"
          title="Server Documentation"
        ></v-btn>
      </h3>
      <div class="mt-4">
        <v-text-field
          v-model="server.url.value"
          label="Server URL"
          clearable
          persistent-hint
          hint="Make sure you trust the remote server!"
        />
      </div>
      <div class="mt-4">
        <v-btn
          color="secondary"
          @click="server.toggleConnection"
          :loading="server.connState.value === ConnectionState.Pending"
        >
          {{ server.connectBtnText.value }}
        </v-btn>
        <span class="ml-5">Status: {{ server.connStatus.value }}</span>
      </div>
    </div>
  </div>
</template>
