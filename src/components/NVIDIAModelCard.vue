<script setup lang="ts">
import { PropType } from 'vue';

interface Chip {
  text: string;
  icon: string;
  color: string;
}

interface DetailItem {
  key: string;
  value: string;
}

interface Reference {
  text: string;
  url: string;
}

defineProps({
  modelName: { type: String, required: true },
  subtitle: { type: String, required: true },
  icon: { type: String, default: 'mdi-brain' },
  chips: { type: Array as PropType<Chip[]>, default: () => [] },
  description: { type: String, default: '' },
  details: { type: Array as PropType<DetailItem[]>, default: () => [] },
  references: { type: Array as PropType<Reference[]>, default: () => [] },
});
</script>

<template>
  <div class="overflow-y-auto overflow-x-hidden ma-2 fill-height">
    <v-card class="mb-4" elevation="2">
      <v-card-title class="d-flex align-center">
        <v-icon class="mr-2" color="primary">{{ icon }}</v-icon>
        <span class="text-h6">{{ modelName }}</span>
      </v-card-title>
      <v-card-subtitle class="text-caption">{{ subtitle }}</v-card-subtitle>
      <v-card-text>
        <v-chip
          v-for="(chip, index) in chips"
          :key="index"
          size="small"
          :color="chip.color"
          variant="outlined"
          class="mr-2"
        >
          <v-icon start size="small">{{ chip.icon }}</v-icon>
          {{ chip.text }}
        </v-chip>
      </v-card-text>
    </v-card>

    <v-card v-if="description || details.length" class="mb-4" elevation="2">
      <v-card-title class="text-subtitle-1">Model Overview</v-card-title>
      <v-card-text>
        <p v-if="description" class="text-body-2 mb-4">{{ description }}</p>
        <v-list lines="one" density="compact">
          <v-list-item v-for="(item, index) in details" :key="index">
            <v-list-item-title class="font-weight-bold">{{ item.key }}</v-list-item-title>
            <v-list-item-subtitle>{{ item.value }}</v-list-item-subtitle>
          </v-list-item>
        </v-list>
      </v-card-text>
    </v-card>

    <v-card v-if="references.length" elevation="2">
       <v-card-title class="text-subtitle-1">References</v-card-title>
      <v-card-text>
        <v-list lines="two" density="compact">
           <v-list-item v-for="(ref, index) in references" :key="index" :href="ref.url" target="_blank">
             <template v-slot:prepend>
                <v-icon color="primary">mdi-book-open-page-variant-outline</v-icon>
             </template>
            <v-list-item-title class="text-body-2 wrap-text">[{{ index + 1 }}] {{ ref.text }}</v-list-item-title>
             <v-list-item-subtitle>Click to view source</v-list-item-subtitle>
          </v-list-item>
        </v-list>
      </v-card-text>
    </v-card>
  </div>
</template>

<style scoped>
.wrap-text {
  white-space: normal;
  word-break: break-word;
}
</style>
