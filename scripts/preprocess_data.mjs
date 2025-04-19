import fs from 'fs';
import path from 'path';
import { cleanText } from './cleanText.mjs';

// === CONFIG ===
const NUM_POEMS = 20;
const DATA_DIR = './data';
const SCORE_FILE = path.join(DATA_DIR, 'poem_scores.csv');

// === STEP 1: Read & Clean Poems ===
const poems = [];
for (let i = 1; i <= NUM_POEMS; i++) {
  const filePath = path.join(DATA_DIR, `poem${i}.txt`);
  const rawText = fs.readFileSync(filePath, 'utf-8');
  const cleaned = cleanText(rawText);
  poems.push(cleaned);
}

// === STEP 2: Build Vocabulary ===
const vocabSet = new Set();
poems.forEach(poem => {
  poem.split(' ').forEach(word => vocabSet.add(word));
});
const vocab = Array.from(vocabSet);
const wordToIndex = Object.fromEntries(vocab.map((word, i) => [word, i]));

// === STEP 3: Convert Poems to Bag-of-Words Vectors ===
const poemVectors = poems.map(poem => {
  const vector = Array(vocab.length).fill(0);
  poem.split(' ').forEach(word => {
    const index = wordToIndex[word];
    if (index !== undefined) {
      vector[index]++;
    }
  });
  return vector;
});

// === STEP 4: Load Scores from CSV ===
const scoreLines = fs.readFileSync(SCORE_FILE, 'utf-8').split('\n').slice(1);
const emotionTargets = scoreLines.map(line => {
  const [filename, ...scores] = line.split(',');
  const cleanScores = scores
    .filter(score => score.trim() !== '')  // remove empty strings
    .map(Number)
    .map(s => s/10);
  return cleanScores;
});

// === Export the Dataset ===
const dataset = poemVectors.map((vector, i) => ({
  input: vector,
  output: emotionTargets[i]
}));

// Save to JSON for training
fs.writeFileSync('./data/processed_dataset.json', JSON.stringify(dataset, null, 2));
console.log('✅ Preprocessing complete. Dataset saved to data/processed_dataset.json');

fs.writeFileSync(
  path.join(DATA_DIR, 'vocab.json'),
  JSON.stringify(vocab, null, 2),
  'utf-8'
);
console.log('✅ Vocabulary saved to data/vocab.json');