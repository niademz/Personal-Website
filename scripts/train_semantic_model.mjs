import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { cleanText } from './cleanText.mjs';

// === CONFIG ===
const GLOVE_FILE = path.join('./embeddings', 'glove.6B.100d.txt');
const EMBEDDING_DIM = 100;
const NUM_POEMS = 54;
const DATA_DIR = './data';

// === STEP 1: Load GloVe Embeddings ===
const gloveEmbeddings = new Map();

console.log('📦 Loading GloVe embeddings...');

const rl = readline.createInterface({
  input: fs.createReadStream(GLOVE_FILE),
  crlfDelay: Infinity
});

for await (const line of rl) {
  const parts = line.split(' ');
  const word = parts[0];
  const vector = parts.slice(1).map(Number);

  if (vector.length === EMBEDDING_DIM) {
    gloveEmbeddings.set(word, vector);
  }
}

console.log(`✅ Loaded ${gloveEmbeddings.size} word vectors from GloVe.`);

// === STEP 2: Load and Clean Poems ===
const allPoems = [];
const termFrequencies = []; // [{ word: count, ... }]
const documentFrequencies = new Map(); // word → numDocsContaining

console.log('📖 Reading and cleaning poems...');

for (let i = 1; i <= NUM_POEMS; i++) {
  const raw = fs.readFileSync(`${DATA_DIR}/poem${i}.txt`, 'utf-8');
  const cleaned = cleanText(raw);
  const tokens = cleaned.split(' ');

  allPoems.push(tokens);

  const tf = {};
  const seen = new Set();

  for (const word of tokens) {
    if (!gloveEmbeddings.has(word)) continue;

    tf[word] = (tf[word] || 0) + 1;

    if (!seen.has(word)) {
      documentFrequencies.set(word, (documentFrequencies.get(word) || 0) + 1);
      seen.add(word);
    }
  }

  termFrequencies.push(tf);
}

console.log(`✅ Loaded and tokenized ${NUM_POEMS} poems.`);

console.log('🧠 Generating poem vectors using TF-IDF + GloVe...');

const poemVectors = [];

for (let i = 0; i < NUM_POEMS; i++) {
  const tf = termFrequencies[i];
  const vectorSum = Array(EMBEDDING_DIM).fill(0);
  let totalWeight = 0;

  for (const [word, count] of Object.entries(tf)) {
    const df = documentFrequencies.get(word) || 1; // prevent div-by-zero
    const idf = Math.log(NUM_POEMS / df);
    const tfidf = count * idf;

    const gloveVec = gloveEmbeddings.get(word);
    if (!gloveVec) continue;

    for (let j = 0; j < EMBEDDING_DIM; j++) {
      vectorSum[j] += gloveVec[j] * tfidf;
    }

    totalWeight += tfidf;
  }

  const averaged = totalWeight === 0
    ? vectorSum // fallback to zero-ish vector if poem had no known words
    : vectorSum.map(v => v / totalWeight);

  poemVectors.push(averaged);
}

console.log('✅ Poem vectors generated!');

// === STEP 3: Load Emotion Scores ===
const SCORE_FILE = path.join(DATA_DIR, 'poem_scores.csv');
const scoreLines = fs.readFileSync(SCORE_FILE, 'utf-8').split('\n').slice(1); // skip header

const emotionTargets = scoreLines.map(line => {
  const [filename, ...scores] = line.split(',');
  return scores.map(s => Number(s.trim()) / 10); // normalize 0–1
});

if (emotionTargets.length !== poemVectors.length) {
  console.error('❌ Mismatch between vectors and scores.');
  process.exit(1);
}

// === STEP 4: Train Semantic Model ===
const inputSize = EMBEDDING_DIM;
const hiddenSize = 32;
const outputSize = 12;
const learningRate = 0.01;
const epochs = 10000;

// initialize weights
function randomMatrix(rows, cols) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => Math.random() * 2 - 1)
  );
}

function randomBias(size) {
  return Array.from({ length: size }, () => Math.random() * 2 - 1);
}

let weights_input_hidden = randomMatrix(inputSize, hiddenSize);
let weights_hidden_output = randomMatrix(hiddenSize, outputSize);
let bias_hidden = randomBias(hiddenSize);
let bias_output = randomBias(outputSize);

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  return x * (1 - x);
}

function mse(predicted, target) {
  return predicted.reduce((sum, p, i) => sum + (p - target[i]) ** 2, 0) / predicted.length;
}

console.log('🎓 Training semantic model...');

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;

  for (let i = 0; i < poemVectors.length; i++) {
    const input = poemVectors[i];
    const target = emotionTargets[i];

    // forward pass
    const hidden_input = Array(hiddenSize).fill(0).map((_, j) =>
      bias_hidden[j] + input.reduce((sum, x_i, k) => sum + x_i * weights_input_hidden[k][j], 0)
    );
    const hidden_output = hidden_input.map(sigmoid);

    const final_input = Array(outputSize).fill(0).map((_, k) =>
      bias_output[k] + hidden_output.reduce((sum, h_j, j) => sum + h_j * weights_hidden_output[j][k], 0)
    );
    const final_output = final_input.map(sigmoid);

    totalLoss += mse(final_output, target);

    // backprop
    const output_errors = final_output.map((out_k, k) => (out_k - target[k]) * sigmoidDerivative(out_k));
    const hidden_errors = hidden_output.map((h_j, j) =>
      sigmoidDerivative(h_j) *
      output_errors.reduce((sum, err_k, k) => sum + err_k * weights_hidden_output[j][k], 0)
    );

    // update weights
    for (let j = 0; j < hiddenSize; j++) {
      for (let k = 0; k < outputSize; k++) {
        weights_hidden_output[j][k] -= learningRate * output_errors[k] * hidden_output[j];
      }
    }

    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        weights_input_hidden[i][j] -= learningRate * hidden_errors[j] * input[i];
      }
    }

    for (let k = 0; k < outputSize; k++) {
      bias_output[k] -= learningRate * output_errors[k];
    }

    for (let j = 0; j < hiddenSize; j++) {
      bias_hidden[j] -= learningRate * hidden_errors[j];
    }
  }

  if (epoch % 100 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${(totalLoss / poemVectors.length).toFixed(4)}`);
  }
}

console.log('✅ Semantic model trained!');

// === STEP 5: Save Model ===
const model = {
  weights_input_hidden,
  weights_hidden_output,
  bias_hidden,
  bias_output,
  hiddenSize,
  outputSize,
  vocabSize: EMBEDDING_DIM
};

fs.writeFileSync('./models/w2v_model.json', JSON.stringify(model, null, 2), 'utf-8');
console.log('✅ Saved model to models/w2v_model.json');

// === STEP 6: Extend Mini GloVe with User Poems ===
const POEMS_JSON = path.join(DATA_DIR, 'poems.json');
const userPoems = JSON.parse(fs.readFileSync(POEMS_JSON, 'utf-8'));

const miniVocab = new Set(documentFrequencies.keys()); // from Wild Iris

for (const { text } of userPoems) {
  const tokens = cleanText(text).split(' ');
  for (const word of tokens) {
    if (gloveEmbeddings.has(word)) {
      miniVocab.add(word);
    }
  }
}

const miniGlove = {};
for (const word of miniVocab) {
  miniGlove[word] = gloveEmbeddings.get(word);
}

fs.writeFileSync('./models/glove-mini.json', JSON.stringify(miniGlove, null, 2), 'utf-8');
console.log(`✅ Saved glove-mini.json with ${miniVocab.size} words.`);
