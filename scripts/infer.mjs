import fs from 'fs';
import { cleanText } from './cleanText.mjs';

// 1) Load model:
const {
  weights_input_hidden,
  weights_hidden_output,
  bias_hidden,
  bias_output,
  vocabSize,
  hiddenSize,
  outputSize
} = JSON.parse(fs.readFileSync('./data/model_weights.json', 'utf-8'));

// 2) Load the vocabulary
const vocab = JSON.parse(fs.readFileSync('./data/vocab.json', 'utf-8'));
const wordToIndex = Object.fromEntries(vocab.map((w,i) => [w,i]));

// 3) Activation:
function sigmoid(x) { return 1/(1+Math.exp(-x)); }

// 4) Inference:
export function predictEmotionScores(rawText) {
  // a) Clean + tokenize
  const cleaned = cleanText(rawText);
  const tokens = cleaned.split(' ');

  // b) Build BoW vector
  const inputVec = Array(vocabSize).fill(0);
  for (let t of tokens) {
    if (wordToIndex[t] !== undefined) {
      inputVec[wordToIndex[t]]++;
    }
  }

  // c) Forward pass
  //  hidden layer
  const hiddenInput = Array(hiddenSize).fill(0).map((_, j) =>
    bias_hidden[j] +
      inputVec.reduce((sum, x_i, i) => sum + x_i * weights_input_hidden[i][j], 0)
  );
  const hiddenOutput = hiddenInput.map(sigmoid);

  //  output layer
  const finalInput = Array(outputSize).fill(0).map((_, k) =>
    bias_output[k] +
      hiddenOutput.reduce((sum, h_j, j) => sum + h_j * weights_hidden_output[j][k], 0)
  );
  const finalOutput = finalInput.map(sigmoid);

  // d) Denormalize back to 0–10
  return finalOutput.map(v => v * 10);
}
