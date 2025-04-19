// scripts/infer-browser.js
let model, vocab, wordToIndex;

// 1) Load everything via fetch
export async function loadModel() {
  model = await fetch('/data/model_weights.json').then(r => r.json());
  vocab = await fetch('/data/vocab.json').then(r => r.json());
  wordToIndex = Object.fromEntries(vocab.map((w,i) => [w,i]));
}

// 2) Sigmoid activation
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// 3) Inference function
export function predictEmotionScores(rawText) {
  // Clean & tokenize (inline cleanText logic or import a browser version)
  const cleaned = rawText
    .toLowerCase()
    .replace(/[^a-zA-Z\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const tokens = cleaned.split(' ');

  // Build BoW vector
  const inputVec = Array(model.vocabSize).fill(0);
  tokens.forEach(t => {
    const idx = wordToIndex[t];
    if (idx != null) inputVec[idx]++;
  });

  // Forward pass → hidden
  const hiddenInput = Array(model.hiddenSize).fill(0).map((_, j) =>
    model.bias_hidden[j] +
      inputVec.reduce((sum, x_i, i) => sum + x_i * model.weights_input_hidden[i][j], 0)
  );
  const hiddenOutput = hiddenInput.map(sigmoid);

  // Forward pass → output
  const finalInput = Array(model.outputSize).fill(0).map((_, k) =>
    model.bias_output[k] +
      hiddenOutput.reduce((sum, h_j, j) => sum + h_j * model.weights_hidden_output[j][k], 0)
  );
  const finalOutput = finalInput.map(sigmoid);

  // Denormalize back to 0–10
  console.log(finalOutput.map(s => s * 10));
  return finalOutput.map(v => v * 10);
}
