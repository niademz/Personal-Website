let models = {
  bow: null,
  semantic: null
};
let vocab, wordToIndex, gloveMini;

export async function loadModels() {
  // Load both models in parallel
  const [bowData, semanticData, vocabData, gloveData] = await Promise.all([
    fetch('/models/bow_model.json').then(r => r.json()),
    fetch('/models/w2v_model.json').then(r => r.json()),
    fetch('/data/vocab.json').then(r => r.json()),
    fetch('/models/glove-mini.json').then(r => r.json())
  ]);

  models.bow = bowData;
  models.semantic = semanticData;
  vocab = vocabData;
  wordToIndex = Object.fromEntries(vocab.map((w, i) => [w, i]));
  gloveMini = gloveData;
}


// 2) Sigmoid activation
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// 3) Inference function
export function predictEmotionScoresBoW(rawText) {
  const model = models.bow;
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

export function predictEmotionScoresSemantic(rawText) {
  const model = models.semantic;
  // Clean and tokenize
  const cleaned = rawText
    .toLowerCase()
    .replace(/[^a-zA-Z\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  const tokens = cleaned.split(' ');

  // Compute TF (term frequency)
  const tf = {};
  for (const token of tokens) {
    if (gloveMini[token]) {
      tf[token] = (tf[token] || 0) + 1;
    }
  }

  // Compute vector sum with pseudo IDF weighting
  const vectorSum = Array(model.vocabSize).fill(0);
  let totalWeight = 0;
  const N = 54; // from Wild Iris
  const dfGuess = 5; // estimate: average word appears in ~5 poems
  const idfBase = Math.log(N / dfGuess); // ~2.3

  for (const [word, freq] of Object.entries(tf)) {
    const tfidf = freq * idfBase;
    const vec = gloveMini[word];

    for (let i = 0; i < model.vocabSize; i++) {
      vectorSum[i] += vec[i] * tfidf;
    }

    totalWeight += tfidf;
  }

  const inputVec = totalWeight === 0
    ? vectorSum
    : vectorSum.map(v => v / totalWeight);

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

  // Denormalize 0–1 → 0–10
  console.log(finalOutput.map(v => v * 10));
  return finalOutput.map(v => v * 10);
}
