import fs from 'fs';
import path from 'path';

// === Load the data ===
const dataset = JSON.parse(fs.readFileSync('./data/processed_dataset.json', 'utf-8'));

// === Network Hyperparameters ===
const inputSize = dataset[0].input.length;
const hiddenSize = 32;
const outputSize = 12;
const learningRate = 0.01;
const epochs = 10000;

for (const { input, output } of dataset) {
    if (input.length !== inputSize) {
      console.error('❌ Inconsistent input size:', input.length, 'Expected:', inputSize);
      process.exit(1);
    }
    if (output.length !== outputSize) {
      console.error('❌ Inconsistent output size:', output.length, 'Expected:', outputSize);
      process.exit(1);
    }
  }

// === Helper functions ===
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  return x * (1 - x);
}

function mse(predicted, target) {
  return predicted.reduce((sum, p, i) => sum + (p - target[i]) ** 2, 0) / predicted.length;
}

// === Initialize weights & biases ===
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

for (let j = 0; j < hiddenSize; j++) {
    if (!Array.isArray(weights_hidden_output[j])) {
      console.error(`❌ weights_hidden_output[${j}] is not an array`);
      process.exit(1);
    }
    for (let k = 0; k < outputSize; k++) {
      if (typeof weights_hidden_output[j][k] !== 'number') {
        console.error(`❌ weights_hidden_output[${j}][${k}] is not a number`);
        console.log(weights_hidden_output[j]);
        process.exit(1);
      }
    }
  }

// === Train the model ===
for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;

  for (const { input, output: target } of dataset) {
    // ---- Forward pass ----
    // hidden layer
const hidden_input = Array(hiddenSize).fill(0).map((_, j) =>
    bias_hidden[j] +
        input.reduce((sum, x_i, i) => sum + x_i * weights_input_hidden[i][j], 0)
    );
    const hidden_output = hidden_input.map(sigmoid);
    
    // output layer
    const final_input = Array(outputSize).fill(0).map((_, k) =>
        bias_output[k] +
        hidden_output.reduce((sum, h_j, j) => sum + h_j * weights_hidden_output[j][k], 0)
    );
    const final_output = final_input.map(sigmoid);

        totalLoss += mse(final_output, target);

    // ---- Backpropagation ----
    const output_errors = final_output.map((out_k, k) => (out_k - target[k]) * sigmoidDerivative(out_k));
    const hidden_errors = hidden_output.map((h_j, j) =>
      sigmoidDerivative(h_j) *
      output_errors.reduce((sum, err_k, k) => sum + err_k * weights_hidden_output[j][k], 0)
    );

    // ---- Update weights hidden → output ----
    for (let j = 0; j < hiddenSize; j++) {
      for (let k = 0; k < outputSize; k++) {
        weights_hidden_output[j][k] -= learningRate * output_errors[k] * hidden_output[j];
      }
    }

    // ---- Update weights input → hidden ----
    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        weights_input_hidden[i][j] -= learningRate * hidden_errors[j] * input[i];
      }
    }

    // ---- Update biases ----
    for (let k = 0; k < outputSize; k++) {
      bias_output[k] -= learningRate * output_errors[k];
    }

    for (let j = 0; j < hiddenSize; j++) {
      bias_hidden[j] -= learningRate * hidden_errors[j];
    }
  }

  if (epoch % 100 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${(totalLoss / dataset.length).toFixed(4)}`);
  }
}

console.log('✅ Training complete!');

const model = {
    weights_input_hidden,
    weights_hidden_output,
    bias_hidden,
    bias_output,
    hiddenSize,
    outputSize,
    vocabSize: inputSize
  };
  
  fs.writeFileSync(
    './models/bow_model.json',
    JSON.stringify(model, null, 2),
    'utf-8'
  );
  
  console.log('✅ Model saved to data/model_weights.json');

