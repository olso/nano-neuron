// formular for converting Fahrenheit to Celsius
// Tf = Tc * 1.8 + 32
//    =      W     B
const W = 1.8;
const B = 32;
const celsiusToFahrenheit = (c: number) => c * W + B;

const generateDataSets = () => {
  // xTrain -> [0, 1, 2, ...] -> celsius
  // yTrain -> [32, 33.8, 35.6, ...] -> fahrenheit

  const xTrain: number[] = []; // celsius
  const yTrain: number[] = []; // fahrenheit

  for (let x = 0; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTrain.push(x);
    yTrain.push(y);
  }

  // xTest -> [0.5, 1.5, 2.5, ...] -> celsius
  // yTest -> [32.9, 34.7, 36.5, ...] -> fahrenheit

  const xTest: number[] = []; // celsius
  const yTest: number[] = []; // fahrenheit

  // We make sure test set has different (but still correct) data

  for (let x = 0.5; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTest.push(x);
    yTest.push(y);
  }

  return {
    xTrain,
    xTest,
    yTrain,
    yTest
  };
};

/**
 * @tutorial https://github.com/trekhleb/nano-neuron
 *
 * Ultimately we want to teach our NanoNeuron to imitate celsiusToFahrenheit function
 * (to learn that W = 1.8 and B = 32) without knowing these parameters in advance.
 */
class NanoNeuron {
  w: number; // w state
  b: number; // b state

  constructor(w: number, b: number) {
    this.w = w;
    this.b = b;
  }

  /**
   * @tutorial https://en.wikipedia.org/wiki/Linear_regression
   * @tutorial https://simple.wikipedia.org/wiki/Linear_regression
   *
   * Linear regression is a way to explain the relationship
   * between a dependent variable and one or more explanatory variables
   * using a straight line
   *
   * @param x our tunable param (celsius)
   *
   * @returns y our predicted fahrenheit
   */
  predict(x: number) {
    return x * this.w + this.b;
  }
}

/**
 * The calculation of the cost (the mistake) between the correct output value of y and prediction
 *
 * if y=33.8 and yPredicted=33.95 then cost=0.0128 -> low = good
 * if y=33.8 and yPredicted=459.1; then cost=90 440 -> high = bad
 *
 * @param y fahrenheit
 * @param yPredicted fahrenheit
 */
function predictionCost(y: number, yPredicted: number) {
  return (y - yPredicted) ** 2 / 2;
}

/**
 * average cost of predictions along the way
 */
function forwardPropagation(
  model: NanoNeuron,
  xTrain: number[], // celsius
  yTrain: number[] // fahrenheit
) {
  const m = xTrain.length;
  const predictions: number[] = [];

  let cost = 0;

  for (let i = 0; i < m; i += 1) {
    const yPredicted = model.predict(xTrain[i]);

    cost += predictionCost(yTrain[i], yPredicted);
    predictions.push(yPredicted);
  }

  const averageCost = cost / m;

  return {
    predictions,
    averageCost
  };
}

/**
 * @tutorial https://github.com/trekhleb/nano-neuron#backward-propagation
 *
 * process of evaluating cost of prediction
 *
 * back and forth
 *
 */
function backwardPropagation(
  predictions: number[],
  xTrain: number[], // celsius
  yTrain: number[] // fahrenheit
) {
  const m = xTrain.length;

  // At the beginning we don't know in which way our parameters 'w' and 'b' need to be changed.
  // Therefore we're setting up the changing steps for each parameters to 0.
  let deltaW = 0;
  let deltaB = 0;

  for (let i = 0; i < m; i += 1) {
    deltaW += (yTrain[i] - predictions[i]) * xTrain[i];
    deltaB += yTrain[i] - predictions[i];
  }

  const averageDeltaW = deltaW / m;
  const averageDeltaB = deltaB / m;

  return {
    averageDeltaB,
    averageDeltaW
  };
}

/**
 * @param model NanoNeuron
 * @param epochs how much time we span to train
 * @param alpha learning rate; multiplier for for deltaW/deltaB
 * @param xTrain celsius[]
 * @param yTrain fahrenheit[]
 */
function trainModel(
  model: NanoNeuron,
  epochs: number,
  alpha: number,
  xTrain: number[],
  yTrain: number[]
) {
  const costHistory = []; // how our model learns

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    //
    const { averageCost, predictions } = forwardPropagation(
      model,
      xTrain,
      yTrain
    );

    costHistory.push(averageCost);

    //
    const { averageDeltaB, averageDeltaW } = backwardPropagation(
      predictions,
      xTrain,
      yTrain
    );

    model.w += averageDeltaW * alpha;
    model.b += averageDeltaB * alpha;
  }

  return costHistory;
}

const randomW = Math.random(); // i.e. -> 0.9492
const randomB = Math.random(); // i.e. -> 0.4570

console.log({ randomW, randomB });

const nanoNeuron = new NanoNeuron(randomW, randomB);

const { xTrain, yTrain, xTest, yTest } = generateDataSets();

// const costHistory = trainModel(nanoNeuron, 5, 0.0005, xTrain, yTrain);
// console.log(costHistory);
const costHistory = trainModel(nanoNeuron, 70000, 0.0005, xTrain, yTrain);

console.log(costHistory[0], costHistory[costHistory.length - 1]);

const celsius = 30;
console.log("our model thinks that ", celsius, "C is ", nanoNeuron.predict(celsius), "F");
console.log(celsius, "C is actually ", celsiusToFahrenheit(celsius), "F")
