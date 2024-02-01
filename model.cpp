#include "model.h"

#include <cassert>
#include <numeric>
#include <cmath>
#include <random>

// ---------------------------- layers ----------------------------

hhLayer::hhLayer(int numNeurons, int numInputs)
{
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;

    activationValue.resize(numNeurons, 0.0f);
    errors.resize(numNeurons, 0.0f);
}

// ---------------------------- Input ----------------------------

hhInputLayer::hhInputLayer(int numNeurons, int numInputs) : hhLayer(numNeurons, numInputs)
{
    biases.resize(numNeurons, 1.0f);
}
    
void hhInputLayer::Forward(const column& input)
{
    activationValue = input;
}

// ---------------------------- Dense ----------------------------


hhDenseLayer::hhDenseLayer(int numNeurons, int numInputs) : hhLayer(numNeurons, numInputs)
{
    weights.resize(numNeurons, column(numInputs));
    biases.resize(numNeurons);
}

void hhDenseLayer::UpdateWeightsAndBiases(const hhLayer& previous, float learningRate)
{
    for (int n = 0; n < numNeurons; n++)
    {
        for (int i = 0; i < numInputs; i++)
        {
            weights[n][i] -= learningRate * previous.activationValue[i] * errors[n];
        }
        biases[n] -= learningRate * errors[n];
    }
}

// ---------------------------- Sigmoid ----------------------------

hhSigmoidLayer::hhSigmoidLayer(int numNeurons, int numInputs) : hhDenseLayer(numNeurons, numInputs)
{
        // Create a random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    // Initialize weights
    for (int i = 0; i < numNeurons; i++) 
    {
        for (int j = 0; j < numInputs; j++) 
        {
            weights[i][j] = distribution(generator);
        }
        biases[i] = distribution(generator);
    }
}

void hhSigmoidLayer::Forward(const column& input)
{
    assert(input.size() == numInputs);
    for (int n = 0; n < numNeurons; n++)
    {
        float raw = std::inner_product(input.begin(), input.end(), weights[n].begin(), 0.0f) + biases[n];
        activationValue[n] = 1.0f / (1.0f + exp(-raw));
    }
}

float hhSigmoidLayer::Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets)
{
    float error = 0.0f;
    for (int n = 0; n < numNeurons; n++)
    {
        const float predicted = activationValue[n];
        const float dp = predicted * (1.0f - predicted);
        if (next == nullptr) // output layer
        {
            errors[n] = (predicted - targets[n]) * 2 * dp;
        }
        else
        {
            errors[n] = 0.0f;
            for (int k = 0; k < next->numNeurons; k++)
            {
                errors[n] += next->errors[k] * next->weights[k][n];
            }
        }
        errors[n] *= dp;
        error += errors[n] * errors[n];
    }

    UpdateWeightsAndBiases(previous, learningRate);
    return error;
}


// ---------------------------- Relu ----------------------------

hhReluLayer::hhReluLayer(int numNeurons, int numInputs) : hhDenseLayer(numNeurons, numInputs)
{
    // Create a random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    // Initialize weights
    for (int n = 0; n < numNeurons; n++) 
    {
        for (int j = 0; j < numInputs; j++) 
        {
            weights[n][j] = distribution(generator);
        }
        biases[n] = distribution(generator);
    }
}

void hhReluLayer::Forward(const column& input)
{
    assert(input.size() == numInputs);
    for (int n = 0; n < numNeurons; n++)
    {
        float raw = std::inner_product(input.begin(), input.end(), weights[n].begin(), 0.0f) + biases[n];
        activationValue[n] = std::max(0.0f, raw);
    }
}

float hhReluLayer::Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets)
{
    float error = 0.0f;
    for (int i = 0; i < numNeurons; i++)
    {
        const float predicted = activationValue[i];
        if (next == nullptr) // output layer
        {
            errors[i] = (targets[i] - predicted) * (predicted > 0.0f ? 1.0f : 0.0f);
        }
        else
        {
            errors[i] = 0.0f;
            for (int j = 0; j < next->numNeurons; j++)
            {
                errors[i] += next->errors[j] * next->weights[j][i];
            }
            errors[i] *= (predicted > 0.0f ? 1.0f : 0.0f);
        }
    }
    UpdateWeightsAndBiases(previous, learningRate);
    return error;
}


// ---------------------------- Softmax ----------------------------

hhSoftmaxLayer::hhSoftmaxLayer(int numNeurons, int numInputs) : hhDenseLayer(numNeurons, numInputs)
{
}

void hhSoftmaxLayer::Forward(const column& input)
{
    float sum = 0.0f;
    for (int i = 0; i < numNeurons; i++)
    {
        float raw = std::inner_product(input.begin(), input.end(), weights[i].begin(), 0.0f) + biases[i];
        activationValue[i] = exp(raw);
        sum += activationValue[i];
    }

    for (int i = 0; i < numNeurons; i++)
    {
        activationValue[i] /= sum;
    }
}

float hhSoftmaxLayer::Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets)
{
     float error = 0.0f;
    for (int i = 0; i < numNeurons; i++)
    {
        const float predicted = activationValue[i];
        if (next == nullptr) // output layer
        {
            errors[i] = (targets[i] - predicted);
        }
        else
        {
            errors[i] = 0.0f;
            for (int j = 0; j < next->numNeurons; j++)
            {
                errors[i] += next->errors[j] * next->weights[j][i];
            }
        }
    }
    
    UpdateWeightsAndBiases(previous, learningRate);
    return error;
}

// ---------------------------- model ----------------------------

hhLayer* hhModel::AddLayer(hhLayerType type, int numNeurons, int numInputs)
{
    if (numNeurons < 1 || numInputs < 0)
        return nullptr;

    hhLayer* layer = nullptr;
    switch (type)
    {
        case hhLayerType::Input:
        {
            if (layers.size() > 0)
                return nullptr;
            layer = new hhInputLayer(numNeurons, numInputs);
            break;
        }

        case hhLayerType::Sigmoid:
        {
            if (layers.size() == 0)
                return nullptr;
            if (layers.back()->numNeurons != numInputs)
                return nullptr;

            layer = new hhSigmoidLayer(numNeurons, numInputs);
            break;
        }

        case hhLayerType::Relu:
        {
            if (layers.size() == 0)
                return nullptr;
            if (layers.back()->numNeurons != numInputs)
                return nullptr;

            layer = new hhReluLayer(numNeurons, numInputs);
            break;
        }

        case hhLayerType::Softmax:
        {
            if (layers.size() == 0)
                return nullptr;
            if (layers.back()->numNeurons != numInputs)
                return nullptr;

            layer = new hhSoftmaxLayer(numNeurons, numInputs);
            break;
        }
        default:
            break;
    }

    layers.push_back(layer);
    return layer;
}

void hhModel::Configure(hhTask& task)
{
    this->task = &task;
    task.Configure(*this);

    for (auto& layer : task.layers)
    {
        AddLayer(layer.type, layer.numNeurons, layer.numInputs);
    }

    indicies.resize(task.inputs.size());
    std::iota(indicies.begin(), indicies.end(), 0);
}

void hhModel::Forward(const column& input)
{
    layers[0]->Forward(input);
    for (int i = 1; i < layers.size(); i++)
    {
        const column& previous = layers[i - 1]->activationValue;
        layers[i]->Forward(previous);
    }
}

float hhModel::Backward(const column& targets)
{
    float error = 0.0f;
    hhLayer* next = nullptr;
    for (size_t i = layers.size() - 1; i > 0; i--)
    {
        const hhLayer& previous = *layers[i - 1];
        const column& useTarget = (next == nullptr) ? targets : next->activationValue;
        error += layers[i]->Backward(previous, next, task->learningRate, useTarget);
        next = layers[i];
    }
    return error;
}

void hhModel::Train()
{
    for (int epoch = 0; epoch < task->epochs; epoch++)
    {
        bool first = true;
        float error = 0.0f;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indicies.begin(), indicies.end(), g);

        const int numItems = task->batchSize > 0 ? task->batchSize : int(task->inputs.size());
        for (int i=0; i < numItems; i++)
        {
            Forward(task->inputs[indicies[i]]);
            error += Backward(task->targets[indicies[i]]);

            if (first && epoch == task->epochs - 1)
            {
                first = false;
                lastTrainError = error;
                if (numEpochs % 1000 == 0)
                    printf("epoch %d, error %f\n", numEpochs, error);
            }
            numEpochs++;
        }
    }
}

const column& hhModel::Predict(const column& input)
{
    Forward(input);
    return layers.back()->activationValue;
}

int argmax(const column& values)
{
    int index = 0;
    double highest = -1000;
    for(int i=0; i < values.size(); i++)
    {
        if (values[i] > highest)
        {
            highest = values[i];
            index = i;
        }
    }
    return index;
}

float dotProduct(const column& a, const column& b)
{
    assert(a.size() == b.size());

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}