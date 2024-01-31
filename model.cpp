#include "model.h"

#include <numeric>
#include <cmath>

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
    weights.resize(numNeurons, column(numInputs, 0.0f));
    biases.resize(numNeurons, 0.0f);
}

void hhDenseLayer::UpdateWeightsAndBiases(const hhLayer& previous, float learningRate)
{
    for (int i = 0; i < numNeurons; i++)
    {
        for (int j = 0; j < numInputs; j++)
        {
            weights[i][j] -= learningRate * previous.activationValue[j] * errors[i];
        }
        biases[i] -= learningRate * errors[i];
    }
}

// ---------------------------- Sigmoid ----------------------------

hhSigmoidLayer::hhSigmoidLayer(int numNeurons, int numInputs) : hhDenseLayer(numNeurons, numInputs)
{
}

void hhSigmoidLayer::Forward(const column& input)
{
    for (int i = 0; i < numNeurons; i++)
    {
        float raw = std::inner_product(input.begin(), input.end(), weights[i].begin(), 0.0f) + biases[i];
        activationValue[i] = 1.0f / (1.0f + exp(-raw));
    }
}

float hhSigmoidLayer::Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets)
{
    float error = 0.0f;
    for (int i = 0; i < numNeurons; i++)
    {
        const float predicted = activationValue[i];
        if (next == nullptr) // output layer
        {
            errors[i] = (targets[i] - predicted) * predicted * (1.0f - predicted);
        }
        else
        {
            errors[i] = 0.0f;
            for (int j = 0; j < next->numNeurons; j++)
            {
                errors[i] += next->errors[j] * next->weights[j][i];
            }
            errors[i] *= predicted * (1.0f - predicted);
        }
        error += errors[i] * errors[i];
    }

    UpdateWeightsAndBiases(previous, learningRate);
    return error;
}


// ---------------------------- Relu ----------------------------

hhReluLayer::hhReluLayer(int numNeurons, int numInputs) : hhDenseLayer(numNeurons, numInputs)
{
}

void hhReluLayer::Forward(const column& input)
{
    for (int i = 0; i < numNeurons; i++)
    {
        float raw = std::inner_product(input.begin(), input.end(), weights[i].begin(), 0.0f) + biases[i];
        activationValue[i] = std::max(0.0f, raw);
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
    hhLayer* layer = nullptr;
    switch (type)
    {
    case hhLayerType::Input:
        layer = new hhInputLayer(numNeurons, numInputs);
        break;
    case hhLayerType::Sigmoid:
        layer = new hhSigmoidLayer(numNeurons, numInputs);
        break;
    case hhLayerType::Relu:
        layer = new hhReluLayer(numNeurons, numInputs);
        break;
    case hhLayerType::Softmax:
        layer = new hhSoftmaxLayer(numNeurons, numInputs);
        break;
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
        for (int i = 0; i < task->inputs.size(); i++)
        {
            Forward(task->inputs[i]);
            error += Backward(task->targets[i]);

            if (first && epoch == task->epochs - 1)
            {
                first = false;
                lastTrainError = error;
                printf("epoch %d, error %f\n", numEpochs, error);
            }
            numEpochs++;
        }
    }
}

float hhModel::Predict(const column& input)
{
    Forward(input);
    return layers.back()->activationValue[0];
}