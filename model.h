
#pragma once

#include "task.h"

class hhLayer
{
public:
    virtual ~hhLayer() = default;

    hhLayer(const int numNeurons, const int numInputs);

    virtual void Forward(const column& input) = 0;
    virtual float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) { return 0.0f;}

    int numNeurons;
    int numInputs;

    column activationValue;
    column errors;
    column biases;
    matrix weights;    
};

class hhInputLayer : public hhLayer
{
public:
    hhInputLayer(int numNeurons, int numInputs);

    void Forward(const column& input) override;
};

class hhDenseLayer : public hhLayer
{
public:
    hhDenseLayer(int numNeurons, int numInputs);

    void UpdateWeightsAndBiases(const hhLayer& previous, float learningRate);
};

class hhSigmoidLayer : public hhDenseLayer
{
public:
    hhSigmoidLayer(int numNeurons, int numInputs);
    void Forward(const column& input) override;
    float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) override;
};

class hhReluLayer : public hhDenseLayer
{
public:
    hhReluLayer(int numNeurons, int numInputs);
    void Forward(const column& input) override;
    float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) override;
};

class hhSoftmaxLayer : public hhDenseLayer
{
public:
    hhSoftmaxLayer(int numNeurons, int numInputs);
    void Forward(const column& input) override;
    float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) override;
};

class hhModel
{
public:

    void Configure(hhTask& task);
    hhLayer* AddLayer(hhLayerType type, int numNeurons, int numInputs);

    void Forward(const column& input);
    float Backward(const column& targets);
    void Train();

    float Predict(const column& input);

    hhTask* task = nullptr;

    int numEpochs = 0;
    float lastTrainError = 0;
    float lastTrainTime = 0;

    std::vector<hhLayer*> layers;

};


