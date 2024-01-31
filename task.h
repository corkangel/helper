#include "utils.h"

class hhModel;

enum class hhLayerType
{
    None,
    Input,
    Sigmoid,
    Relu,
    Softmax,
};

struct hhTaskLayer
{
    hhLayerType type;
    int numNeurons;
    int numInputs;
};

class hhTask
{
public:

    virtual ~hhTask() = default;

    void AddLayer(hhLayerType type, int numNeurons, int numInputs)
    {
        layers.push_back({type, numNeurons, numInputs});
    }

    matrix inputs;
    matrix targets;
    
    float learningRate;
    int epochs;
    int batchSize;
    std::vector<hhTaskLayer> layers;

    virtual void Configure(hhModel& model) = 0;
    virtual void Render(hhModel& model) {};    
};

