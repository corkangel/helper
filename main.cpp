#include <iostream>

#include "model.h"


class ColorTask : public hhTask
{
public:

    void Configure(hhModel& model) override
    {
        learningRate = 0.01f;
        epochs = 10;
        batchSize = 0;
        inputs = {
            {.1f, .1f},
            {.2f, .4f},
            {.9f, .5f},
            {.5f, .9f}};

        targets = {
            {.0f, .0f, .0f},         
            {.1f, .1f, .97f},                 
            {.1f, .9f, .97f},
            {.9f, .1f, .1f }};

        AddLayer(hhLayerType::Input, 2, 0);
        AddLayer(hhLayerType::Sigmoid, 4, 2);
        AddLayer(hhLayerType::Sigmoid, 3, 4);
    }

    void Render(hhModel& model) override
    {

    }
};


int main(int, char**)
{
    ColorTask task;
    hhModel model;
    model.Configure(task);

    for (int i=0; i < 10000; i++)
    {
        model.Train();
        if (model.lastTrainError < 0.001f)
        {
            break;
        }
    }
}
