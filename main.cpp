#include <iostream>

#include "model.h"
#include "render.h"

class ColorTask : public hhTask
{
public:

    void Configure(hhModel& model) override
    {
        learningRate = 0.04f;
        epochs = 20;
        batchSize = 0;
        inputs = {
            {.1f, .1f},
            {.8f, .7f},
            {.7f, .1f},
            {.5f, .2f},
            {.2f, .1f}};

        targets = {
            {.9f, .0f, .0f},         
            {.9f, .0f, .9f},         
            {.0f, .0f, .97f},                 
            {.1f, .9f, .7f},                 
            {.0f, .9f, .1f }};

        AddLayer(hhLayerType::Input, 2, 0);
        AddLayer(hhLayerType::Sigmoid, 9, 2);
        AddLayer(hhLayerType::Sigmoid, 3, 9);
    }
};


int main(int, char**)
{
    ColorTask task;
    hhModel model;
    model.Configure(task);

    renderWindow rw;

    const int gridSize = 40;
    matrix outs(gridSize*gridSize);
    for (auto&& o : outs)
        o.resize(3, 0); // (r,g,b)
    
    bool running = 1;
    while (running)
    {
        model.Train();

        for (int y=0; y < gridSize; y++)
        {
            for (int x=0; x < gridSize; x++)
            {
                column ins(2); // (x,y)
                ins[0] = x*1.0f/gridSize;
                ins[1] = y*1.0f/gridSize;
                model.Predict(ins);
                outs[y*gridSize + x] = model.layers.back()->activationValue;           
            }
        }

        rw.ProcessEvents(running);

        rw.BeginDisplay();
        rw.DisplayTitle(model.numEpochs, model.lastTrainError, "Helper");
        rw.DisplayGrid(gridSize, outs);
        rw.EndDisplay();
    }
}
