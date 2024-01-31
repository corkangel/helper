#include <iostream>

#include "model.h"
#include "render.h"

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
        for (int y=0; y < 20; y++)
        {
            for (int x=0; x < 20; x++)
            {
                column input = {x / 20.0f, y / 20.0f};
                model.Predict(input);
                const auto& output = model.layers.back()->activationValue;
                printf("(%d, %d) -> (%.2f, %.2f, %.2f)\n", x, y, output[0], output[1], output[2]);
            }
        }
    }
};


int main(int, char**)
{
    ColorTask task;
    hhModel model;
    model.Configure(task);

    renderWindow rw;

    const int gridSize = 80;
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
