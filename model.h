
#pragma once

#include "task.h"

class hhLayer
{
public:
    virtual ~hhLayer() = default;


};

class hhModel
{
public:

    void Configure(hhTask& task)
    {
        this->task = &task;
        task.Configure(*this);
    }

    hhTask* task = nullptr;

    int numEpochs = 0;
    float lastTrainError = 0;
    float lastTrainTime = 0;

    std::vector<hhLayer*> layers;

};


