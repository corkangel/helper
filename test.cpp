#include <cassert>

#include "model.h"

bool nothing()
{
    hhModel m;
    return true;
}

bool layers()
{
    {
        // no input layers
        hhModel m;
        assert(m.AddLayer(hhLayerType::Sigmoid, 1, 0) == nullptr); 
    }

    {
        // multiple input layers
        hhModel m;
        m.AddLayer(hhLayerType::Input, 2, 0);
        m.AddLayer(hhLayerType::Input, 2, 0);
    }

    {
        hhModel m;
        m.AddLayer(hhLayerType::Input, 2, 0);
        m.AddLayer(hhLayerType::Sigmoid, 2, 2);

        // negative numNeurons
        assert(m.AddLayer(hhLayerType::Relu, -1, 0) == nullptr);
        assert(m.AddLayer(hhLayerType::Relu, 2, -1) == nullptr);

        // mismatched previousLayer
        assert(m.AddLayer(hhLayerType::Relu, 2, 3) == nullptr);        
    }

    return true;
}

const matrix simpleInputs = {
    {.1f, .1f},
    {.5f, .5f}
};

const matrix simpleTargets = { {1}, {5} };

void initSimpleModel(hhModel& m, hhLayerType lt)
{
    m.AddLayer(hhLayerType::Input, 2, 0);
    m.AddLayer(lt, 1, 2);
    m.AddLayer(lt, 1, 1);
}

const int softmaxTestSize = 6;
const column softmaxTargets = { {.2f, 0.5f, 0.0f, 0.0f, 0.1f, .3f} };


bool predict()
{
    // run the prediction code without any training.

    {
        hhModel m;
        initSimpleModel(m, hhLayerType::Sigmoid);

        column out1(1), out2(1);
        out1 = m.Predict(simpleInputs[0]);
        out2 = m.Predict(simpleInputs[0]);
        assert(out1[0] == out2[0]);
    }

    {
        hhModel m;
        initSimpleModel(m, hhLayerType::Relu);

        column out1(1), out2(1);
        out1 = m.Predict(simpleInputs[0]);
        out2 = m.Predict(simpleInputs[0]);
        assert(out1[0] == out2[0]);
    }

    {
        hhModel m;
        m.AddLayer(hhLayerType::Input, 2, 0);
        m.AddLayer(hhLayerType::Relu, 1, 2);
        m.AddLayer(hhLayerType::Softmax, softmaxTestSize, 1);

        column out1(softmaxTestSize), out2(softmaxTestSize);
        out1 = m.Predict(simpleInputs[0]);
        out2 = m.Predict(simpleInputs[0]);
        assert(out1[0] == out2[0]);
        assert(argmax(out1) == argmax(out2));

        double total = 0;
        for (auto o : out1) total+= o;
        assert(pow((total-1),2) < 0.0001);
    }

    
    // test individual neuron activation function value
    {
        hhInputLayer il(2, 0);
        hhSigmoidLayer dl(1, 2);

        dl.biases = {0.3f};
        dl.weights[0][0] = 0.23f;
        dl.weights[0][1] = -0.1f;

        column inputs(2);
        inputs[0] = 1;
        inputs[1] = 0;

        dl.Forward(inputs);

        const double expectedActivationValue = 0.6295f;
        assert(abs(dl.activationValue[0]-expectedActivationValue) < 0.001);
    }

    {
        hhInputLayer il(1, 0);
        hhSigmoidLayer dl(1, 1);
        hhSigmoidLayer ol(1, 1);

        dl.biases = {0.3f};
        dl.weights[0][0] = 0.3f;

        column inputs(1);
        inputs[0] = 1;

        column targets(1);
        targets[0] = 0;

        dl.Forward(inputs);
        
        ol.Backward(dl, nullptr, 0.5, targets);
        dl.Backward(il, &ol, 0.5, ol.activationValue);

    }

    return true;
}

class TestTask : public hhTask
{
    public:
    void Configure(hhModel& model) override
    {
        learningRate = 0.1f;
        epochs = 1;
        batchSize = 0;
        inputs = simpleInputs;
        targets = simpleTargets;
        AddLayer(hhLayerType::Input, 2, 0);
        AddLayer(hhLayerType::Sigmoid, 1, 2);
        AddLayer(hhLayerType::Sigmoid, 1, 1);
    }
};

bool backwards()
{
    {
        hhModel m;
        TestTask t;
        m.Configure(t);

        column out1(1);

        out1 = m.Predict(simpleInputs[0]);
        double loss1 = m.Backward(simpleTargets[0]);

        out1 = m.Predict(simpleInputs[1]);
        double loss2 = m.Backward(simpleTargets[0]);

        assert(loss2 != loss1);
    }

    return true;
}

// // ------------------------------ float->int  test  ------------------------------

// // doubles and their corresponding integers

// const uint32 numbersBatchSize = 20;
// matrix numbersBatchDoubles;
// matrix numbersBatchIntegers;
// matrix numbersTestDoubles;
// matrix numbersTestIntegers;

// void initNumbers()
// {
//     srand(10910910);
//     numbersBatchDoubles.resize(numbersBatchSize);
//     numbersBatchIntegers.resize(numbersBatchSize);
//     numbersTestDoubles.resize(numbersBatchSize);
//     numbersTestIntegers.resize(numbersBatchSize);

//     for (uint32 i=0; i < numbersBatchSize; i++)
//     {
//         numbersBatchDoubles[i].resize(1);
//         numbersBatchDoubles[i][0] = (rand() % 100) * 0.1;

//         numbersBatchIntegers[i].resize(1);
//         numbersBatchIntegers[i][0] = (double)(uint32)numbersBatchDoubles[i][0];

//         numbersTestDoubles[i].resize(1);
//         numbersTestDoubles[i][0] = (rand() % 100) * 0.1;

//         numbersTestIntegers[i].resize(1);
//         numbersTestIntegers[i][0] = (double)(uint32)numbersTestDoubles[i][0];
//     }
// }


// double trainNumbers(model& m, uint32 numEpochs)
// {
//     m.Train(numbersBatchDoubles, numbersBatchIntegers, numEpochs, 0.02);

//     //printf("trainNumbers loss: %f\n", m.loss);

//     double loss = 0;
//     column predictions(1);
//     for (uint32 i=0; i < numbersBatchSize; i++)
//     {
//         m.PredictSingleInput(numbersTestDoubles[i], predictions);

//         loss += pow(predictions[0] - numbersTestIntegers[i][0], 2);
//     }

//     return loss / numbersBatchSize;
// }

// bool numbers()
// {
//     initNumbers();

//     hhModel m;

//     layer* l = m.AddInputLayer(1);
//     l = m.AddDenseLayer(3, ActivationFunction::Sigmoid, l);
//     l = m.AddDenseLayer(1, ActivationFunction::Relu, l);

//     for (uint32 i=0; i <20; i++)
//     {
//         double loss = trainNumbers(m, 5);
//         printf("train numbers loss: %f %f\n", loss, m.loss);
//     }
//     return true;
// }


// ------------------------------ seeds test ------------------------------

const matrix seedsDataset =
{
    {2.7810836f, 2.550537003f},
    {1.465489372f, 2.362125076f},
    {3.396561688f, 4.400293529f},
    {1.38807019f, 1.850220317f},
    {3.06407232f, 3.005305973f},
    {7.627531214f, 2.759262235f},
    {5.332441248f, 2.088626775f},
    {6.922596716f, 1.77106367f},
    {8.675418651f, -0.242068655f},
    {7.673756466f, 3.508563011f}
};

// all the outputs are the same?!
const matrix seedsOutputs = { 
    {1,0},
    {1,0},
    {1,0},
    {1,0},
    {1,0},
    {0,1},
    {0,1},
    {0,1},
    {0,1},
    {0,1}
};


class SeedTask : public hhTask
{
    public:
    void Configure(hhModel& model) override
    {
        learningRate = 0.5f;
        epochs = 10;
        batchSize = 0;
        inputs = seedsDataset;
        targets = seedsOutputs;
        AddLayer(hhLayerType::Input, 2, 0);
        AddLayer(hhLayerType::Sigmoid, 1, 2);
        AddLayer(hhLayerType::Sigmoid, 2, 1);
    }
};

bool seeds()
{
    hhModel m;
    SeedTask t;
    m.Configure(t);

    hhLayer& output = *m.layers[2];

    for (int i = 0; i < 20; i++)
    {
        m.Train();
        //printf("seeds -  loss sum:%f act:%f grad:%f err:%f\n", m.lastTrainError, output.activationValue[0], output.errors[0],  output.errors[0]);
    }

    return true;
}

void check(const char* name, const int result)
{
    printf("TEST: [%-12s] %s\n", name, result ? "success" : "fail");
}

int main(int, char**)
{
    printf("tests begin\n");
    check("nothing", nothing());
    check("layers", layers());
    check("predict", predict());
    check("backwards", backwards());
    //check("numbers", numbers());
    check("seeds", seeds());
    printf("tests end\n");
    return 1;
}