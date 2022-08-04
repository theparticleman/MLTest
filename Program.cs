using Microsoft.ML;
using Microsoft.ML.Data;

// const string pathToModel = @"C:\Users\Jon\Downloads\Portuguese_Writing_2PL_080322.zip";
// const string pathToModel = @"C:\Users\Jon\models\2022_08_04_09_01_16\Model.zip";
// const string pathToModel = @"C:\Users\Jon\models\2022_08_04_11_46_10\OnlineGradientDescentModel.zip";
const string pathToModel = @"C:\Users\Jon\models\2022_08_04_15_29_41\OnlineGradientDescentModel.zip";
// const string pathToModel = @"C:\Users\Jon\models\2022_08_04_10_53_54\FastTreeModel.zip";
float theta = -3.808434f;
const float thetaSE = -5.6102382f;
Random rand = new Random();

var mlContext = new MLContext();
var trainedModel = mlContext.Model.Load(pathToModel, out var inputSchema);

var sample = new OpiData
{
    theta = theta,
    thetaSE = thetaSE
};

var predictionFunction = mlContext.Model.CreatePredictionEngine<OpiData, OpiPrediction>(trainedModel);

while (theta < 2.5f)
{
    var prediction = predictionFunction.Predict(sample);
    var transformedScore = LinearTransform(prediction.opi);

    Console.WriteLine($"theta: {theta}, thetaSE: {thetaSE}, prediction: {prediction.opi}, transformed: {transformedScore}");
    // Console.WriteLine($"prediction: {prediction.opi}");

    theta += rand.NextSingle() / 4;

    sample.theta = theta;
}

float LinearTransform(float x)
{
    const float b = 11.0f;
    const float a = 0.1f;

    const float minX =  -2.9895396f;
    const float maxX = 8.763837f;

    return (b - a) * ((x - minX) / (maxX - minX)) + a;
}


public class OpiData
{
    [LoadColumn(0)]
    public float opi = 1;

    [LoadColumn(1)]
    public float theta;

    [LoadColumn(2)]
    public float thetaSE;


}

public class OpiPrediction
{
    [ColumnName("Score")]
    public float opi;
}