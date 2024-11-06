using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.IO;

namespace WineClusteringEvaluation
{
    public class WineData
    {
        [LoadColumn(0)] public float Alcohol { get; set; }
        [LoadColumn(1)] public float Malic_Acid { get; set; }
        [LoadColumn(2)] public float Ash { get; set; }
        [LoadColumn(3)] public float Ash_Alcanity { get; set; }
        [LoadColumn(4)] public float Magnesium { get; set; }
        [LoadColumn(5)] public float Total_Phenols { get; set; }
        [LoadColumn(6)] public float Flavanoids { get; set; }
        [LoadColumn(7)] public float Nonflavanoid_Phenols { get; set; }
        [LoadColumn(8)] public float Proanthocyanins { get; set; }
        [LoadColumn(9)] public float Color_Intensity { get; set; }
        [LoadColumn(10)] public float Hue { get; set; }
        [LoadColumn(11)] public float OD280 { get; set; }
        [LoadColumn(12)] public float Proline { get; set; }
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")] public uint PredictedClusterId { get; set; }
        public float[] Score { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            string dataPath = @"C:\Users\stere\source\repos\Lab6\wine-clustering.csv";
            string modelPath = "wineClusteringModel.zip";

            // Load the dataset
            IDataView data = mlContext.Data.LoadFromTextFile<WineData>(path: dataPath, hasHeader: true, separatorChar: ',');

            // Define feature columns for clustering
            string[] featureColumns = new[]
            {
                "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
                "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols", "Proanthocyanins",
                "Color_Intensity", "Hue", "OD280", "Proline"
            };

            // Define the pipeline for training
            var pipeline = mlContext.Transforms
                .Concatenate("Features", featureColumns)
                .Append(mlContext.Clustering.Trainers.KMeans(
                    featureColumnName: "Features", numberOfClusters: 3));

            // Train the model
            var model = pipeline.Fit(data);

            // Save the model
            mlContext.Model.Save(model, data.Schema, modelPath);
            Console.WriteLine($"Model saved to: {modelPath}");
        }
    }
}
