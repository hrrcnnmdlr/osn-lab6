namespace Lab6
{
    using Microsoft.ML;
    using WineClusteringEvaluation;

    public class MLModelEngine
    {
        private static Lazy<MLModelEngine> _instance = new Lazy<MLModelEngine>(() => new MLModelEngine());
        public static MLModelEngine Instance => _instance.Value;

        private readonly ITransformer _model;
        private readonly MLContext _mlContext;

        private MLModelEngine()
        {
            _mlContext = new MLContext();
            _model = _mlContext.Model.Load("wineClusteringModel.zip", out _);
        }

        public ClusterPrediction Predict(WineData input)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<WineData, ClusterPrediction>(_model);
            return predictionEngine.Predict(input);
        }
    }
}
