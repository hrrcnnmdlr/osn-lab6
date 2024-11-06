using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using WineClusteringEvaluation;

namespace WineClusteringApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class WineClusteringController : ControllerBase
    {
        private readonly PredictionEnginePool<WineData, ClusterPrediction> _predictionEnginePool;

        public WineClusteringController(PredictionEnginePool<WineData, ClusterPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [HttpPost("predict")]
        public ActionResult<ClusterPrediction> PredictCluster([FromBody] WineData wineData)
        {
            // Make prediction using the loaded model
            var prediction = _predictionEnginePool.Predict(wineData);

            // Return the predicted cluster id
            return Ok(prediction);
        }
    }
}
