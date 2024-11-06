using Microsoft.Extensions.ML;
using WineClusteringEvaluation;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();

// Register the Prediction Engine Pool
builder.Services.AddPredictionEnginePool<WineData, ClusterPrediction>()
        .FromFile(@"C:\Users\stere\source\repos\Lab6\wineClusteringModel.zip");  // Path to the trained model

// Configure Swagger for API documentation
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
