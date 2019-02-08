using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Azure.CognitiveServices.Language.TextAnalytics.Models;

namespace MicrosoftTextAnalytics
{
    class Program
    {
        private static TextAnalyticsService s_analyticsService;

        static void Main( string[] args )
        {
            var msApiSubscriptionKey = ""; //TODO: Put your own key here.
            var azureRegion = AzureRegions.Southcentralus; 
            s_analyticsService = new TextAnalyticsService(msApiSubscriptionKey, azureRegion);

            var textSourceDirectory = @"C:\Users\alam\Federation of State Medical Boards\Information Services - BACE\Sexual Board Orders\";
            var outputPath = @"C:\temp\MY_BACE\Key Phrases\";
            AnalyzeFolderDocuments(textSourceDirectory, outputPath);
        }


        private static void AnalyzeFolderDocuments(string folderPath, string outputPath)
        {
            var batchSize = 20;
            var files = Directory.EnumerateFiles(folderPath, "*.txt", SearchOption.AllDirectories);

            //Break down file list into batches of batchSize
            var filesBatched = files.Select(( id, index ) => new { id, index }).GroupBy(x => x.index / batchSize).
                Select(g => g.Select(x => x.id));
            foreach (var batch in filesBatched)
            {

                //Input we will send to Microsoft's text analytics API as a List 
                var batchInput = new List<MultiLanguageInput>();
                
                var batchRequest = new List<TextAnalyticsRequest>();

                foreach (var file in batch)
                {
                    //Record initial file information for each file to be analyzed
                    var request = new TextAnalyticsRequest() {
                        FileName = Path.GetFileNameWithoutExtension(file),
                        FolderName = Path.GetFileName(Path.GetDirectoryName(file)),
                        FileText = File.ReadAllText(file)
                    };

                    //Break down file text into chunks of 5000 chars as Microsoft's API only accepts inputs with 5000 char lengths
                    var fileChunks = GetTextChunks(request.FileText);
                    var chunkNum = 0;
                    foreach (var chunkText in fileChunks)
                        batchInput.Add(new MultiLanguageInput("en", $"{request.FileName}_{++chunkNum}", chunkText));
                    request.NumberOfChunks = chunkNum;

                    batchRequest.Add(request);
                }


                if (batchRequest.Count() == 0)
                    continue;
                ////var entityBatchResults = s_analyticsService.GetEntities(batchInput, CancellationToken.None).Result;
                ////var sentimentBatchResults = s_analyticsService.GetSentiment(batchInput, CancellationToken.None).Result;

                //Focusing more on key phrases than entity or sentiment api endpoints
                    var keyPhraseBatchResults = s_analyticsService.GetKeyPhrases(batchInput, CancellationToken.None).Result;

                foreach (var request in batchRequest)
                {
                    //Break down file information with results differently for saving to DB or another file
                    var completeResult = new TextAnalysisRecord() {
                        RequestInfo = request,
                        Fid = Regex.Match(request.FileName, @"FID(\d{9})_?").Groups[1].Value,
                        OrderNumber = Int32.Parse(Regex.Match(request.FileName, @"_OD(\d+)").Groups[1].Value),
                        //Entities = entityBatchResults.Documents.Where(x => x.Id == request.FileName)?.Entities,
                        //EntityErrors = entityBatchResults.Errors.Where(e => e.Id == request.FileName).Select(e => e.Message),
                        //Sentiment = sentimentBatchResults.Documents.Where(x => x.Id == request.FileName)?.Score,
                        //SentimentErrors = sentimentBatchResults.Errors.Where(e => e.Id == request.FileName).Select(e => e.Message),
                        KeyPhrases = keyPhraseBatchResults.Documents.Where(x => x.Id.StartsWith(request.FileName)).SelectMany(x => x.KeyPhrases).ToList(),
                        KeyPhraseErrors = keyPhraseBatchResults.Errors.Where(e => e.Id.StartsWith(request.FileName)).Select(e => e.Message)

                    };

                    var path = Path.Combine(outputPath, $"{request.FolderName}\\{request.FileName}.json");
                    //Save the record to a json file for review
                    completeResult.SaveRecordJsonFile(path);
                }
            }
        }

        private static IEnumerable<string> GetTextChunks( string str, int chunkSize = 5000 )
        {
            for (int i = 0; i < str.Length; i += chunkSize)
                yield return str.Substring(i, Math.Min(chunkSize, str.Length - i));
        }
    }
}
